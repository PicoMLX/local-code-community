#if PICO_NATIVE_LOCAL_CODE
import Atomics
import Foundation
import LLM
import NNC

public typealias PicoLocalCodeEventCallback =
  @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Int32

@_cdecl("pico_local_code_abi_version")
public func picoLocalCodeABIVersion() -> Int32 { 1 }

/// Calls the callback synchronously. Both pointers are borrowed for the duration of the call.
/// Returns 0 for a completed or incomplete response, 1 for cancellation, and 2 for failure.
@_cdecl("pico_local_code_generate_v1")
public func picoLocalCodeGenerate(
  _ requestJSON: UnsafePointer<CChar>?, _ callback: PicoLocalCodeEventCallback?,
  _ context: UnsafeMutableRawPointer?
) -> Int32 {
  guard let callback else { return 2 }
  let transport = NativeResponsesTransport(callback: callback, context: context)
  guard let requestJSON, let request = String(validatingCString: requestJSON) else {
    transport.fail(code: "invalid_request", message: "Pass a UTF-8 Responses request.")
    return 2
  }
  return NativeGenerationRuntime.shared.generate(Data(request.utf8), transport: transport)
}

private enum NativeRequestError: LocalizedError {
  case invalid(String)

  var errorDescription: String? {
    switch self {
    case .invalid(let message): message
    }
  }
}

private final class NativeResponsesTransport: ResponsesTransport {
  private let callback: PicoLocalCodeEventCallback
  private let context: UnsafeMutableRawPointer?
  private(set) var isDisconnected = false
  private(set) var succeeded = false
  private(set) var eventCount = 0

  init(callback: PicoLocalCodeEventCallback, context: UnsafeMutableRawPointer?) {
    self.callback = callback
    self.context = context
  }

  func beginEventStream() {}
  func sendKeepAlive() {}
  func close() {}

  func respond(status: Int, contentType: String, body: Data) {
    let object = try? JSONSerialization.jsonObject(with: body) as? [String: Any]
    let error = object?["error"] as? [String: Any]
    fail(
      code: error?["code"] as? String ?? "generation_failed",
      message: error?["message"] as? String ?? "Generation failed (status \(status)).")
  }

  func sendEvent(name: String?, data: Data) {
    guard !isDisconnected else { return }
    eventCount += 1
    if name == "response.completed" || name == "response.incomplete" {
      succeeded = true
    }
    String(decoding: data, as: UTF8.self).withCString { string in
      isDisconnected = callback(context, string) == 0
    }
  }

  func fail(code: String, message: String) {
    emit("response.failed", fields: [
      "response": ["status": "failed", "error": ["code": code, "message": message]]
    ])
  }

  func emit(_ type: String, fields: [String: Any]) {
    var object = fields
    object["type"] = type
    guard let data = try? JSONSerialization.data(withJSONObject: object) else { return }
    sendEvent(name: type, data: data)
  }
}

private final class NativeGenerationRuntime: @unchecked Sendable {
  static let shared = NativeGenerationRuntime()
  private let lock = NSLock()
  private var session: Session?

  private struct Files: Equatable {
    let checkpoint: String
    let engram: String
    let drafter: String?

    init(_ request: [String: Any]) throws {
      checkpoint = try Self.readableCheckpoint(request["checkpoint_path"], key: "checkpoint_path")
      let engramFile = try Self.readableFile(request["engram_path"], key: "engram_path")
      guard UInt64(engramFile.size) == DeepSeek4_1EngramConfiguration.deepSeekV4_1Flash.fileSize
      else {
        throw NativeRequestError.invalid("The DeepSeek 4.1 Engram file has an unexpected size.")
      }
      engram = engramFile.url.path
      if let value = request["drafter_path"], !(value is NSNull) {
        drafter = try Self.readableCheckpoint(value, key: "drafter_path")
      } else {
        drafter = nil
      }
    }

    private static func readableCheckpoint(_ value: Any?, key: String) throws -> String {
      let file = try readableFile(value, key: key)
      let handle = try FileHandle(forReadingFrom: file.url)
      defer { try? handle.close() }
      guard try handle.read(upToCount: 16) == Data("SQLite format 3\0".utf8) else {
        throw NativeRequestError.invalid("The file for \(key) is not an NNC checkpoint.")
      }
      return file.url.path
    }

    private static func readableFile(_ value: Any?, key: String) throws -> (url: URL, size: Int) {
      guard let path = value as? String, path.hasPrefix("/") else {
        throw NativeRequestError.invalid("Set \(key) to an absolute model file path.")
      }
      let file = URL(fileURLWithPath: path).resolvingSymlinksInPath()
      let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
      guard values?.isRegularFile == true, let size = values?.fileSize, size > 0,
        FileManager.default.isReadableFile(atPath: file.path)
      else {
        throw NativeRequestError.invalid("The file for \(key) is missing or unreadable.")
      }
      return (file, size)
    }
  }

  private final class Session {
    let files: Files
    let graph = DynamicGraph()
    let stream = StreamContext(.GPU(0))
    let generator: DeepSeek4_1TextGenerator

    init(files: Files) {
      self.files = files
      generator = DeepSeek4_1TextGenerator(
        filePath: files.checkpoint, engramFilePath: files.engram, drafterFile: files.drafter)
      graph.maxConcurrency = .limit(4)
    }
  }

  func generate(_ data: Data, transport: NativeResponsesTransport) -> Int32 {
    guard lock.try() else {
      transport.fail(code: "engine_busy", message: "Local Code is already generating.")
      return 2
    }
    defer { lock.unlock() }

    do {
      let object = try ResponsesRequest.object(from: data)
      let request = try ResponsesRequest(body: data)
      guard let model = object["model"] as? String, !model.isEmpty else {
        throw NativeRequestError.invalid("Set the Responses request model ID.")
      }
      guard !request.hasImages else {
        throw NativeRequestError.invalid("The native Local Code bridge accepts text input.")
      }
      let files = try Files(object)
      let generationSession: Session
      if let session, session.files == files {
        generationSession = session
      } else {
        generationSession = Session(files: files)
        session = generationSession
      }

      // Each request owns fresh KV and decode state. The generator copy shares the
      // model storage, and nil disables both reading and writing the persistent context cache.
      let adapter = DeepSeekResponsesModelAdapter(
        textGenerator: generationSession.generator, projectPath: nil)
      let agent = ResponsesAgent(
        adapter: adapter, graph: generationSession.graph, streamContext: generationSession.stream,
        timeSlice: 1, activeTurnCount: ManagedAtomic(1))
      let writer = ResponsesResponseWriter(transport: transport, isStreaming: true, model: model)
      writer.start()
      if transport.isDisconnected { return 1 }
      let budget: Int
      switch request.thinking {
      case .disabled?: budget = 0
      case .enabled(let budgetOverride)?: budget = min(budgetOverride ?? 131_072, 262_144)
      case nil: budget = 131_072
      }
      let prompt = adapter.prompt(
        for: request, reasoningEffort: request.reasoningEffort ?? .high, isThinking: budget > 0)
      let turn = ResponsesAgent.Request(
        request: request, prompt: prompt, thinkingBudget: budget, writer: writer,
        transport: transport)
      var lastEventCount = transport.eventCount
      agent.runSynchronously(turn) { progress, _ in
        if transport.isDisconnected { return .abort }
        switch progress {
        case .prefill(_, let finished, let total):
          transport.emit(
            "pico.local_code.progress",
            fields: ["phase": "prefill", "finished": finished, "total": total])
        case .generate where transport.eventCount == lastEventCount:
          transport.emit("pico.local_code.progress", fields: ["phase": "decode"])
        default:
          break
        }
        lastEventCount = transport.eventCount
        return transport.isDisconnected ? .abort : .continue
      }
      if transport.isDisconnected { return 1 }
      return transport.succeeded ? 0 : 2
    } catch ResponsesRequestError.invalid(let message) {
      transport.fail(code: "invalid_request", message: message)
    } catch ResponsesRequestError.unsupported(_, let message) {
      transport.fail(code: "unsupported_parameter", message: message)
    } catch {
      transport.fail(code: "invalid_request", message: error.localizedDescription)
    }
    return transport.isDisconnected ? 1 : 2
  }
}
#endif
