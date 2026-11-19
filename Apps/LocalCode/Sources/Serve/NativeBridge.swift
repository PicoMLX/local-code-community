#if PICO_NATIVE_LOCAL_CODE
import Atomics
import Foundation
import LLM
import NNC

public typealias PicoLocalCodeEventCallback =
  @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<CChar>?) -> Int32

/// The version of `pico_local_code_generate_v1`'s contract. The token-level entry points at the
/// end of this file carry their own version in their names; a loader finds them by symbol.
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

/// Held while any model generates, through the chat entry point or a token-level handle.
/// Generation saves and restores process-wide graph settings, such as the queue watermark, so two
/// generations that overlap, even on different models, can leave those settings changed.
private let inferenceGate = NSLock()

private final class NativeGenerationRuntime: @unchecked Sendable {
  static let shared = NativeGenerationRuntime()
  private let lock = NSLock()
  private var session: Session?

  fileprivate struct Files: Equatable {
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
    guard inferenceGate.try() else {
      transport.fail(code: "engine_busy", message: "Local Code is already generating.")
      return 2
    }
    defer { inferenceGate.unlock() }
    lock.lock()
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

// MARK: - Token-level entry points

/// Receives generated token IDs, borrowed for the duration of the call. A call with no tokens
/// reports that the request is still running. Return 0 to cancel the request.
public typealias PicoLocalCodeTokenCallback =
  @convention(c) (UnsafeMutableRawPointer?, UnsafePointer<Int32>?, Int32) -> Int32

/// One DeepSeek 4.1 model. Opening is cheap: the weights load on the first generation.
private final class NativeTokenSession {
  /// Held while the model generates, through either entry point.
  let lock = NSLock()
  let graph = DynamicGraph()
  let stream = StreamContext(.GPU(0))
  var generator: DeepSeek4_1TextGenerator
  /// Conversations between chat requests. Each agent holds the KV cache of the prompt it last
  /// evaluated, so a request that goes on with its conversation evaluates only what is new.
  /// Guarded by `lock`.
  var idleAgents = ResponsesAgentPool<ResponsesAgent>()
  /// Lays out the chat requests' prompts. Each agent has an adapter of its own, which holds its
  /// caches, so this one is free while a request generates.
  let promptAdapter: DeepSeekResponsesModelAdapter<DeepSeek4_1TextGenerator>
  let promptLock = NSLock()

  init(files: NativeGenerationRuntime.Files) {
    generator = DeepSeek4_1TextGenerator(
      filePath: files.checkpoint, engramFilePath: files.engram, drafterFile: files.drafter)
    graph.maxConcurrency = .limit(4)
    promptAdapter = DeepSeekResponsesModelAdapter(textGenerator: generator, projectPath: nil)
  }

  func makeAgent() -> ResponsesAgent {
    ResponsesAgent(
      adapter: DeepSeekResponsesModelAdapter(textGenerator: generator, projectPath: nil),
      graph: graph, streamContext: stream, timeSlice: 1, activeTurnCount: ManagedAtomic(1))
  }
}

private func copy(_ message: String, to buffer: UnsafeMutablePointer<CChar>?, capacity: Int32) {
  guard let buffer, capacity > 0 else { return }
  _ = message.withCString { strlcpy(buffer, $0, Int(capacity)) }
}

/// Returns a model handle, or nil after writing the reason to `error`. `drafter` may be nil.
@_cdecl("pico_local_code_open_v2")
public func picoLocalCodeOpen(
  _ checkpoint: UnsafePointer<CChar>?, _ engram: UnsafePointer<CChar>?,
  _ drafter: UnsafePointer<CChar>?, _ error: UnsafeMutablePointer<CChar>?, _ errorCapacity: Int32
) -> UnsafeMutableRawPointer? {
  var request: [String: Any] = [:]
  request["checkpoint_path"] = checkpoint.flatMap { String(validatingCString: $0) }
  request["engram_path"] = engram.flatMap { String(validatingCString: $0) }
  request["drafter_path"] = drafter.flatMap { String(validatingCString: $0) }
  do {
    let session = NativeTokenSession(files: try NativeGenerationRuntime.Files(request))
    return Unmanaged.passRetained(session).toOpaque()
  } catch let failure {
    copy(failure.localizedDescription, to: error, capacity: errorCapacity)
    return nil
  }
}

/// Unloads the weights and the conversations' KV caches. No request may be running.
@_cdecl("pico_local_code_close_v2")
public func picoLocalCodeClose(_ handle: UnsafeMutableRawPointer?) {
  guard let handle else { return }
  let session = Unmanaged<NativeTokenSession>.fromOpaque(handle).takeRetainedValue()
  session.lock.lock()
  for agent in session.idleAgents.removeAll() { agent.unloadCaches() }
  session.generator.unload()
  session.lock.unlock()
}

/// The bytes the weights hold once loaded, at the tier this machine selects.
@_cdecl("pico_local_code_resident_bytes_v2")
public func picoLocalCodeResidentBytes(_ handle: UnsafeMutableRawPointer?) -> Int64 {
  guard let handle else { return 0 }
  let session = Unmanaged<NativeTokenSession>.fromOpaque(handle).takeUnretainedValue()
  return Int64(clamping: session.generator.residentWeightsSize)
}

/// Writes up to `capacity` token IDs and returns how many `text` has, or -1 for invalid UTF-8.
/// Special tokens written out in `text` become their IDs; nothing is added at either end.
@_cdecl("pico_local_code_tokenize_v2")
public func picoLocalCodeTokenize(
  _ text: UnsafePointer<CChar>?, _ tokens: UnsafeMutablePointer<Int32>?, _ capacity: Int32
) -> Int32 {
  guard let text, let string = String(validatingCString: text) else { return -1 }
  let (ids, _) = DeepSeek4_1TextGenerator.unsafeTokenizer.tokenize(
    text: string, addSpecialTokens: false)
  if let tokens {
    for index in 0..<min(ids.count, Int(max(capacity, 0))) { tokens[index] = ids[index] }
  }
  return Int32(clamping: ids.count)
}

/// `pico_local_code_tokenize_v2` for text given as `length` bytes of UTF-8, which may hold NUL
/// characters.
@_cdecl("pico_local_code_tokenize_v3")
public func picoLocalCodeTokenizeBytes(
  _ text: UnsafePointer<UInt8>?, _ length: Int32, _ tokens: UnsafeMutablePointer<Int32>?,
  _ capacity: Int32
) -> Int32 {
  guard length >= 0, text != nil || length == 0,
    let string = String(
      validating: UnsafeBufferPointer(start: text, count: Int(length)), as: UTF8.self)
  else { return -1 }
  let (ids, _) = DeepSeek4_1TextGenerator.unsafeTokenizer.tokenize(
    text: string, addSpecialTokens: false)
  if let tokens {
    for index in 0..<min(ids.count, Int(max(capacity, 0))) { tokens[index] = ids[index] }
  }
  return Int32(clamping: ids.count)
}

/// Writes the tokens' bytes followed by a NUL, truncated to `capacity`, and returns their length
/// without that NUL. Read them by the returned length: they may hold NUL characters, and they are
/// UTF-8 except where the tokens begin or end inside a character. A caller that decodes a reply in
/// pieces keeps those bytes rather than losing the text around them. Call with a nil buffer first
/// to size it.
@_cdecl("pico_local_code_detokenize_v2")
public func picoLocalCodeDetokenize(
  _ tokens: UnsafePointer<Int32>?, _ count: Int32, _ text: UnsafeMutablePointer<CChar>?,
  _ capacity: Int32
) -> Int32 {
  let ids = tokens.map { Array(UnsafeBufferPointer(start: $0, count: Int(max(count, 0)))) } ?? []
  let bytes = DeepSeek4_1TextGenerator.unsafeTokenizer.decodeBytes(ids)
  if let text, capacity > 0 {
    let written = min(bytes.count, Int(capacity) - 1)
    for index in 0..<written { text[index] = CChar(bitPattern: bytes[index]) }
    text[written] = 0
  }
  return Int32(clamping: bytes.count)
}

/// Generates from raw prompt token IDs with a fresh KV cache, calling `callback` synchronously.
/// `maxTokens` at or below 0 leaves the limit to the context length. `deterministic` nonzero
/// picks the most likely token; zero samples with the model's own settings.
///
/// `usage` receives four values: prompt tokens evaluated, tokens generated, prefill
/// microseconds and decode microseconds.
///
/// Returns 0 when the model stopped, 1 when the callback cancelled, 2 for a failure and 3 when
/// `maxTokens` or the context length ended the request.
///
/// One model generates at a time across the process: a call waits while another handle, or the
/// chat entry point, is generating. A second call on the same handle while one runs fails.
@_cdecl("pico_local_code_generate_tokens_v2")
public func picoLocalCodeGenerateTokens(
  _ handle: UnsafeMutableRawPointer?, _ prompt: UnsafePointer<Int32>?, _ promptCount: Int32,
  _ maxTokens: Int32, _ deterministic: Int32, _ callback: PicoLocalCodeTokenCallback?,
  _ context: UnsafeMutableRawPointer?, _ usage: UnsafeMutablePointer<Int64>?
) -> Int32 {
  let endOfSentence = endOfSentenceToken
  let vocabularySize = Int32(DeepSeek4_1ModelConfiguration.deepSeekV4_1Flash.vocabularySize)
  guard let handle, let prompt, promptCount > 0, let callback else { return 2 }
  let ids = Array(UnsafeBufferPointer(start: prompt, count: Int(promptCount)))
  guard ids.allSatisfy({ (0..<vocabularySize).contains($0) }) else { return 2 }
  let session = Unmanaged<NativeTokenSession>.fromOpaque(handle).takeUnretainedValue()
  guard session.lock.try() else { return 2 }
  defer { session.lock.unlock() }
  inferenceGate.lock()
  defer { inferenceGate.unlock() }

  let limit = maxTokens > 0 ? Int(maxTokens) : Int.max
  var delivered = 0
  var cancelled = false
  // Hands the callback the tokens it has not seen, up to the limit.
  func deliver(_ generated: [Int32]) {
    let end = min(generated.count, limit)
    guard end > delivered, !cancelled else { return }
    let proceed = generated[delivered..<end].withUnsafeBufferPointer {
      callback(context, $0.baseAddress, Int32($0.count))
    }
    delivered = end
    cancelled = proceed == 0
  }

  var kvCache = DeepSeek4_1TextGenerator.KVCache()
  // A nil project path gives the generator no readable or writable context cache.
  let contextCache = DeepSeek4_1ContextCache(
    projectPath: nil, checkpointStride: session.generator.checkpointStride)
  let graph = session.graph
  let input = graph.variable(Tensor<Int32>(ids, .CPU, .C(ids.count)).toGPU(0))
  var generated = graph.withStream(session.stream) {
    graph.withNoGrad {
      session.generator.decode(
        graph: graph, tokens: input, tokenOffset: 0, thinkingBudget: 0,
        deterministic: deterministic != 0, kvCache: &kvCache, contextTokens: ids,
        contextCache: contextCache
      ) { _, _, generated, _ in
        if generated.count > delivered {
          deliver(generated)
        } else {
          cancelled = callback(context, nil, 0) == 0
        }
        return cancelled || generated.count >= limit ? .abort : .continue
      }
    }
  }
  session.stream.joined()

  let stopped = generated.last == endOfSentence
  if stopped { generated.removeLast() }
  deliver(generated)
  let statistics = session.generator.lastInteractionStatistics
  if let usage {
    usage[0] = Int64(statistics.prefillTokensSoFar)
    usage[1] = Int64(delivered)
    usage[2] = Int64(statistics.prefillDurationSoFar * 1_000_000)
    usage[3] = Int64(statistics.tokenGenerationDurationSoFar * 1_000_000)
  }
  if cancelled { return 1 }
  if stopped && generated.count <= limit { return 0 }
  // The decoder returns nothing for a prompt it cannot place in its context.
  return generated.isEmpty ? 2 : 3
}

/// The one token that ends a DeepSeek 4.1 reply.
private let endOfSentenceToken: Int32 = 1

/// `pico_local_code_generate_tokens_v2` with a fifth `usage` value: the ID of the end token that
/// stopped the reply when the result is 0, and -1 otherwise. A chat format can end a reply with
/// more than one kind of token, so the caller is told which.
@_cdecl("pico_local_code_generate_tokens_v3")
public func picoLocalCodeGenerateTokensNamingStop(
  _ handle: UnsafeMutableRawPointer?, _ prompt: UnsafePointer<Int32>?, _ promptCount: Int32,
  _ maxTokens: Int32, _ deterministic: Int32, _ callback: PicoLocalCodeTokenCallback?,
  _ context: UnsafeMutableRawPointer?, _ usage: UnsafeMutablePointer<Int64>?
) -> Int32 {
  let status = picoLocalCodeGenerateTokens(
    handle, prompt, promptCount, maxTokens, deterministic, callback, context, usage)
  usage?[4] = status == 0 ? Int64(endOfSentenceToken) : -1
  return status
}

// MARK: - Chat entry points

/// The most conversations a model keeps KV caches for between chat requests. Each holds the cache
/// of its whole prompt until `pico_local_code_trim_v1` lets it go or a newer one takes its place.
private let maximumIdleConversations = 4

/// A chat request as a turn: its conversation laid out in the model's chat format.
private struct NativeChatTurn {
  let request: ResponsesRequest
  let model: String
  let prompt: ResponsesPrompt
  let thinkingBudget: Int

  init(body: Data, session: NativeTokenSession) throws {
    request = try ResponsesRequest(chatCompletionsBody: body)
    let object = try JSONSerialization.jsonObject(with: body) as? [String: Any]
    guard let model = object?["model"] as? String, !model.isEmpty else {
      throw NativeRequestError.invalid("Set the request's model ID.")
    }
    guard !request.hasImages else {
      throw NativeRequestError.invalid("The native Local Code bridge accepts text input.")
    }
    self.model = model
    switch request.thinking {
    case .disabled?: thinkingBudget = 0
    case .enabled(let budgetOverride)?: thinkingBudget = min(budgetOverride ?? 131_072, 262_144)
    case nil: thinkingBudget = 131_072
    }
    session.promptLock.lock()
    defer { session.promptLock.unlock() }
    prompt = session.promptAdapter.prompt(
      for: request, reasoningEffort: request.reasoningEffort ?? .high, isThinking: thinkingBudget > 0)
  }
}

/// Hands each Chat Completions chunk to the callback as its JSON text, and an error as
/// `{"error": {...}}`.
private final class NativeChatTransport: ResponsesTransport {
  private let callback: PicoLocalCodeEventCallback
  private let context: UnsafeMutableRawPointer?
  private(set) var isDisconnected = false
  /// Whether the response reached its end, `[DONE]`.
  private(set) var didFinish = false
  private(set) var didFail = false

  init(callback: PicoLocalCodeEventCallback, context: UnsafeMutableRawPointer?) {
    self.callback = callback
    self.context = context
  }

  func beginEventStream() {}
  func sendKeepAlive() {}
  func close() {}

  // The bridge streams, so a whole response only ever arrives as a failure.
  func respond(status: Int, contentType: String, body: Data) {
    didFail = true
    deliver(body)
  }

  func sendEvent(name: String?, data: Data) {
    if data == Data("[DONE]".utf8) {
      didFinish = true
      return
    }
    if (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["error"] != nil {
      didFail = true
    }
    deliver(data)
  }

  func fail(code: String, message: String) {
    didFail = true
    let body = ["error": ["type": "invalid_request_error", "code": code, "message": message]]
    deliver((try? JSONSerialization.data(withJSONObject: body)) ?? Data())
  }

  /// Asks the callback, with no chunk, whether the request should go on.
  func poll() {
    guard !isDisconnected else { return }
    isDisconnected = callback(context, nil) == 0
  }

  private func deliver(_ data: Data) {
    guard !isDisconnected else { return }
    String(decoding: data, as: UTF8.self).withCString { string in
      isDisconnected = callback(context, string) == 0
    }
  }
}

/// Keeps what a turn's end says its agent's KV cache holds.
private final class NativeTurnObserver: ResponsesAgentDelegate {
  var retainedPrompt = ResponsesPrompt()

  func responsesAgent(_ agent: ResponsesAgent, didUpdate progress: ResponsesAgent.Progress) {}

  func responsesAgent(_ agent: ResponsesAgent, didFinish summary: ResponsesAgent.Summary) {
    retainedPrompt = summary.retainedPrompt
  }
}

/// Runs one turn of a conversation given as an OpenAI Chat Completions request body: the model's
/// own chat format lays out the messages and tools, and its own parser splits the reply. Calls
/// `callback` synchronously with each streamed `chat.completion.chunk`, the usage chunk last, as
/// JSON text borrowed for the call. A call with nil asks whether the request should go on. Return
/// 0 to cancel. A failure arrives as `{"error": {...}}`.
///
/// A conversation that goes on from an earlier request's reply skips the prompt that request
/// evaluated: the model keeps the KV caches of its last few conversations.
///
/// Returns 0 when the turn finished, 1 when the callback cancelled and 2 for a failure. Requests
/// to one model run one at a time, and one model generates at a time across the process.
@_cdecl("pico_local_code_chat_v1")
public func picoLocalCodeChat(
  _ handle: UnsafeMutableRawPointer?, _ requestJSON: UnsafePointer<CChar>?,
  _ callback: PicoLocalCodeEventCallback?, _ context: UnsafeMutableRawPointer?
) -> Int32 {
  guard let callback else { return 2 }
  let transport = NativeChatTransport(callback: callback, context: context)
  guard let handle, let requestJSON, let body = String(validatingCString: requestJSON) else {
    transport.fail(code: "invalid_request", message: "Pass a model handle and a UTF-8 request.")
    return 2
  }
  let session = Unmanaged<NativeTokenSession>.fromOpaque(handle).takeUnretainedValue()
  let turn: NativeChatTurn
  do {
    turn = try NativeChatTurn(body: Data(body.utf8), session: session)
  } catch ResponsesRequestError.invalid(let message) {
    transport.fail(code: "invalid_request", message: message)
    return 2
  } catch ResponsesRequestError.unsupported(_, let message) {
    transport.fail(code: "unsupported_parameter", message: message)
    return 2
  } catch {
    transport.fail(code: "invalid_request", message: error.localizedDescription)
    return 2
  }

  session.lock.lock()
  defer { session.lock.unlock() }
  inferenceGate.lock()
  defer { inferenceGate.unlock() }

  let reused =
    session.idleAgents.takeAgent(continuing: turn.prompt)
    ?? session.idleAgents.takeAgent(supersededBy: turn.prompt)
  if reused == nil, session.idleAgents.count >= maximumIdleConversations,
    let expired = session.idleAgents.expire()
  {
    expired.unloadCaches()
  }
  let agent = reused ?? session.makeAgent()
  let observer = NativeTurnObserver()
  agent.delegate = observer
  let writer = ChatCompletionsResponseWriter(
    transport: transport, isStreaming: true, includesUsage: true, model: turn.model)
  writer.start()
  agent.runSynchronously(
    ResponsesAgent.Request(
      request: turn.request, prompt: turn.prompt, thinkingBudget: turn.thinkingBudget,
      writer: writer, transport: transport)
  ) { _, _ in
    transport.poll()
    return transport.isDisconnected ? .abort : .continue
  }
  agent.delegate = nil

  // The conversation as the request gave it, without the reply's opening, which is what a later
  // request repeats when it asks again.
  let replyOpening = agent.adapter.replyOpening(isThinking: turn.thinkingBudget > 0)
  let conversation = turn.prompt.prefix(max(0, turn.prompt.tokens.count - replyOpening.count))
  if !observer.retainedPrompt.tokens.isEmpty {
    session.idleAgents.add(
      agent, prompt: observer.retainedPrompt, conversation: conversation, wasReused: reused != nil)
  } else {
    agent.unloadCaches()
  }
  if transport.isDisconnected { return 1 }
  return transport.didFinish && !transport.didFail ? 0 : 2
}

/// The prompt tokens a Chat Completions request holds once laid out in the model's chat format,
/// counting those a cache would supply, or -1 after writing the reason to `error`.
@_cdecl("pico_local_code_count_chat_tokens_v1")
public func picoLocalCodeCountChatTokens(
  _ handle: UnsafeMutableRawPointer?, _ requestJSON: UnsafePointer<CChar>?,
  _ error: UnsafeMutablePointer<CChar>?, _ errorCapacity: Int32
) -> Int32 {
  guard let handle, let requestJSON, let body = String(validatingCString: requestJSON) else {
    copy("Pass a model handle and a UTF-8 request.", to: error, capacity: errorCapacity)
    return -1
  }
  let session = Unmanaged<NativeTokenSession>.fromOpaque(handle).takeUnretainedValue()
  do {
    return Int32(clamping: try NativeChatTurn(body: Data(body.utf8), session: session).prompt.tokens.count)
  } catch ResponsesRequestError.invalid(let message) {
    copy(message, to: error, capacity: errorCapacity)
  } catch ResponsesRequestError.unsupported(_, let message) {
    copy(message, to: error, capacity: errorCapacity)
  } catch let failure {
    copy(failure.localizedDescription, to: error, capacity: errorCapacity)
  }
  return -1
}

/// Lets go of the KV caches the model keeps for its conversations between chat requests, waiting
/// for a running request first. Returns the bytes this process's footprint fell by.
@_cdecl("pico_local_code_trim_v1")
public func picoLocalCodeTrim(_ handle: UnsafeMutableRawPointer?) -> Int64 {
  guard let handle else { return 0 }
  let session = Unmanaged<NativeTokenSession>.fromOpaque(handle).takeUnretainedValue()
  session.lock.lock()
  defer { session.lock.unlock() }
  let agents = session.idleAgents.removeAll()
  guard !agents.isEmpty else { return 0 }
  let before = physicalFootprint()
  for agent in agents { agent.unloadCaches() }
  return max(0, before - physicalFootprint())
}

/// The memory this process holds, as the system charges it, GPU buffers included.
private func physicalFootprint() -> Int64 {
  var info = task_vm_info_data_t()
  var count = mach_msg_type_number_t(
    MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
  let result = withUnsafeMutablePointer(to: &info) {
    $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
      task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
    }
  }
  return result == KERN_SUCCESS ? Int64(info.phys_footprint) : 0
}
#endif
