import CoreText
import XCTest

@testable import SwiftMath

final class MTWidthConstrainedLayoutTests: XCTestCase {
  private let font = MTFontManager.fontManager.defaultFont

  private func display(_ latex: String, maxWidth: CGFloat) throws -> MTMathListDisplay {
    let list = try XCTUnwrap(MTMathListBuilder.build(fromString: latex))
    let font = try XCTUnwrap(font)
    return try XCTUnwrap(
      MTTypesetter.createLineForMathList(list, font: font, style: .display, maxWidth: maxWidth))
  }

  func testConstrainedAccentKeepsItsFraction() throws {
    let latex = "\\hat{\\frac{a}{b}}"
    let unconstrained = try display(latex, maxWidth: 0)
    let constrained = try display(latex, maxWidth: 200)
    let accent = try XCTUnwrap(constrained.subDisplays.first as? MTAccentDisplay)
    let fraction = try XCTUnwrap(accent.accentee?.subDisplays.first as? MTFractionDisplay)

    XCTAssertEqual(
      (fraction.numerator?.subDisplays.first as? MTCTLineDisplay)?.attributedString?.string, "𝑎")
    XCTAssertEqual(
      (fraction.denominator?.subDisplays.first as? MTCTLineDisplay)?.attributedString?.string, "𝑏")
    XCTAssertNotNil(accent.accent)
    XCTAssertEqual(constrained.width, unconstrained.width, accuracy: 0.001)
    XCTAssertEqual(constrained.ascent, unconstrained.ascent, accuracy: 0.001)
    XCTAssertEqual(constrained.descent, unconstrained.descent, accuracy: 0.001)
  }

  func testConstrainedAccentKeepsBothScripts() throws {
    try assertScripts("\\hat{x}^2_3")
  }

  func testConstrainedTextAccentKeepsBothScripts() throws {
    try assertScripts("\\text{\\hat{x}^2_3}")
  }

  func testConstrainedTextAccentKeepsItsNestedFontStyle() throws {
    for (command, expected) in [("mathbf", "𝐱"), ("mathit", "𝑥"), ("mathbb", "𝕩")] {
      let latex = "\\text{\\hat{\\\(command){x}}}"
      let unconstrained = try display(latex, maxWidth: 0)
      let constrained = try display(latex, maxWidth: 200)
      let accent = constrained.subDisplays.first as? MTAccentDisplay
      XCTAssertNotNil(accent, "Nested \(command) must keep its mathematical accent")
      guard let accent else { continue }
      XCTAssertEqual(
        (accent.accentee?.subDisplays.first as? MTCTLineDisplay)?.attributedString?.string,
        expected, command)
      XCTAssertEqual(constrained.width, unconstrained.width, accuracy: 0.001, command)
      XCTAssertEqual(constrained.ascent, unconstrained.ascent, accuracy: 0.001, command)
      XCTAssertEqual(constrained.descent, unconstrained.descent, accuracy: 0.001, command)
    }
  }

  private func assertScripts(_ latex: String) throws {
    let unconstrained = try display(latex, maxWidth: 0)
    let constrained = try display(latex, maxWidth: 200)
    let accent = try XCTUnwrap(constrained.subDisplays.first as? MTAccentDisplay)
    let scripts = try XCTUnwrap(accent.accentee).subDisplays.compactMap { $0 as? MTMathListDisplay }
    let superscript = try XCTUnwrap(scripts.first { $0.type == .superscript })
    let subscriptDisplay = try XCTUnwrap(scripts.first { $0.type == .ssubscript })

    XCTAssertEqual(
      (superscript.subDisplays.first as? MTCTLineDisplay)?.attributedString?.string, "2")
    XCTAssertEqual(
      (subscriptDisplay.subDisplays.first as? MTCTLineDisplay)?.attributedString?.string, "3")
    XCTAssertGreaterThan(superscript.position.y, 0)
    XCTAssertLessThan(subscriptDisplay.position.y, 0)
    XCTAssertEqual(constrained.width, unconstrained.width, accuracy: 0.001)
    XCTAssertEqual(constrained.ascent, unconstrained.ascent, accuracy: 0.001)
    XCTAssertEqual(constrained.descent, unconstrained.descent, accuracy: 0.001)
  }

  private func width(_ text: String) throws -> CGFloat {
    let font = try XCTUnwrap(font)
    let attributed = NSAttributedString(
      string: text,
      attributes: [kCTFontAttributeName as NSAttributedString.Key: font.ctFont as Any])
    return CGFloat(
      CTLineGetTypographicBounds(CTLineCreateWithAttributedString(attributed), nil, nil, nil))
  }

  private func assertBreak(_ text: String, after prefix: String, before next: String) throws {
    let font = try XCTUnwrap(font)
    let typesetter = MTTypesetter(withFont: font, style: .text, cramped: false, spaced: false)
    let maxWidth = try (width(prefix) + width(prefix + next)) / 2
    let index = try XCTUnwrap(
      typesetter.findBestBreakPoint(in: text, font: font.ctFont, maxWidth: maxWidth))

    XCTAssertEqual(String(text[..<index]), prefix)
    XCTAssertLessThanOrEqual(try width(String(text[..<index])), maxWidth)
  }

  func testLineBreakWithSupplementaryMathLetters() throws {
    try assertBreak("𝑥𝑦 𝑧𝑤 𝑎𝑏", after: "𝑥𝑦 ", before: "𝑧")
  }

  func testLineBreakWithCombiningAccents() throws {
    try assertBreak(
      "a\u{0301}b\u{0301} c\u{0301}d\u{0301} e\u{0301}f\u{0301}",
      after: "a\u{0301}b\u{0301} ", before: "c\u{0301}")
  }
}
