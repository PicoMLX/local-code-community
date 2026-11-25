import XCTest

@testable import SwiftMath

final class MTMathUILabelReuseTests: XCTestCase {
  #if os(macOS)
    func testUnconstrainedReuseDoesNotInheritPreviousFormulaWidth() throws {
      let label = MTMathUILabel()
      label.labelMode = .text
      label.latex = "x+y"
      label.layoutSubtreeIfNeeded()
      let short = try XCTUnwrap(label.displayList)
      label.frame = CGRect(
        origin: .zero, size: CGSize(width: short.width, height: short.ascent + short.descent))

      let latex = "\\text{one two three four}"
      label.latex = latex
      label.layoutSubtreeIfNeeded()
      let reused = try XCTUnwrap(label.displayList)

      let fresh = MTMathUILabel()
      fresh.labelMode = .text
      fresh.latex = latex
      fresh.layoutSubtreeIfNeeded()
      let expected = try XCTUnwrap(fresh.displayList)

      XCTAssertGreaterThan(
        expected.width, label.bounds.width, "The second formula must exceed the old frame")
      XCTAssertEqual(reused.width, expected.width, accuracy: 0.001)
      XCTAssertEqual(reused.ascent, expected.ascent, accuracy: 0.001)
      XCTAssertEqual(reused.descent, expected.descent, accuracy: 0.001)
      XCTAssertEqual(reused.width, label.intrinsicContentSize.width, accuracy: 0.001)
    }
  #endif

  func testLayoutKeepsExplicitPreferredWidth() throws {
    let label = MTMathUILabel(frame: CGRect(x: 0, y: 0, width: 500, height: 100))
    label.labelMode = .text
    label.latex = "\\text{one two three four}"
    label.preferredMaxLayoutWidth = 100
    label._layoutSubviews()
    let actual = try XCTUnwrap(label.displayList)
    let expected = try XCTUnwrap(
      MTTypesetter.createLineForMathList(
        label.mathList, font: label.font, style: .text, maxWidth: 100))

    XCTAssertEqual(actual.width, expected.width, accuracy: 0.001)
    XCTAssertEqual(actual.ascent, expected.ascent, accuracy: 0.001)
    XCTAssertEqual(actual.descent, expected.descent, accuracy: 0.001)
    XCTAssertGreaterThan(actual.ascent + actual.descent, try XCTUnwrap(label.font).fontSize)
  }
}
