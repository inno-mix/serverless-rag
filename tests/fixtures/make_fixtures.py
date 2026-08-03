"""Regenerate the test fixtures (TESTING.md §5).

Run from the repository root::

    python tests/fixtures/make_fixtures.py

The PDF is emitted as raw PDF syntax rather than through a rendering library, so
generating fixtures needs no dependency beyond the standard library and the
document's facts stay fixed and known. The same facts are asserted by the
ingestion unit tests and, from T2.4, by the integration smoke test.
"""

from pathlib import Path

FIXTURE_DIR = Path(__file__).resolve().parent

#: Known facts, one page per inner list. Keep these stable — tests assert on them.
PDF_PAGES: list[list[str]] = [
    [
        "Acme Corporation Quarterly Report",
        "",
        "Acme Corporation reported third quarter revenue of 4.2 million dollars.",
        "Revenue grew 12 percent compared with the same quarter last year.",
        "The company operates three regional offices.",
    ],
    [
        "Operations Summary",
        "",
        "Acme Corporation employs 87 people across all regions.",
        "The head office is located in Wellington.",
        "Customer retention reached 94 percent during the quarter.",
    ],
]

SAMPLE_TXT = (
    "Acme Corporation reported third quarter revenue of 4.2 million dollars.\n"
    "Revenue grew 12 percent compared with the same quarter last year.\n"
    "\n"
    "The company operates three regional offices.\n"
)

SAMPLE_MD = (
    "# Acme Corporation Quarterly Report\n"
    "\n"
    "Acme Corporation reported third quarter revenue of 4.2 million dollars.\n"
    "\n"
    "## Operations\n"
    "\n"
    "The company operates three regional offices and employs 87 people.\n"
)


def _escape(text: str) -> str:
    """Escape the three characters that are special inside a PDF string."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> bytes:
    """Build a PDF content stream that draws `lines` in 12pt Helvetica."""
    parts = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
    for line in lines:
        parts.append(f"({_escape(line)}) Tj")
        parts.append("T*")
    parts.append("ET")
    return "\n".join(parts).encode("latin-1")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Assemble a minimal, valid multi-page PDF.

    Args:
        pages: One list of text lines per page.

    Returns:
        The complete PDF file as bytes, with a correct cross-reference table.
    """
    font_id = 3 + 2 * len(pages)
    page_ids = [3 + 2 * i for i in range(len(pages))]

    objects: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            "<< /Type /Pages /Kids [{}] /Count {} >>".format(
                " ".join(f"{pid} 0 R" for pid in page_ids), len(pages)
            ).encode("latin-1")
        ),
        font_id: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }

    for page_number, (page_id, lines) in enumerate(zip(page_ids, pages, strict=True)):
        contents_id = page_id + 1
        objects[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {contents_id} 0 R "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode("latin-1")
        stream = _content_stream(lines)
        objects[contents_id] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") + stream + b"\nendstream"
        )
        del page_number

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for object_id in sorted(objects):
        offsets[object_id] = len(out)
        out += f"{object_id} 0 obj\n".encode("latin-1")
        out += objects[object_id]
        out += b"\nendobj\n"

    xref_offset = len(out)
    highest = max(objects)
    out += f"xref\n0 {highest + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"
    for object_id in range(1, highest + 1):
        out += f"{offsets.get(object_id, 0):010d} 00000 n \n".encode("latin-1")
    out += f"trailer\n<< /Size {highest + 1} /Root 1 0 R >>\n".encode("latin-1")
    out += f"startxref\n{xref_offset}\n%%EOF\n".encode("latin-1")
    return bytes(out)


def main() -> None:
    """Write every fixture into this directory."""
    (FIXTURE_DIR / "sample.pdf").write_bytes(build_pdf(PDF_PAGES))
    (FIXTURE_DIR / "sample.txt").write_text(SAMPLE_TXT, encoding="utf-8")
    (FIXTURE_DIR / "sample.md").write_text(SAMPLE_MD, encoding="utf-8")
    # Only the extension is under test, so the bytes are deliberately not a
    # real Office document (TESTING.md §5).
    (FIXTURE_DIR / "unsupported.docx").write_bytes(b"")
    print(f"fixtures written to {FIXTURE_DIR}")


if __name__ == "__main__":
    main()
