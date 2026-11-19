# Read-only text search in a PDF: prints the sentences around each keyword. Usage: pdfgrep.py file.pdf kw1 kw2 ...
import sys, re
path, kws = sys.argv[1], [k.lower() for k in sys.argv[2:]]
text = None
try:
    import fitz  # PyMuPDF
    doc = fitz.open(path); pages = [p.get_text() for p in doc]
except Exception as e1:
    try:
        from pypdf import PdfReader
        pages = [(p.extract_text() or "") for p in PdfReader(path).pages]
    except Exception as e2:
        print("no PDF reader:", e1, e2); sys.exit(0)
for i, t in enumerate(pages):
    flat = re.sub(r"\s+", " ", t)
    for kw in kws:
        for m in re.finditer(re.escape(kw), flat.lower()):
            a = max(0, m.start() - 260); b = min(len(flat), m.end() + 320)
            print(("[p%d] %s: ...%s..." % (i + 1, kw, flat[a:b])).encode("ascii", "replace").decode())
            print()
