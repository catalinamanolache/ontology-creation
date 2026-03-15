from pypdf import PdfReader
import os

pdf1 = 'Ontologies, semantic layers and knowledge graphs.pdf'
if os.path.exists(pdf1):
    txt1 = ""
    for p in PdfReader(pdf1).pages:
        if p.extract_text():
            txt1 += p.extract_text() + "\n"
    with open("onto.txt", "w", encoding="utf-8") as f:
        f.write(txt1)

pdf2 = 'sample.pdf'
if os.path.exists(pdf2):
    txt2 = ""
    for p in PdfReader(pdf2).pages:
        if p.extract_text():
            txt2 += p.extract_text() + "\n"
    with open("sample.txt", "w", encoding="utf-8") as f:
        f.write(txt2)
