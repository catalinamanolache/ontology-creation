from pypdf import PdfReader
import os

pdf3 = 'conference_latex_template.pdf'
if os.path.exists(pdf3):
    txt3 = ""
    for p in PdfReader(pdf3).pages:
        if p.extract_text():
            txt3 += p.extract_text() + "\n"
    with open("conference.txt", "w", encoding="utf-8") as f:
        f.write(txt3)
