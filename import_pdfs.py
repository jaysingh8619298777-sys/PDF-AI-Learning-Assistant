import os
from pypdf import PdfReader

text = ""

pdf_folder = "pdfs"

for pdf in os.listdir(pdf_folder):

    if pdf.endswith(".pdf"):

        pdf_path = os.path.join(pdf_folder, pdf)

        reader = PdfReader(pdf_path)

        for page in reader.pages:

            if page.extract_text():
                text += page.extract_text()

print(text[:1000])
print("\nTotal Characters:", len(text))

text = text.encode(
    "utf-8",
    errors="ignore"
).decode("utf-8")

with open(
    "knowledge_base.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write(text)

print("Knowledge base saved successfully!")