import os
import pdfplumber
import pandas as pd
from docx import Document


UPLOAD_DIR = "data/uploads"


def save_uploaded_file(uploaded_file):
    """
    Save uploaded Streamlit file into data/uploads.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path


def load_document_text(file_path):
    """
    Load text from PDF, TXT, DOCX, XLSX.
    Images will be handled by OCR engine separately.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return load_txt(file_path)

    if ext == ".pdf":
        return load_pdf(file_path)

    if ext == ".docx":
        return load_docx(file_path)

    if ext in [".xlsx", ".xls"]:
        return load_xlsx(file_path)

    if ext in [".png", ".jpg", ".jpeg"]:
        return ""

    return ""


def load_txt(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(file_path):
    text = ""

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text.strip()


def load_docx(file_path):
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def load_xlsx(file_path):
    excel_file = pd.ExcelFile(file_path)
    all_text = []

    for sheet_name in excel_file.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        all_text.append(f"Sheet: {sheet_name}")
        all_text.append(df.to_string(index=False))

    return "\n\n".join(all_text)
