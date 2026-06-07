import os

from modules.extractor import extract_logistics_fields
from modules.file_loader import load_document_text
from modules.llm_engine import run_mllm_extraction
from modules.ocr_engine import run_tesseract_ocr


def run_hybrid_extraction(file_path, model_name="GPT-4o mini"):
    """
    OCR/text extraction followed by schema-based LLM structuring.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg"]:
        raw_text = run_tesseract_ocr(file_path)
    else:
        raw_text = load_document_text(file_path)

    llm_result = run_mllm_extraction(raw_text, model_name)
    fields = llm_result.get("extracted_fields") or extract_logistics_fields(raw_text)

    return {
        "raw_text": raw_text,
        "extracted_fields": fields,
        "status": llm_result.get("status", "Hybrid OCR + LLM structuring completed."),
    }
