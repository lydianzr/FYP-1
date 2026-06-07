from modules.file_loader import load_document_text
from modules.ocr_engine import run_tesseract_ocr


def run_paddleocr_extraction(file_path):
    """
    OCR 2.0 adapter using PaddleOCR when available.
    Falls back cleanly so the prototype can run on machines without PaddleOCR.
    """
    try:
        from paddleocr import PaddleOCR
    except ImportError:
        return (
            "[PaddleOCR is not installed]\n\n"
            "Install paddleocr to run OCR 2.0 benchmarking. "
            "Falling back to the standard document text loader where possible.\n\n"
            f"{load_document_text(file_path)}"
        ).strip()

    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    results = ocr.ocr(file_path, cls=True)
    lines = []

    for page in results or []:
        for item in page or []:
            if len(item) >= 2 and len(item[1]) >= 1:
                lines.append(item[1][0])

    if lines:
        return "\n".join(lines)

    return run_tesseract_ocr(file_path)
