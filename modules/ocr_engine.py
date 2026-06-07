import pytesseract
from PIL import Image
import cv2
import os


# For Windows, uncomment and update this path if needed:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def preprocess_image_for_ocr(image_path):
    """
    Basic preprocessing for OCR:
    grayscale, denoising, thresholding.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Image could not be loaded. Please check the file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)

    threshold = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]

    processed_path = os.path.join("data/processed", "preprocessed_image.png")
    cv2.imwrite(processed_path, threshold)

    return processed_path


def run_tesseract_ocr(image_path):
    """
    Run Tesseract OCR on image file.
    """
    processed_path = preprocess_image_for_ocr(image_path)
    image = Image.open(processed_path)

    text = pytesseract.image_to_string(image)

    return text.strip()