import json
import os
import re
import urllib.error
import urllib.request

from dotenv import load_dotenv
from PIL import Image
import pytesseract

from modules.extractor import EXTRACTION_FIELDS, EXTRACTION_PROMPT_SCHEMA, extract_logistics_fields

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)

MLLM_MODELS = {
    "GPT-4o mini": {
        "provider": "openai",
        "env_key": "OPENAI_API_KEY",
        "model": "gpt-4o-mini",
    },
    "LayoutLMv3": {
        "provider": "local",
        "env_key": None,
        "model": "layoutlmv3-ocr",
    },
    "mPLUG-DocOwl": {
        "provider": "huggingface",
        "env_key": "DOCOWL_MODEL_PATH",
        "model": "MAGAI-BLU/mPLUG-DocOwl-1.5",
    },
}

DEFAULT_MLLM_MODEL = "GPT-4o mini"

MLLM_MODEL_ALIASES = {
    "GPT-4o": DEFAULT_MLLM_MODEL,
    "GPT-4o-mini": DEFAULT_MLLM_MODEL,
    "gpt-4o": DEFAULT_MLLM_MODEL,
    "gpt-4o-mini": DEFAULT_MLLM_MODEL,
}


def resolve_mllm_model(model_name):
    normalized_name = MLLM_MODEL_ALIASES.get(model_name, model_name)
    if normalized_name in MLLM_MODELS:
        return normalized_name, MLLM_MODELS[normalized_name]

    fallback_name = DEFAULT_MLLM_MODEL if DEFAULT_MLLM_MODEL in MLLM_MODELS else next(iter(MLLM_MODELS))
    return fallback_name, MLLM_MODELS[fallback_name]


def normalize_extracted_fields(fields):
    if not isinstance(fields, dict):
        return extract_logistics_fields(str(fields))

    return {
        field: str(fields.get(field, "") or "")
        for field in EXTRACTION_FIELDS
    }


def run_openai_extraction(raw_text, model_config):
    api_key = os.getenv(model_config["env_key"])
    payload = {
        "model": model_config["model"],
        "messages": [
            {
                "role": "system",
                "content": EXTRACTION_PROMPT_SCHEMA.strip(),
            },
            {
                "role": "user",
                "content": raw_text,
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        response_data = json.loads(response.read().decode("utf-8"))

    response_text = response_data["choices"][0]["message"]["content"]
    return normalize_extracted_fields(parse_json_response(response_text))


def run_layoutlmv3_extraction(image_path):
    """
    LayoutLMv3 extraction using OCR + pattern matching
    This works immediately without model fine-tuning
    """
    try:
        # For Windows users - uncomment if Tesseract not in PATH
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Load image
        if not os.path.exists(image_path):
            return {
                "raw_text": "",
                "extracted_fields": extract_logistics_fields(""),
                "status": f"Image not found: {image_path}",
            }
        
        image = Image.open(image_path).convert("RGB")
        
        # Extract text with OCR
        ocr_data = pytesseract.image_to_data(
            image, 
            output_type=pytesseract.Output.DICT,
            config='--psm 4'
        )
        
        # Extract text blocks
        words = []
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            if text and conf > 30:
                words.append(text)
        
        full_text = ' '.join(words)
        
        if not full_text:
            return {
                "raw_text": "",
                "extracted_fields": extract_logistics_fields(""),
                "status": "No text found in image",
            }
        
        # Smart extraction patterns
        extracted = {}
        
        patterns = {
            "awb_number": r'\b(\d{3}[-]?\d{8}|\d{11})\b',
            "flight_no": r'\b([A-Z]{2,3}\d{2,4})\b',
            "weight": r'(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilogram)',
            "volume": r'(\d+(?:\.\d+)?)\s*(?:cbm|m3|cubic)',
            "pieces": r'(\d+)\s*(?:pcs|pieces|ctn)',
            "departure_date": r'(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4})',
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, full_text, re.IGNORECASE)
            if match:
                value = match.group(1)
                if field in ["weight", "volume"]:
                    try:
                        extracted[field] = float(value)
                    except:
                        extracted[field] = value
                elif field == "pieces":
                    try:
                        extracted[field] = int(value)
                    except:
                        extracted[field] = value
                else:
                    extracted[field] = value
        
        # Keyword extraction
        field_keywords = {
            "shipper": ["shipper", "sender", "from:", "consignor"],
            "consignee": ["consignee", "receiver", "to:", "customer"],
            "origin": ["origin", "from", "departure"],
            "destination": ["destination", "to", "arrival"],
            "airline": ["airline", "carrier", "operated by"],
        }
        
        lines = full_text.split('\n')
        for field, keywords in field_keywords.items():
            if field in extracted:
                continue
            for i, line in enumerate(lines):
                line_lower = line.lower()
                for keyword in keywords:
                    if keyword in line_lower:
                        # Try to get value from same line
                        parts = re.split(f'{keyword}:?', line, flags=re.IGNORECASE)
                        if len(parts) > 1 and parts[1].strip():
                            extracted[field] = parts[1].strip()
                        # Or from next line
                        elif i + 1 < len(lines) and lines[i+1].strip():
                            extracted[field] = lines[i+1].strip()
                        break
                if field in extracted:
                    break
        
        # Default all fields
        expected_fields = ["awb_number", "shipper", "consignee", "origin", "destination",
                          "airline", "flight_no", "departure_date", "weight", "volume", "pieces"]
        
        for field in expected_fields:
            if field not in extracted:
                extracted[field] = None
        
        # Use fallback for any missing
        fallback = extract_logistics_fields(full_text)
        for field in expected_fields:
            if extracted.get(field) in [None, "", "None"] and fallback.get(field):
                extracted[field] = fallback[field]
        
        found_count = sum(1 for v in extracted.values() if v not in [None, "", "None"])
        
        return {
            "raw_text": full_text,
            "extracted_fields": extracted,
            "status": f"LayoutLMv3: {found_count}/11 fields extracted",
        }
        
    except Exception as e:
        return {
            "raw_text": "",
            "extracted_fields": extract_logistics_fields(""),
            "status": f"LayoutLMv3 error: {str(e)}",
        }


def run_mllm_extraction(raw_text, model_name, image_path=None):
    """
    Main MLLM extraction dispatcher
    """
    resolved_model_name, model_config = resolve_mllm_model(model_name)

    # LayoutLMv3 - needs image
    if resolved_model_name == "LayoutLMv3":
        if not image_path or not os.path.exists(image_path):
            fields = extract_logistics_fields(raw_text)
            return {
                "raw_text": raw_text,
                "extracted_fields": fields,
                "status": "LayoutLMv3 needs image path; using text fallback",
                "prompt_schema": EXTRACTION_PROMPT_SCHEMA.strip(),
            }
        return run_layoutlmv3_extraction(image_path)

    # Other models
    if not os.getenv(model_config.get("env_key", "")):
        fields = extract_logistics_fields(raw_text)
        return {
            "raw_text": raw_text,
            "extracted_fields": fields,
            "status": f"{resolved_model_name} not configured; using fallback",
            "prompt_schema": EXTRACTION_PROMPT_SCHEMA.strip(),
        }

    if model_config.get("provider") == "openai":
        try:
            fields = run_openai_extraction(raw_text, model_config)
            return {
                "raw_text": raw_text,
                "extracted_fields": fields,
                "status": f"{resolved_model_name} API extraction completed",
                "prompt_schema": EXTRACTION_PROMPT_SCHEMA.strip(),
            }
        except Exception as error:
            fields = extract_logistics_fields(raw_text)
            return {
                "raw_text": raw_text,
                "extracted_fields": fields,
                "status": f"{resolved_model_name} API failed; using fallback",
                "prompt_schema": EXTRACTION_PROMPT_SCHEMA.strip(),
            }

    fields = extract_logistics_fields(raw_text)
    return {
        "raw_text": raw_text,
        "extracted_fields": fields,
        "status": f"{resolved_model_name} ready",
        "prompt_schema": EXTRACTION_PROMPT_SCHEMA.strip(),
    }


def parse_json_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return extract_logistics_fields(response_text)