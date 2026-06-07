import re


EXTRACTION_FIELDS = [
    "document_type",
    "invoice_number",
    "date",
    "shipper",
    "consignee",
    "airway_bill_number",
    "delivery_order_number",
    "packing_list_number",
    "purchase_order_number",
    "origin",
    "destination",
    "total_amount",
]


EXTRACTION_PROMPT_SCHEMA = f"""
Extract logistics document information as JSON.
Return only valid JSON with these keys:
{", ".join(EXTRACTION_FIELDS)}

Use an empty string when a field is not present. Preserve original values exactly.
"""


def extract_logistics_fields(raw_text):
    """
    Rule-based fallback extraction using the same schema as the MLLM prompt.
    """

    fields = {
        "document_type": detect_document_type(raw_text),
        "invoice_number": find_invoice_number(raw_text),
        "date": find_date(raw_text),
        "shipper": find_keyword_line(raw_text, ["shipper", "sender", "from"]),
        "consignee": find_keyword_line(raw_text, ["consignee", "receiver", "to"]),
        "airway_bill_number": find_by_label(raw_text, ["airway bill", "awb", "awb no", "mawb", "hawb"]),
        "delivery_order_number": find_by_label(raw_text, ["delivery order", "do no", "d/o no"]),
        "packing_list_number": find_by_label(raw_text, ["packing list", "pl no"]),
        "purchase_order_number": find_by_label(raw_text, ["purchase order", "po no", "po number"]),
        "origin": find_by_label(raw_text, ["origin", "port of loading", "from"]),
        "destination": find_by_label(raw_text, ["destination", "port of discharge", "to"]),
        "total_amount": find_total_amount(raw_text)
    }

    return fields


def detect_document_type(text):
    text_lower = text.lower()

    if "invoice" in text_lower:
        return "Invoice"
    if "delivery order" in text_lower or "do no" in text_lower:
        return "Delivery Order"
    if "packing list" in text_lower:
        return "Packing List"
    if "airway bill" in text_lower or "awb" in text_lower:
        return "Airway Bill"
    if "purchase order" in text_lower or "po number" in text_lower:
        return "Purchase Order"

    return "Unknown"


def find_invoice_number(text):
    patterns = [
        r"invoice\s*(no|number|#)?\s*[:\-]?\s*([A-Za-z0-9\-\/]+)",
        r"inv\s*(no|number|#)?\s*[:\-]?\s*([A-Za-z0-9\-\/]+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(2)

    return ""


def find_date(text):
    patterns = [
        r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b",
        r"\b\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2}\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0)

    return ""


def find_total_amount(text):
    patterns = [
        r"total\s*(amount)?\s*[:\-]?\s*(RM|USD|\$)?\s*([0-9,]+\.\d{2})",
        r"(RM|USD|\$)\s*([0-9,]+\.\d{2})"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)

    return ""


def find_by_label(text, labels):
    for label in labels:
        pattern = rf"{re.escape(label)}\s*(no|number|#)?\s*[:\-]?\s*([A-Za-z0-9\-\/., ]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_value(match.group(2))

    return ""


def find_keyword_line(text, keywords):
    lines = text.splitlines()

    for line in lines:
        line_lower = line.lower()

        for keyword in keywords:
            if keyword in line_lower:
                return line.strip()

    return ""


def clean_value(value):
    return value.strip().splitlines()[0].strip(" :-")
