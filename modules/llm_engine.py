import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

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
        "provider": "huggingface",
        "env_key": "LAYOUTLMV3_MODEL_PATH",
        "model": "microsoft/layoutlmv3-base",
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


def run_mllm_extraction(raw_text, model_name):
    """
    Prototype MLLM adapter.

    API calls are intentionally gated behind installed SDKs and configured keys so
    benchmarking can be enabled without breaking the local fallback workflow.
    """
    resolved_model_name, model_config = resolve_mllm_model(model_name)

    if not os.getenv(model_config["env_key"]):
        fields = extract_logistics_fields(raw_text)
        return {
            "raw_text": raw_text,
            "extracted_fields": fields,
            "status": f"{resolved_model_name} not configured; used rule-based schema fallback.",
            "prompt_schema": EXTRACTION_PROMPT_SCHEMA.strip(),
        }

    if model_config["provider"] == "openai":
        try:
            fields = run_openai_extraction(raw_text, model_config)
            return {
                "raw_text": raw_text,
                "extracted_fields": fields,
                "status": f"{resolved_model_name} API extraction completed.",
                "prompt_schema": EXTRACTION_PROMPT_SCHEMA.strip(),
            }
        except (KeyError, json.JSONDecodeError, urllib.error.URLError, TimeoutError) as error:
            fields = extract_logistics_fields(raw_text)
            return {
                "raw_text": raw_text,
                "extracted_fields": fields,
                "status": f"{resolved_model_name} API failed ({error.__class__.__name__}); used rule-based schema fallback.",
                "prompt_schema": EXTRACTION_PROMPT_SCHEMA.strip(),
            }

    fields = extract_logistics_fields(raw_text)
    return {
        "raw_text": raw_text,
        "extracted_fields": fields,
        "status": f"{resolved_model_name} adapter ready; API call implementation can be enabled for paid runs.",
        "prompt_schema": EXTRACTION_PROMPT_SCHEMA.strip(),
    }


def parse_json_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        return extract_logistics_fields(response_text)
