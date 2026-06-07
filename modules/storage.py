import os
import json
import shutil
import pandas as pd
from datetime import datetime
from uuid import uuid4


UPLOAD_DIR = "data/uploads"
PROCESSED_DIR = "data/processed"
METRICS_DIR = "data/metrics"
METRICS_FILE = os.path.join(METRICS_DIR, "metrics.csv")
DOCUMENT_REGISTRY_FILE = os.path.join(PROCESSED_DIR, "document_registry.json")
OUTPUT_REGISTRY_FILE = os.path.join(PROCESSED_DIR, "extraction_output_registry.json")
METRIC_REGISTRY_FILE = os.path.join(METRICS_DIR, "metric_registry.json")
VECTOR_STORE_DIR = "vector_store"


def utc_timestamp():
    return datetime.now().isoformat(timespec="seconds")


def read_json_list(file_path):
    if not os.path.exists(file_path):
        return []

    if os.path.getsize(file_path) == 0:
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            backup_path = f"{file_path}.invalid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.replace(file_path, backup_path)
            write_json_list(file_path, [])
            return []

    return data if isinstance(data, list) else []


def write_json_list(file_path, rows):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False)


def _clear_directory_contents(directory):
    deleted = 0
    os.makedirs(directory, exist_ok=True)

    project_root = os.path.abspath(os.getcwd())
    target_dir = os.path.abspath(directory)
    if os.path.commonpath([project_root, target_dir]) != project_root:
        raise ValueError(f"Refusing to clear outside project directory: {target_dir}")

    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        deleted += 1

    return deleted


def reset_demo_data():
    """
    Clear all user/demo data while keeping project source files intact.
    """
    deleted_counts = {
        "uploads": _clear_directory_contents(UPLOAD_DIR),
        "processed": _clear_directory_contents(PROCESSED_DIR),
        "metrics": _clear_directory_contents(METRICS_DIR),
        "vector_store": _clear_directory_contents(VECTOR_STORE_DIR),
    }

    write_json_list(DOCUMENT_REGISTRY_FILE, [])
    write_json_list(OUTPUT_REGISTRY_FILE, [])
    write_json_list(METRIC_REGISTRY_FILE, [])

    return deleted_counts


def get_file_format(file_name):
    ext = os.path.splitext(file_name)[1].lower().lstrip(".")

    if ext in ["png", "jpg", "jpeg", "tif", "tiff"]:
        return "image"
    if ext == "pdf":
        return "PDF"
    if ext in ["xlsx", "xls"]:
        return "XLSX"
    if ext == "docx":
        return "DOCX"
    if ext == "txt":
        return "TXT"

    return ext.upper() if ext else "unknown"


def register_document(file_path, file_name, document_type="Unknown", ground_truth=None):
    documents = load_document_registry()
    existing = next((doc for doc in documents if doc.get("file_name") == file_name), None)

    if existing:
        existing["file_path"] = file_path
        existing["document_type"] = document_type or existing.get("document_type", "Unknown")
        existing["ground_truth_available"] = bool(ground_truth) or existing.get("ground_truth_available", False)
        if ground_truth:
            existing["ground_truth_json"] = ground_truth
        existing["upload_status"] = "uploaded"
        write_json_list(DOCUMENT_REGISTRY_FILE, documents)
        return existing

    document = {
        "document_id": f"doc_{uuid4().hex[:10]}",
        "file_name": file_name,
        "file_path": file_path,
        "file_format": get_file_format(file_name),
        "document_type": document_type or "Unknown",
        "uploaded_at": utc_timestamp(),
        "upload_status": "uploaded",
        "ground_truth_available": bool(ground_truth),
        "ground_truth_json": ground_truth or {},
    }
    documents.append(document)
    write_json_list(DOCUMENT_REGISTRY_FILE, documents)
    return document


def load_document_registry():
    return read_json_list(DOCUMENT_REGISTRY_FILE)


def get_document(document_id):
    return next(
        (doc for doc in load_document_registry() if doc.get("document_id") == document_id),
        None,
    )


def load_output_registry():
    return read_json_list(OUTPUT_REGISTRY_FILE)


def save_extraction_output(output):
    outputs = load_output_registry()
    output = dict(output)
    output.setdefault("output_id", f"out_{uuid4().hex[:10]}")
    output.setdefault("created_at", utc_timestamp())
    outputs.append(output)
    write_json_list(OUTPUT_REGISTRY_FILE, outputs)
    return output


def update_extraction_output(output_id, updates):
    outputs = load_output_registry()

    for output in outputs:
        if output.get("output_id") == output_id:
            output.update(updates)
            write_json_list(OUTPUT_REGISTRY_FILE, outputs)
            return output

    return None


def find_extraction_outputs(document_id=None, approach=None, model_name=None, module_source=None):
    outputs = load_output_registry()

    if document_id:
        outputs = [output for output in outputs if output.get("document_id") == document_id]
    if approach:
        outputs = [output for output in outputs if output.get("approach") == approach]
    if model_name:
        outputs = [output for output in outputs if output.get("model_name") == model_name]
    if module_source:
        outputs = [output for output in outputs if output.get("module_source") == module_source]

    return outputs


def get_output(output_id):
    return next(
        (output for output in load_output_registry() if output.get("output_id") == output_id),
        None,
    )


def save_metric_result(metrics):
    if metrics.get("precision") is None:
        return None

    metric_results = read_json_list(METRIC_REGISTRY_FILE)
    result = dict(metrics)
    result.setdefault("metric_result_id", f"met_{uuid4().hex[:10]}")
    result.setdefault("created_at", utc_timestamp())
    metric_results.append(result)
    write_json_list(METRIC_REGISTRY_FILE, metric_results)
    return result


def load_metric_registry():
    return read_json_list(METRIC_REGISTRY_FILE)


def save_processed_result(result):
    """
    Save extracted result as JSON.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = result["document_name"].replace(" ", "_")
    file_name = f"{timestamp}_{safe_name}.json"

    output_path = os.path.join(PROCESSED_DIR, file_name)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    return output_path


def save_metrics(metrics):
    """
    Append metrics to CSV.
    """
    os.makedirs(METRICS_DIR, exist_ok=True)

    new_row = pd.DataFrame([metrics])

    if os.path.exists(METRICS_FILE):
        old_df = pd.read_csv(METRICS_FILE)
        final_df = pd.concat([old_df, new_row], ignore_index=True)
    else:
        final_df = new_row

    final_df.to_csv(METRICS_FILE, index=False)
    return save_metric_result(metrics)


def load_metrics():
    """
    Load metrics CSV.
    """
    if not os.path.exists(METRICS_FILE):
        return pd.DataFrame()

    return pd.read_csv(METRICS_FILE)


def load_all_processed_texts():
    """
    Load all processed JSON files for RAG.
    """
    if not os.path.exists(PROCESSED_DIR):
        return []

    documents = []

    for file_name in os.listdir(PROCESSED_DIR):
        if file_name in [
            os.path.basename(DOCUMENT_REGISTRY_FILE),
            os.path.basename(OUTPUT_REGISTRY_FILE),
        ]:
            continue

        if file_name.endswith(".json"):
            file_path = os.path.join(PROCESSED_DIR, file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    continue

            extracted_fields = data.get("extracted_fields", {})
            structured_text = "\n".join(
                f"{key}: {value}"
                for key, value in extracted_fields.items()
                if value
            )
            document_text = "\n\n".join(
                part
                for part in [data.get("raw_text", ""), structured_text]
                if part
            )

            documents.append({
                "document_name": data.get("document_name", file_name),
                "text": document_text
            })

    return documents


def load_rag_documents_from_outputs(output_ids):
    outputs = load_output_registry()
    selected_outputs = [
        output for output in outputs if output.get("output_id") in set(output_ids)
    ]

    documents = []
    for output in selected_outputs:
        fields = output.get("extracted_fields_json", {})
        structured_text = "\n".join(
            f"{key}: {value}"
            for key, value in fields.items()
            if value
        )
        text = "\n\n".join(
            part
            for part in [output.get("extracted_text", ""), structured_text]
            if part
        )
        documents.append({
            "document_name": output.get("output_id", "output"),
            "text": text,
        })

    return documents
