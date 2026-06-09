import json
import importlib
import os
import time
from html import escape

import pandas as pd
import streamlit as st

from modules.evaluator import calculate_basic_metrics
from modules.extractor import detect_document_type, extract_logistics_fields
from modules.file_loader import load_document_text, save_uploaded_file
from modules.hybrid_engine import run_hybrid_extraction
from modules.llm_engine import MLLM_MODELS, resolve_mllm_model, run_mllm_extraction
from modules.ocr2_engine import run_paddleocr_extraction
from modules.ocr_engine import run_tesseract_ocr
from modules.rag_engine import answer_question, build_vector_index
import modules.storage as storage_module


storage_module = importlib.reload(storage_module)
find_extraction_outputs = storage_module.find_extraction_outputs
get_document = storage_module.get_document
get_output = storage_module.get_output
load_document_registry = storage_module.load_document_registry
load_metric_registry = storage_module.load_metric_registry
load_output_registry = storage_module.load_output_registry
load_rag_documents_from_outputs = storage_module.load_rag_documents_from_outputs
register_document = storage_module.register_document
reset_demo_data = storage_module.reset_demo_data
save_extraction_output = storage_module.save_extraction_output
save_metrics = storage_module.save_metrics
save_processed_result = storage_module.save_processed_result
update_extraction_output = storage_module.update_extraction_output


DOCUMENT_TYPES = [
    "Auto-detect",
    "Bill of Lading",
    "Invoice",
    "Packing List",
    "Airway Bill",
    "Delivery Order",
    "Purchase Order",
    "Unknown",
]

MLLM_APPROACHES = [
    "LayoutLMv3",
    "mPLUG-DocOwl",
    "GPT-4o mini",
]

FULL_BENCHMARK_APPROACHES = [
    "Tesseract OCR",
    "PaddleOCR V4",
    "GPT-4o mini",
    "Hybrid OCR+LLM",
]


def inject_prototype_theme():
    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #ffffff;
            --bg-secondary: #f5f4f0;
            --bg-tertiary: #ece9e4;
            --bg-info: #e6f1fb;
            --bg-success: #eaf3de;
            --bg-warn: #faeeda;
            --bg-danger: #f6e7e2;
            --text-primary: #242522;
            --text-secondary: #5f5e5a;
            --text-tertiary: #9c9a92;
            --text-info: #185fa5;
            --text-success: #3b6d11;
            --text-warn: #854f0b;
            --text-danger: #8f4a3b;
            --border: rgba(0,0,0,0.10);
            --border-md: rgba(0,0,0,0.18);
            --border-info: rgba(24,95,165,0.25);
            --border-success: rgba(99,153,34,0.25);
            --radius-sm: 6px;
            --radius-md: 8px;
            --radius-lg: 10px;
            --font: "Segoe UI", system-ui, sans-serif;
        }

        html, body, [class*="css"] {
            font-family: var(--font);
        }

        body, .stApp {
            background: var(--bg-secondary);
            color: var(--text-primary);
        }

        .block-container {
            padding-top: 46px;
            padding-bottom: 32px;
            max-width: 1380px;
        }

        [data-testid="stSidebar"] {
            background: var(--bg-primary);
            border-right: 0.5px solid var(--border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 12px;
        }

        h1 {
            font-size: 20px !important;
            line-height: 1.25 !important;
            font-weight: 600 !important;
            letter-spacing: 0 !important;
            margin-bottom: 2px !important;
        }

        h2, h3 {
            color: var(--text-primary);
            letter-spacing: 0 !important;
        }

        h2 {
            font-size: 17px !important;
            font-weight: 600 !important;
        }

        h3 {
            font-size: 15px !important;
            font-weight: 600 !important;
        }

        p, label, span, div {
            letter-spacing: 0 !important;
        }

        [data-testid="stCaptionContainer"] {
            color: var(--text-secondary);
            font-size: 12px;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background: var(--bg-primary);
            border: 0.5px solid var(--border);
            border-radius: var(--radius-md);
            padding: 6px;
            margin-bottom: 16px;
            overflow-x: auto;
        }

        .stTabs [data-baseweb="tab"] {
            height: auto;
            min-height: 34px;
            padding: 6px 14px;
            border-radius: var(--radius-sm);
            color: var(--text-secondary);
            font-size: 13px;
            border: 0.5px solid transparent;
        }

        .stTabs [aria-selected="true"] {
            background: var(--bg-info);
            color: var(--text-info);
            border-color: var(--border-info);
            font-weight: 600;
        }

        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }

        div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlockBorderWrapper"],
        .proto-card {
            background: var(--bg-primary);
            border: 0.5px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 16px 20px;
        }

        .proto-card {
            margin-bottom: 14px;
        }

        .proto-section-title {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 15px;
            font-weight: 600;
            margin: 0 0 14px;
            color: var(--text-primary);
        }

        .proto-card-title {
            display: flex;
            align-items: center;
            gap: 7px;
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: .05em !important;
            margin-bottom: 12px;
            text-transform: uppercase;
        }

        .proto-notice {
            border-radius: var(--radius-md);
            display: flex;
            align-items: flex-start;
            gap: 8px;
            line-height: 1.55;
            margin-bottom: 14px;
            padding: 10px 14px;
            font-size: 12px;
        }

        .proto-info { background: var(--bg-info); color: #0c447c; }
        .proto-warn { background: var(--bg-warn); color: var(--text-warn); }
        .proto-success { background: var(--bg-success); color: var(--text-success); }

        .proto-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            border-radius: 4px;
            padding: 2px 7px;
            font-size: 10px;
            font-weight: 700;
            white-space: nowrap;
        }

        .badge-research { display: none; }
        .badge-ocr { background: #edf3e7; color: #536b3d; }
        .badge-ocr2 { background: #e1f5ee; color: #0f6e56; }
        .badge-mllm { background: #eeedfe; color: #534ab7; }
        .badge-hybrid { background: #f5eadb; color: #7a5a2f; }
        .badge-muted { background: var(--bg-secondary); color: var(--text-tertiary); }

        .doc-card {
            border-left: 2px solid transparent;
            padding: 8px 12px;
            margin: 0 -8px 6px;
            border-radius: 0 var(--radius-md) var(--radius-md) 0;
            transition: background .12s;
        }

        .doc-card:hover {
            background: var(--bg-secondary);
        }

        .doc-row {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .doc-icon {
            align-items: center;
            border-radius: 6px;
            display: inline-flex;
            flex-shrink: 0;
            font-size: 10px;
            font-weight: 700;
            height: 28px;
            justify-content: center;
            width: 28px;
        }

        .fmt-image, .fmt-img { background: var(--bg-warn); color: var(--text-warn); }
        .fmt-pdf { background: #f6e7e2; color: var(--text-danger); }
        .fmt-xlsx, .fmt-xls { background: var(--bg-success); color: var(--text-success); }
        .fmt-docx { background: #eeedfe; color: #534ab7; }
        .fmt-txt { background: #f1efe8; color: var(--text-secondary); }

        .doc-name {
            color: var(--text-primary);
            font-size: 12px;
            font-weight: 600;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .doc-type {
            color: var(--text-tertiary);
            font-size: 10px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .doc-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
            margin-top: 5px;
            padding-left: 36px;
        }

        .pill {
            align-items: center;
            border-radius: 10px;
            display: inline-flex;
            font-size: 10px;
            gap: 3px;
            padding: 1px 6px;
        }

        .pill-ok { background: var(--bg-success); color: var(--text-success); }
        .pill-warn { background: var(--bg-warn); color: var(--text-warn); }
        .pill-none { background: var(--bg-secondary); color: var(--text-tertiary); }

        .stButton > button,
        .stDownloadButton > button {
            background: var(--bg-primary);
            border: 0.5px solid var(--border-md);
            border-radius: var(--radius-sm);
            color: var(--text-primary);
            font-size: 12px;
            font-weight: 600;
            min-height: 34px;
            padding: 7px 14px;
            transition: all .12s;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: var(--bg-secondary);
            border-color: var(--border-md);
            color: var(--text-primary);
        }

        .stButton > button[kind="primary"] {
            background: #185fa5;
            border-color: #185fa5;
            color: white;
        }

        div[data-testid="stButtonGroup"] button,
        div[data-testid="stButtonGroup"] [role="button"] {
            border-radius: var(--radius-sm) !important;
            border: 0.5px solid var(--border-md) !important;
            background: var(--bg-primary) !important;
            color: var(--text-secondary) !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            min-height: 32px !important;
        }

        div[data-testid="stButtonGroup"] button[aria-pressed="true"],
        div[data-testid="stButtonGroup"] button[aria-selected="true"],
        div[data-testid="stButtonGroup"] button[aria-checked="true"],
        div[data-testid="stButtonGroup"] button[data-selected="true"],
        div[data-testid="stButtonGroup"] button[kind="pillsActive"],
        div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-pillsActive"],
        div[data-testid="stButtonGroup"] [role="button"][aria-pressed="true"] {
            background: #185fa5 !important;
            border-color: #185fa5 !important;
            color: #ffffff !important;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.32);
        }

        div[data-testid="stButtonGroup"] button[aria-pressed="true"] *,
        div[data-testid="stButtonGroup"] button[aria-selected="true"] *,
        div[data-testid="stButtonGroup"] button[aria-checked="true"] *,
        div[data-testid="stButtonGroup"] button[data-selected="true"] *,
        div[data-testid="stButtonGroup"] button[kind="pillsActive"] *,
        div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-pillsActive"] *,
        div[data-testid="stButtonGroup"] [role="button"][aria-pressed="true"] * {
            color: #ffffff !important;
        }

        div[data-testid="stButtonGroup"] button:hover,
        div[data-testid="stButtonGroup"] [role="button"]:hover {
            background: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
        }

        div[data-testid="stButtonGroup"] button[aria-pressed="true"]:hover,
        div[data-testid="stButtonGroup"] button[aria-selected="true"]:hover,
        div[data-testid="stButtonGroup"] button[aria-checked="true"]:hover,
        div[data-testid="stButtonGroup"] button[data-selected="true"]:hover,
        div[data-testid="stButtonGroup"] button[kind="pillsActive"]:hover,
        div[data-testid="stButtonGroup"] button[data-testid="stBaseButton-pillsActive"]:hover {
            background: #185fa5 !important;
            border-color: #185fa5 !important;
            color: #ffffff !important;
        }

        [data-baseweb="tag"] {
            background: var(--bg-info) !important;
            color: var(--text-info) !important;
        }

        [data-baseweb="tag"] span {
            color: var(--text-info) !important;
        }

        [data-testid="stMetric"] {
            background: var(--bg-primary);
            border: 0.5px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 14px 16px;
        }

        [data-testid="stMetricLabel"] {
            color: var(--text-secondary);
            font-size: 11px;
            text-transform: uppercase;
        }

        [data-testid="stMetricValue"] {
            color: var(--text-primary);
            font-size: 24px;
            font-weight: 700;
        }

        div[data-testid="stDataFrame"],
        div[data-testid="stTable"] {
            border: 0.5px solid var(--border);
            border-radius: var(--radius-md);
            overflow: hidden;
        }

        .stAlert {
            border-radius: var(--radius-md);
            font-size: 12px;
        }

        input, textarea, [data-baseweb="select"] > div {
            border-radius: var(--radius-sm) !important;
            border-color: var(--border-md) !important;
            font-size: 13px !important;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            font-size: 12px !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            color: var(--text-secondary);
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
        }

        @media (max-width: 900px) {
            .block-container { padding-left: 12px; padding-right: 12px; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title, badge=None):
    badge_html = f' <span class="proto-badge badge-research">{badge}</span>' if badge else ""
    st.markdown(f'<div class="proto-section-title">{title}{badge_html}</div>', unsafe_allow_html=True)


def render_notice(message, kind="info"):
    st.markdown(f'<div class="proto-notice proto-{kind}">{message}</div>', unsafe_allow_html=True)


def approach_badge(approach):
    badge_map = {
        "OCR": ("badge-ocr", "OCR"),
        "OCR_2_0": ("badge-ocr2", "OCR 2.0"),
        "MLLM": ("badge-mllm", "MLLM"),
        "HYBRID": ("badge-hybrid", "Hybrid"),
    }
    badge_class, label = badge_map.get(approach, ("badge-muted", approach or "Unknown"))
    return f'<span class="proto-badge {badge_class}">{label}</span>'


def extraction_group_label(option):
    group_map = {
        "Tesseract OCR": "OCR",
        "PaddleOCR V4": "OCR 2.0",
        "GPT-4o mini": "MLLM",
        "Hybrid OCR+LLM": "Hybrid",
    }
    return f"{group_map.get(option, 'Method')} | {option}"


def readable_label(value):
    return str(value or "Unknown").replace("_", " ").title()


def readable_approach(value):
    approach_map = {
        "OCR": "OCR",
        "OCR_2_0": "OCR 2.0",
        "MLLM": "MLLM",
        "HYBRID": "Hybrid",
    }
    return approach_map.get(value, readable_label(value))


def display_dataframe(df, column_labels=None, **kwargs):
    labels = {
        "document_name": "Document",
        "Document_Name": "Document",
        "document_type": "Document Type",
        "document_format": "Format",
        "selected_model": "Selected Model",
        "approach_type": "Approach Type",
        "approach": "Approach",
        "model_name": "Model",
        "module_source": "Module Source",
        "output_id": "Output ID",
        "document_id": "Document ID",
        "processing_time": "Processing Time (s)",
        "processing_time_seconds": "Processing Time (s)",
        "character_count": "Character Count",
        "word_count": "Word Count",
        "precision": "Precision",
        "recall": "Recall",
        "f1_score": "F1 Score",
        "far": "FAR",
        "correct_fields": "Correct Fields",
        "extracted_field_count": "Extracted Fields",
        "ground_truth_field_count": "Ground Truth Fields",
        "created_at": "Created At",
        "runs": "Runs",
        "avg_precision": "Avg Precision",
        "avg_recall": "Avg Recall",
        "avg_f1": "Avg F1",
        "avg_far": "Avg FAR",
        "avg_processing_time": "Avg Processing Time (s)",
    }
    if column_labels:
        labels.update(column_labels)

    st.dataframe(df.rename(columns=labels), use_container_width=True)


def file_format_badge(file_format):
    normalized = (file_format or "file").lower().replace(".", "")
    label_map = {
        "image": "IMG",
        "jpg": "IMG",
        "jpeg": "IMG",
        "png": "IMG",
        "pdf": "PDF",
        "xlsx": "XLS",
        "xls": "XLS",
        "docx": "DOC",
        "txt": "TXT",
    }
    return normalized, label_map.get(normalized, "FILE")


def ensure_data_folders():
    os.makedirs("data/uploads", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/metrics", exist_ok=True)


def parse_ground_truth(uploaded_ground_truth):
    if uploaded_ground_truth is None:
        return None

    try:
        return json.loads(uploaded_ground_truth.getvalue().decode("utf-8"))
    except json.JSONDecodeError:
        st.sidebar.error("Ground truth file must be valid JSON.")
        return None


def document_label(document):
    return f"{document['file_name']} | {document.get('document_type', 'Unknown')} | {document['document_id']}"


def output_label(output):
    doc = get_document(output.get("document_id")) or {}
    model = output.get("model_name", "unknown")
    approach = output.get("approach", "unknown")
    source = output.get("module_source", "unknown")
    return f"{doc.get('file_name', output.get('document_id'))} | {source} | {approach} | {model} | {output['output_id']}"


def get_selected_documents(documents, prompt="Select document(s)", key="selected_documents"):
    if not documents:
        st.warning("No uploaded documents yet. Add documents from the sidebar workspace.")
        return []

    documents_by_id = {
        document["document_id"]: document
        for document in documents
    }
    selected_ids = st.multiselect(
        prompt,
        options=list(documents_by_id.keys()),
        format_func=lambda document_id: documents_by_id[document_id]["file_name"],
        key=key,
    )
    selected_id_set = set(selected_ids)
    return [document for document in documents if document["document_id"] in selected_id_set]


def prepare_input_text(document):
    file_path = document["file_path"]
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg"]:
        return run_tesseract_ocr(file_path)

    return load_document_text(file_path)


def run_pipeline(document, module_source, approach, model_name, force_rerun=False):
    if approach in ["MLLM", "HYBRID"]:
        model_name, _ = resolve_mllm_model(model_name)

    existing_outputs = find_extraction_outputs(
        document_id=document["document_id"],
        approach=approach,
        model_name=model_name,
        module_source=module_source,
    )

    if existing_outputs and not force_rerun:
        return existing_outputs[-1], False

    start_time = time.time()
    file_path = document["file_path"]
    ext = os.path.splitext(file_path)[1].lower()

    if approach == "OCR":
        extracted_text = run_tesseract_ocr(file_path) if ext in [".png", ".jpg", ".jpeg"] else load_document_text(file_path)
        extracted_fields = extract_logistics_fields(extracted_text)
        status = "Tesseract/text extraction completed."

    elif approach == "OCR_2_0":
        extracted_text = run_paddleocr_extraction(file_path)
        extracted_fields = extract_logistics_fields(extracted_text)
        status = "OCR 2.0 extraction completed."

    elif approach == "MLLM":
        extracted_text = prepare_input_text(document)
        # Get image path if document is an image (for LayoutLMv3)
        file_path = document["file_path"]
        ext = os.path.splitext(file_path)[1].lower()
        image_path = file_path if ext in [".png", ".jpg", ".jpeg"] else None
        # Pass image_path to LayoutLMv3 if needed
        llm_result = run_mllm_extraction(extracted_text, model_name, image_path=image_path)
        extracted_fields = llm_result["extracted_fields"]
        status = llm_result["status"]

    elif approach == "HYBRID":
        hybrid_result = run_hybrid_extraction(file_path, model_name)
        extracted_text = hybrid_result["raw_text"]
        extracted_fields = hybrid_result["extracted_fields"]
        status = hybrid_result["status"]

    else:
        extracted_text = prepare_input_text(document)
        extracted_fields = extract_logistics_fields(extracted_text)
        status = "Fallback extraction completed."

    processing_time = round(time.time() - start_time, 3)
    output = save_extraction_output({
        "document_id": document["document_id"],
        "module_source": module_source,
        "approach": approach,
        "model_name": model_name,
        "extracted_text": extracted_text,
        "extracted_fields_json": extracted_fields,
        "processing_time": processing_time,
        "status": status,
    })

    ground_truth = document.get("ground_truth_json") or {}
    metrics = calculate_basic_metrics(
        document_name=document["file_name"],
        selected_model=f"{approach} - {model_name}",
        raw_text=extracted_text,
        processing_time=processing_time,
        extracted_fields=extracted_fields,
        ground_truth=ground_truth,
        document_format=document.get("file_format", ""),
    )
    metrics["output_id"] = output["output_id"]
    metrics["document_id"] = document["document_id"]
    metrics["module_source"] = module_source
    metrics["approach"] = approach
    metrics["model_name"] = model_name
    metric_result = save_metrics(metrics)
    if metric_result:
        output = update_extraction_output(
            output["output_id"],
            {"metric_result_id": metric_result["metric_result_id"]},
        ) or output

    save_processed_result({
        "document_name": document["file_name"],
        "document_id": document["document_id"],
        "selected_model": f"{approach} - {model_name}",
        "raw_text": extracted_text,
        "extracted_fields": extracted_fields,
        "processing_time": processing_time,
        "pipeline_status": status,
    })

    return output, True


def show_workspace_sidebar():
    st.sidebar.markdown(
        """
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
            <div style="width:28px;height:28px;border-radius:6px;background:#1a1a18;color:#fff;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;">DW</div>
            <div style="font-size:12px;font-weight:700;color:#1a1a18;">Document Workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload document",
        type=["pdf", "png", "jpg", "jpeg", "txt", "docx", "xlsx", "xls"],
    )
    selected_type = st.sidebar.selectbox("Document type", DOCUMENT_TYPES)
    ground_truth_file = st.sidebar.file_uploader("Optional ground truth JSON", type=["json"])

    if uploaded_file is not None and st.sidebar.button("Add to Workspace"):
        file_path = save_uploaded_file(uploaded_file)
        raw_text = load_document_text(file_path)
        detected_type = detect_document_type(raw_text) if raw_text else "Unknown"
        document_type = detected_type if selected_type == "Auto-detect" else selected_type
        ground_truth = parse_ground_truth(ground_truth_file)
        document = register_document(
            file_path=file_path,
            file_name=uploaded_file.name,
            document_type=document_type,
            ground_truth=ground_truth,
        )
        st.sidebar.success(f"Saved {document['document_id']}")

    with st.sidebar.expander("Reset data"):
        st.caption("Clear uploads, extraction outputs, metrics, processed files, and vector store data.")
        confirm_reset = st.checkbox(
            "I understand this will remove all demo data",
            key="confirm_demo_reset",
        )
        if st.button("Reset Everything", key="reset_demo_data"):
            if not confirm_reset:
                st.warning("Tick the confirmation checkbox first.")
            else:
                deleted_counts = reset_demo_data()
                st.session_state.clear()
                st.session_state["demo_reset_done"] = deleted_counts
                st.rerun()

    if st.session_state.get("demo_reset_done"):
        st.sidebar.success("Demo data reset. Workspace is clean.")
        st.session_state.pop("demo_reset_done", None)

    documents = load_document_registry()
    outputs = load_output_registry()

    st.sidebar.markdown("### Uploaded Documents")
    if documents:
        for document in documents:
            document_outputs = [output for output in outputs if output.get("document_id") == document["document_id"]]
            mllm_done = any(output.get("module_source") == "mllm_benchmark" for output in document_outputs)
            benchmark_done = any(output.get("module_source") == "extraction_benchmark" for output in document_outputs)
            indexed_output_ids = set(st.session_state.get("rag_indexed_output_ids", []))
            rag_indexed = any(output.get("output_id") in indexed_output_ids for output in document_outputs)
            rag_source_available = any(output.get("module_source") in ["direct_rag", "mllm_benchmark", "extraction_benchmark"] for output in document_outputs)
            fmt_class, fmt_label = file_format_badge(document.get("file_format", "file"))
            gt_class = "pill-ok" if document.get("ground_truth_available", False) else "pill-none"
            gt_label = "GT" if document.get("ground_truth_available", False) else "No GT"
            mllm_class = "pill-ok" if mllm_done else ("pill-warn" if benchmark_done else "pill-none")
            mllm_label = "MLLM done" if mllm_done else ("Bench done" if benchmark_done else "MLLM not run")
            rag_class = "pill-ok" if rag_indexed else ("pill-warn" if rag_source_available else "pill-none")
            rag_label = "RAG indexed" if rag_indexed else ("RAG source" if rag_source_available else "No RAG")
            file_name = escape(document["file_name"])
            document_type = escape(document.get("document_type", "Unknown"))

            st.sidebar.markdown(
                f"""
                <div class="doc-card">
                    <div class="doc-row">
                        <div class="doc-icon fmt-{fmt_class}">{fmt_label}</div>
                        <div style="min-width:0;">
                            <div class="doc-name">{file_name}</div>
                            <div class="doc-type">{document_type} | {len(document_outputs)} outputs</div>
                        </div>
                    </div>
                    <div class="doc-pills">
                        <span class="pill {gt_class}">{gt_label}</span>
                        <span class="pill {mllm_class}">{mllm_label}</span>
                        <span class="pill {rag_class}">{rag_label}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.sidebar.info("Upload a document once, then reuse it across all modules.")

    return documents, outputs


def page_rag_assistant(documents, outputs):
    render_section_title("RAG Document Assistant")
    render_notice("Choose saved extraction outputs, build a retrieval index, then ask grounded questions over logistics document content.")

    if not documents:
        st.warning("Upload a document from the sidebar workspace first.")
        return

    selected_docs = get_selected_documents(documents, "Select active document(s)", key="rag_selected_documents")
    active_ids = {document["document_id"] for document in selected_docs}
    available_outputs = [output for output in outputs if output.get("document_id") in active_ids]

    st.markdown("### Extraction Source")
    if selected_docs:
        method = st.selectbox(
            "Run extraction method if no suitable output exists",
            ["Use existing output", "Tesseract OCR", "PaddleOCR V4", "GPT-4o mini", "Hybrid OCR+LLM"],
            key="rag_extraction_method",
        )
        force_rerun = st.checkbox(
            "Rerun even if an existing matching output is available",
            key="rag_force_rerun",
        )

        if method != "Use existing output" and st.button("Run Extraction for RAG", type="primary", key="rag_run_extraction"):
            method_map = {
                "Tesseract OCR": ("OCR", "Tesseract"),
                "PaddleOCR V4": ("OCR_2_0", "PaddleOCR V4"),
                "GPT-4o mini": ("MLLM", "GPT-4o mini"),
                "Hybrid OCR+LLM": ("HYBRID", "GPT-4o mini"),
            }
            approach, model_name = method_map[method]
            for document in selected_docs:
                output, created = run_pipeline(document, "direct_rag", approach, model_name, force_rerun)
                st.write(f"{'Created' if created else 'Reused'} output: {output['output_id']}")
            st.rerun()

    if not available_outputs:
        st.info("No extraction outputs available yet for the selected document(s). Run one extraction first.")
        return

    selected_output_labels = st.multiselect(
        "Select extraction output(s) as RAG source",
        options=[output_label(output) for output in available_outputs],
        key="rag_selected_outputs",
    )
    selected_output_ids = [label.rsplit(" | ", 1)[-1] for label in selected_output_labels]

    question = st.text_input("Ask a question", placeholder="Example: What is the invoice number?", key="rag_question")
    top_k = st.slider("Retrieved chunks", min_value=1, max_value=10, value=3, key="rag_top_k")

    if st.button("Build/Update RAG Index and Answer", type="primary", key="rag_answer"):
        if not selected_output_ids:
            st.warning("Select at least one extraction output.")
            return
        if not question.strip():
            st.warning("Please enter a question.")
            return

        rag_documents = load_rag_documents_from_outputs(selected_output_ids)
        index_data = build_vector_index(rag_documents)
        answer, sources = answer_question(question, index_data, top_k=top_k)
        st.session_state["rag_indexed_output_ids"] = selected_output_ids

        st.markdown("### Answer")
        st.write(answer)
        #st.markdown("### Retrieved Source Chunks")
        #for i, source in enumerate(sources, start=1):
            #source_name = source.get("document_name", source.get("Document_Name", "Unknown source"))
            #st.markdown(f"**Source {i}: {source_name} | score: {source.get('retrieval_score', 'n/a')}**")
            #st.write(source["text"])


def page_mllm_benchmark(documents):
    render_section_title("MLLM Benchmark by Document Type")
    render_notice("Compare multimodal language model extraction across document types, formats, and the shared logistics field schema.")

    selected_docs = get_selected_documents(documents, key="mllm_selected_documents")
    selected_models = st.pills(
        "Select MLLMs to compare",
        MLLM_APPROACHES,
        selection_mode="multi",
        default=[],
        format_func=lambda option: f"MLLM | {option}",
        key="mllm_selected_models",
    )
    force_rerun = st.checkbox("Rerun existing MLLM outputs", key="mllm_force_rerun")

    if selected_docs:
        st.markdown("### Input Preparation Preview")
        preview_rows = []
        for document in selected_docs:
            fmt = document["file_format"]
            prep = "Use image directly" if fmt == "image" else "Convert to clean structured text"
            if fmt == "PDF":
                prep = "Extract page text or convert pages to images depending on model"
            if fmt == "XLSX":
                prep = "Convert sheets to structured markdown/table text"
            preview_rows.append({
                "document_id": document["document_id"],
                "file_name": document["file_name"],
                "document_type": document.get("document_type", "Unknown"),
                "file_format": fmt,
                "input_preparation": prep,
            })
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)

    if st.button("Run MLLM Benchmark", type="primary", key="mllm_run_benchmark"):
        if not selected_docs:
            st.warning("Select at least one document.")
            return

        for document in selected_docs:
            for selected_model in selected_models:
                model_name = selected_model
                output, created = run_pipeline(document, "mllm_benchmark", "MLLM", model_name, force_rerun)
                st.write(f"{'Created' if created else 'Reused'} {model_name} output for {document['file_name']}: {output['output_id']}")

    show_recent_outputs(
        module_source="mllm_benchmark",
        document_ids=[document["document_id"] for document in selected_docs],
    )


def page_full_extraction_benchmark(documents):
    render_section_title("Full Extraction Benchmark")
    render_notice("Compare OCR, OCR 2.0, MLLM, and hybrid OCR plus LLM methods using saved workspace documents.")

    selected_docs = get_selected_documents(documents, key="extract_selected_documents")
    selected_approaches = st.pills(
        "Select approaches/models to run",
        FULL_BENCHMARK_APPROACHES,
        selection_mode="multi",
        default=[],
        format_func=extraction_group_label,
        key="extract_selected_approaches",
    )
    force_rerun = st.checkbox("Rerun existing benchmark outputs", key="extract_force_rerun")

    if st.button("Run Full Extraction Benchmark", type="primary", key="extract_run_benchmark"):
        if not selected_docs:
            st.warning("Select at least one document.")
            return

        if not selected_approaches:
            st.warning("Select at least one extraction approach.")
            return

        method_map = {
            "Tesseract OCR": ("OCR", "Tesseract"),
            "PaddleOCR V4": ("OCR_2_0", "PaddleOCR V4"),
            "GPT-4o mini": ("MLLM", "GPT-4o mini"),
            "Hybrid OCR+LLM": ("HYBRID", "GPT-4o mini"),
        }
        for document in selected_docs:
            for selected_approach in selected_approaches:
                approach, model_name = method_map[selected_approach]
                output, created = run_pipeline(document, "extraction_benchmark", approach, model_name, force_rerun)
                st.write(f"{'Created' if created else 'Reused'} {selected_approach} output for {document['file_name']}: {output['output_id']}")

    show_recent_outputs(
        module_source="extraction_benchmark",
        document_ids=[document["document_id"] for document in selected_docs],
    )


def show_recent_outputs(module_source=None, document_ids=None):
    outputs = load_output_registry()
    if document_ids is not None:
        active_ids = set(document_ids)
        if not active_ids:
            st.info("Select document(s) to view matching outputs.")
            return
        outputs = [output for output in outputs if output.get("document_id") in active_ids]

    if module_source:
        outputs = [output for output in outputs if output.get("module_source") == module_source]

    if not outputs:
        st.info("No outputs saved yet for the selected document(s).")
        return

    rows = []
    for output in outputs:
        document = get_document(output.get("document_id")) or {}
        rows.append({
            "Output ID": output["output_id"],
            "Document": document.get("file_name", output.get("document_id")),
            "Document Type": document.get("document_type", "Unknown"),
            "Format": readable_label(document.get("file_format", "unknown")),
            "Module Source": readable_label(output.get("module_source")),
            "Approach": readable_approach(output.get("approach")),
            "Model": output.get("model_name"),
            "Processing Time (s)": output.get("processing_time"),
            "Created At": output.get("created_at"),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def page_results_dashboard(documents, outputs):
    render_section_title("Results & Metrics Dashboard")
    render_notice("Review saved documents, extraction outputs, metric records, and reusable RAG sources from one registry view.", "success")

    metrics_df = pd.DataFrame(load_metric_registry())
    all_metrics_df = pd.read_csv("data/metrics/metrics.csv") if os.path.exists("data/metrics/metrics.csv") else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Uploaded Documents", len(documents))
    c2.metric("Extraction Outputs", len(outputs))
    c3.metric("Metric Results", len(metrics_df))
    c4.metric("Available RAG Sources", len(outputs))

    if not all_metrics_df.empty:
        scored = all_metrics_df.dropna(subset=["f1_score"], how="all") if "f1_score" in all_metrics_df.columns else pd.DataFrame()

        if not scored.empty:
            best = scored.sort_values("f1_score", ascending=False).iloc[0]
            fastest = all_metrics_df.sort_values("processing_time_seconds", ascending=True).iloc[0]
            st.markdown("### Summary")
            st.write(f"Best model by F1-score: {best.get('model_name', best.get('selected_model'))} ({best.get('f1_score')})")
            st.write(f"Fastest model: {fastest.get('model_name', fastest.get('selected_model'))} ({fastest.get('processing_time_seconds')}s)")

        st.markdown("### All Processing Records")
        display_dataframe(all_metrics_df, width="stretch", hide_index=True)

        run_count_column = "document_name" if "document_name" in all_metrics_df.columns else "Document_Name"
        if {"document_type", "approach", "model_name", "f1_score", "processing_time_seconds", run_count_column}.issubset(all_metrics_df.columns):
            comparison = (
                all_metrics_df.groupby(["document_type", "approach", "model_name"])
                .agg(
                    runs=(run_count_column, "count"),
                    avg_precision=("precision", "mean"),
                    avg_recall=("recall", "mean"),
                    avg_f1=("f1_score", "mean"),
                    avg_far=("far", "mean"),
                    avg_processing_time=("processing_time_seconds", "mean"),
                )
                .reset_index()
            )
            st.markdown("### Comparison by Document Type")
            display_dataframe(comparison, width="stretch", hide_index=True)

    st.markdown("### Saved Extraction Outputs")
    show_recent_outputs()


st.set_page_config(
    page_title="IDP + RAG Workspace",
    page_icon="DOC",
    layout="wide",
)

ensure_data_folders()
inject_prototype_theme()

st.title("Intelligent Document Processing + RAG Workspace")
st.caption("Upload once, reuse documents across RAG, MLLM benchmarking, and full extraction benchmarking.")

documents, outputs = show_workspace_sidebar()

page = st.tabs([
    "RAG Document Assistant",
    "MLLM Benchmark by Document Type",
    "Full Extraction Benchmark",
    "Results & Metrics Dashboard",
])

with page[0]:
    page_rag_assistant(documents, outputs)

with page[1]:
    page_mllm_benchmark(documents)

with page[2]:
    page_full_extraction_benchmark(documents)

with page[3]:
    page_results_dashboard(documents, outputs)
