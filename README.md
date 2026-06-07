# Multimodal LLM for Intelligent Document Processing and RAG

Final Year Project prototype for logistics document extraction, benchmarking, and retrieval-augmented question answering.

## Project Scope

This system processes logistics documents such as invoices, airway bills, delivery orders, and packing lists. It supports PDF, image, Word, Excel, and TXT uploads, extracts structured fields, records benchmarking metrics, and lets users ask natural-language questions over processed document content.

The prototype follows three FYP2 development modules:

1. MLLM extraction: compare GPT-4o, Gemini 1.5 Pro, Claude 3.5 Sonnet, and Qwen-VL using one shared extraction schema.
2. OCR and hybrid benchmarking: compare Tesseract, PaddleOCR, direct MLLM extraction, and OCR plus LLM structuring.
3. RAG pipeline: chunk extracted content, embed document chunks, retrieve with dense and sparse ranking, and answer questions with grounded source chunks.

## Current Implementation

- Streamlit interface with a shared Document Workspace in the sidebar.
- Uploaded documents are saved once in a Document Registry and reused across RAG, MLLM benchmarking, and full extraction benchmarking.
- Extraction outputs are saved in an Extraction Output Registry so completed model/pipeline runs can be reused instead of rerun.
- Metric results are saved only when ground truth is available; extraction and processing time can still run without ground truth.
- Multi-format ingestion through `pdfplumber`, `python-docx`, `openpyxl`, and TXT loading.
- Tesseract OCR for image documents.
- PaddleOCR, MLLM, and hybrid module wrappers with graceful fallbacks when optional API keys or packages are unavailable.
- Rule-based logistics field extraction using the shared schema.
- Optional ground-truth JSON upload for precision, recall, F1, and field accuracy rate (FAR).
- Metrics grouped by document type, document format, approach, and model for comparing MLLMs across invoices, airway bills, delivery orders, and packing lists.
- Hybrid RAG retrieval using sentence-transformer dense embeddings plus TF-IDF sparse retrieval.

## Application Pages

1. RAG Document Assistant: main user workflow for selecting an existing extraction output, indexing it, and asking questions.
2. MLLM Benchmark by Document Type: Objective 1 workflow for comparing MLLM-only methods across document types and file formats.
3. Full Extraction Benchmark: Objective 2 workflow for comparing OCR, OCR 2.0, MLLM, and Hybrid OCR+LLM categories.
4. Results & Metrics Dashboard: central view of uploaded documents, saved extraction outputs, metric results, best model, fastest model, and comparison tables.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Tesseract OCR must also be installed locally. On Windows, update `modules/ocr_engine.py` if the executable is not located at:

```text
C:\Program Files\Tesseract-OCR\tesseract.exe
```

## Optional API Keys

Create a `.env` file for API-backed model runs:

```text
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

Without keys, the app keeps running and records the selected method as a prototype fallback.

## Ground Truth Format

Upload a JSON file in the Document Processing tab to calculate extraction metrics:

```json
{
  "document_type": "Invoice",
  "invoice_number": "INV-001",
  "date": "2026-05-20",
  "shipper": "Example Sender",
  "consignee": "Example Receiver",
  "total_amount": "RM 100.00"
}
```

Keep real logistics documents and personally identifiable information out of public repositories.
