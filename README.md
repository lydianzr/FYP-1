# Multimodal LLM for Intelligent Document Processing and RAG


An interactive, all-in-one workspace for intelligent document processing (IDP). Upload your documents once and leverage multiple extraction approaches, from traditional OCR to cutting-edge Multimodal LLMs (MLLMs)—and build a Retrieval-Augmented Generation (RAG) assistant over the extracted content.

## Core Features

*   **Unified Workspace:** Upload documents (PDF, images, Word, Excel, text) once and reuse them across all features.
*   **Multiple Extraction Methods:** Compare the performance of:
    *   `Tesseract OCR`
    *   `PaddleOCR V4`
    *   `LayoutLMv3`
    *   `Hybrid OCR+LLM`
*   **MLLM Benchmark:** Benchmark and compare the extraction results of different MLLMs (e.g., LayoutLMv3, GPT-4o mini) by document type.
*   **RAG Document Assistant:** Select extraction outputs, build a vector index, and ask grounded, context-aware questions about your logistics documents.
*   **Interactive Results Dashboard:** Visualize performance metrics (F1 Score, Precision, Recall, Processing Time) with dynamic charts and filterable data tables.
*   **Persistent Data Storage:** All uploads, extractions, and metrics are saved locally, allowing you to pick up where you left off.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lydianzr/FYP-1.git
    cd FYP-1

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py