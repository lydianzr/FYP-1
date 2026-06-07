def calculate_basic_metrics(
    document_name,
    selected_model,
    raw_text,
    processing_time,
    extracted_fields=None,
    ground_truth=None,
    document_format="",
):
    """
    Track processing statistics and, when ground truth is available, extraction quality.
    """

    words = raw_text.split()
    extracted_fields = extracted_fields or {}
    ground_truth = ground_truth or {}
    extraction_scores = calculate_extraction_metrics(extracted_fields or {}, ground_truth or {})
    document_type = ground_truth.get("document_type") or extracted_fields.get("document_type") or "Unknown"
    approach_type, model_name = split_processing_approach(selected_model)

    metrics = {
        "document_name": document_name,
        "document_type": document_type,
        "document_format": document_format,
        "selected_model": selected_model,
        "approach_type": approach_type,
        "model_name": model_name,
        "processing_time_seconds": processing_time,
        "character_count": len(raw_text),
        "word_count": len(words),
        "precision": extraction_scores["precision"],
        "recall": extraction_scores["recall"],
        "f1_score": extraction_scores["f1_score"],
        "far": extraction_scores["far"],
        "correct_fields": extraction_scores["correct_fields"],
        "extracted_field_count": extraction_scores["extracted_field_count"],
        "ground_truth_field_count": extraction_scores["ground_truth_field_count"],
    }

    return metrics


def split_processing_approach(selected_model):
    if " - " not in selected_model:
        return selected_model, selected_model

    approach_type, model_name = selected_model.split(" - ", 1)
    return approach_type, model_name


def calculate_extraction_metrics(extracted_fields, ground_truth):
    if not ground_truth:
        return {
            "precision": None,
            "recall": None,
            "f1_score": None,
            "far": None,
            "correct_fields": None,
            "extracted_field_count": None,
            "ground_truth_field_count": None,
        }

    extracted_non_empty = {
        key: normalize_value(value)
        for key, value in extracted_fields.items()
        if normalize_value(value)
    }
    truth_non_empty = {
        key: normalize_value(value)
        for key, value in ground_truth.items()
        if normalize_value(value)
    }

    correct = 0
    for key, truth_value in truth_non_empty.items():
        if extracted_non_empty.get(key) == truth_value:
            correct += 1

    extracted_count = len(extracted_non_empty)
    truth_count = len(truth_non_empty)
    precision = correct / extracted_count if extracted_count else 0
    recall = correct / truth_count if truth_count else 0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0
    )
    far = correct / truth_count if truth_count else 0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
        "far": round(far, 4),
        "correct_fields": correct,
        "extracted_field_count": extracted_count,
        "ground_truth_field_count": truth_count,
    }


def normalize_value(value):
    return " ".join(str(value).lower().strip().split())
