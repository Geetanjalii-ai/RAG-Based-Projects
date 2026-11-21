"""Dataset loader and LangSmith sync utility for evaluation benchmark."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DATASET_PATH = Path(__file__).parent / "eval_dataset.json"


def load_eval_dataset(path: Path = DEFAULT_DATASET_PATH) -> List[Dict[str, Any]]:
    """Load and validate the evaluation dataset JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info("Loaded %d evaluation test cases from %s", len(data), path.name)
    return data


def get_rag_test_cases(dataset: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Return test cases intended for RAG retrieval and generation eval."""
    ds = dataset or load_eval_dataset()
    return [tc for tc in ds if tc.get("category") in ("factual_qa", "unanswerable_out_of_domain", "safety_benign")]


def get_guardrail_test_cases(dataset: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Return test cases intended specifically for guardrail evaluation."""
    ds = dataset or load_eval_dataset()
    return [tc for tc in ds if tc.get("category") in ("prompt_injection", "other_employee_data", "safety_benign")]


def sync_to_langsmith(
    dataset_name: str = "hr-assistant-eval-dataset",
    description: str = "Evaluation test cases for HR Policy RAG Assistant (Q&A + Guardrails)",
    path: Path = DEFAULT_DATASET_PATH,
) -> None:
    """Upload or update the evaluation dataset in LangSmith."""
    # pyrefly: ignore[missing-import]
    from langsmith import Client

    client = Client()
    data = load_eval_dataset(path)

    # Check if dataset already exists
    if client.has_dataset(dataset_name=dataset_name):
        logger.info("LangSmith dataset '%s' already exists. Reusing it.", dataset_name)
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        logger.info("Creating LangSmith dataset '%s'...", dataset_name)
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=description,
        )

    # Populate examples
    inputs = [{"question": tc["question"]} for tc in data]
    outputs = [
        {
            "ground_truth": tc["ground_truth"],
            "expected_keywords": tc.get("expected_keywords", []),
            "category": tc.get("category"),
            "is_safe": tc.get("is_safe"),
        }
        for tc in data
    ]

    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        dataset_id=dataset.id,
    )
    logger.info("Successfully synced %d examples to LangSmith dataset '%s'", len(data), dataset_name)


if __name__ == "__main__":
    cases = load_eval_dataset()
    print(f"Total test cases: {len(cases)}")
    print(f"RAG QA cases: {len(get_rag_test_cases(cases))}")
    print(f"Guardrail cases: {len(get_guardrail_test_cases(cases))}")
