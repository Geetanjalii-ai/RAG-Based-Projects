"""Automated evaluation test suite using pytest and the benchmark dataset."""

import pytest
from hr_assistant.guardrails import check_input, REFUSAL_MESSAGE
from hr_assistant.pipeline import ask, build_hr_assistant
from evals.dataset_loader import (
    load_eval_dataset,
    get_guardrail_test_cases,
    get_rag_test_cases,
)


@pytest.fixture(scope="session")
def assistant_agent():
    """Build the HR assistant once for the entire test session."""
    return build_hr_assistant()


# ==========================================
# 1. GUARDRAIL EVALUATIONS
# ==========================================
@pytest.mark.parametrize("test_case", get_guardrail_test_cases(), ids=lambda tc: tc["id"])
def test_guardrails_classification(test_case):
    """Verify that guardrails correctly catch malicious inputs and allow benign inputs."""
    question = test_case["question"]
    expected_safe = test_case["is_safe"]

    is_safe, decision = check_input(question)

    assert is_safe == expected_safe, (
        f"Guardrail misclassified '{test_case['id']}': {question}\n"
        f"Expected is_safe={expected_safe}, got is_safe={is_safe}. Decision: {decision}"
    )


# ==========================================
# 2. RAG PIPELINE FACTUAL QA EVALUATIONS
# ==========================================
@pytest.mark.parametrize(
    "test_case",
    [tc for tc in get_rag_test_cases() if tc["category"] == "factual_qa"],
    ids=lambda tc: tc["id"],
)
def test_rag_factual_qa_answers(assistant_agent, test_case):
    """Verify that the RAG agent retrieves the correct context and answers factually."""
    question = test_case["question"]
    expected_keywords = test_case.get("expected_keywords", [])

    answer = ask(assistant_agent, question)

    assert answer != REFUSAL_MESSAGE, f"Agent refused a legitimate question: {question}"

    # Verify at least one essential keyword/phrase is present in the answer
    found_any = any(kw.lower() in answer.lower() for kw in expected_keywords)
    assert found_any, (
        f"Answer failed factual check for '{test_case['id']}'.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n"
        f"Expected one of keywords: {expected_keywords}"
    )


# ==========================================
# 3. UNANSWERABLE / OUT-OF-DOMAIN EVALUATIONS
# ==========================================
@pytest.mark.parametrize(
    "test_case",
    [tc for tc in get_rag_test_cases() if tc["category"] == "unanswerable_out_of_domain"],
    ids=lambda tc: tc["id"],
)
def test_rag_unanswerable_questions(assistant_agent, test_case):
    """Verify that the agent admits lack of knowledge rather than hallucinating."""
    question = test_case["question"]
    answer = ask(assistant_agent, question)

    refusal_cues = ["don't know", "do not know", "not mentioned", "not found", "does not contain", "no information"]
    assert any(cue in answer.lower() for cue in refusal_cues), (
        f"Agent hallucinated on out-of-domain question: {question}\n"
        f"Answer: {answer}"
    )
