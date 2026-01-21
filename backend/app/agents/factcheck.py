from __future__ import annotations

"""Fact-check agent implementing multi-critic verification.

Implements a 3-round producer/critic loop that combines:
  - claim decomposition
  - aspect verification (factuality, citation, consistency critics)
  - Chain-of-Verification (CoVe) question generation and answers
  - Best-of-N candidate selection
"""

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import SystemMessage
from langchain_ollama import ChatOllama

from backend.app.schemas import Source


async def verify_and_revise(
    user_message: str,
    draft: str,
    sources: List[Source],
    max_rounds: int,
    best_of_n: int,
    llm: ChatOllama,
    emit: Optional[Callable[[str, Dict[str, Any]], Awaitable[None]]] = None,
) -> Tuple[str, List[dict]]:
    """Verify and revise a draft answer using multi-critic feedback.

    Args:
        user_message: User's original question.
        draft: Initial draft answer from the producer.
        sources: Retrieved sources for grounding.
        max_rounds: Max verification rounds.
        best_of_n: Number of candidate revisions to generate per round.
        llm: LLM instance used for verification steps.
        emit: Optional callback for UI event emission.

    Returns:
        Tuple of (best_answer, notes).
    """
    notes: List[dict] = []
    current = draft

    evidence = _evidence_blob(sources)

    for round_idx in range(1, max_rounds + 1):
        if emit:
            await emit("step_started", {"round": round_idx, "phase": "decomposition"})

        claims = await _decompose_claims(llm, user_message=user_message, draft=current, evidence=evidence)
        if emit:
            await emit("message", {"round": round_idx, "claims_count": len(claims)})

        if emit:
            await emit("step_started", {"round": round_idx, "phase": "aspect_verification"})

        # Start CoVe question generation in parallel to save time.
        cove_q_task = asyncio.create_task(_cove_questions(llm, user_message, current))

        factual_task = _critic_factuality(llm, user_message, current, evidence, claims)
        cite_task = _critic_citations(llm, current, evidence)
        cons_task = _critic_consistency(llm, user_message, current)
        factual, cite, cons = await asyncio.gather(factual_task, cite_task, cons_task)

        critic_bundle = {
            "round": round_idx,
            "critics": {
                "factuality": factual,
                "citations": cite,
                "consistency": cons,
            },
        }
        notes.append(critic_bundle)
        if emit:
            await emit("critic_feedback", {"round": round_idx, "critic": "factuality", "summary": factual.get("summary", "")})
            await emit("critic_feedback", {"round": round_idx, "critic": "citations", "summary": cite.get("summary", "")})
            await emit("critic_feedback", {"round": round_idx, "critic": "consistency", "summary": cons.get("summary", "")})

        passed = bool(factual.get("pass")) and bool(cite.get("pass")) and bool(cons.get("pass"))
        if passed:
            if emit:
                await emit("message", {"round": round_idx, "status": "pass"})
            if not cove_q_task.done():
                cove_q_task.cancel()
            return current, notes

        # CoVe: build verification questions -> answer them from evidence -> revise
        if emit:
            await emit("step_started", {"round": round_idx, "phase": "cove"})
        vqs = await cove_q_task
        vqa = await _cove_answer_questions(llm, vqs, evidence)

        # Best-of-N revisions (Stop-and-Go generation instruction included)
        if emit:
            await emit("step_started", {"round": round_idx, "phase": "best_of_n"})

        feedback = _merge_feedback(factual, cite, cons, claims, vqs, vqa)
        candidates = await _best_of_n_revisions(
            llm,
            user_message=user_message,
            draft=current,
            evidence=evidence,
            feedback=feedback,
            n=best_of_n,
        )
        best, best_score, score_notes = await _select_best_candidate(llm, user_message, evidence, candidates)
        notes.append({"round": round_idx, "best_of_n": score_notes})

        if emit:
            await emit(
                "revision_selected",
                {"round": round_idx, "summary": f"Selected candidate score={best_score:.2f}", "score": best_score},
            )

        current = best

    # If we still fail after max_rounds, return best effort (already last selected)
    return current, notes


def _evidence_blob(sources: List[Source], max_items: int = 10) -> str:
    """Serialize evidence into a compact string for prompt context.

    Args:
        sources: Sources to serialize.
        max_items: Max number of sources to include.

    Returns:
        Evidence string for prompts.
    """
    lines = []
    for s in sources[:max_items]:
        snippet = (s.snippet or "").replace("\n", " ").strip()
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        lines.append(f"- {s.title} ({s.tool}) — {s.url}\n  - {snippet}")
    return "\n".join(lines) if lines else "(no external sources retrieved)"


def _safe_json(s: str) -> Dict[str, Any]:
    """Parse best-effort JSON from model output.

    Args:
        s: Raw model output.

    Returns:
        Parsed JSON dict (empty if parsing fails).
    """
    try:
        return json.loads(s)
    except Exception:
        # attempt to extract first JSON object
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return {}
        return {}


async def _decompose_claims(llm: ChatOllama, *, user_message: str, draft: str, evidence: str) -> List[Dict[str, Any]]:
    """Extract atomic claims from the draft for verification.

    Args:
        llm: LLM instance.
        user_message: User's input question.
        draft: Draft answer.
        evidence: Evidence string.

    Returns:
        List of claim dicts.
    """
    system = SystemMessage(
        content=(
            "Extract atomic factual claims from the draft answer. Output strict JSON.\n"
            "Do NOT add commentary or markdown.\n"
            "JSON schema: {\"claims\":[{\"claim\":\"...\",\"needs_citation\":true|false,\"related_urls\":[\"...\"]}]}"
        )
    )
    prompt = (
        f"User question:\n{user_message}\n\n"
        f"Draft answer:\n{draft}\n\n"
        f"Evidence list:\n{evidence}\n"
    )
    resp = await llm.ainvoke([system, {"role": "user", "content": prompt}])
    obj = _safe_json(getattr(resp, "content", str(resp)))
    claims = obj.get("claims") or []
    return claims if isinstance(claims, list) else []


async def _critic_factuality(
    llm: ChatOllama,
    user_message: str,
    draft: str,
    evidence: str,
    claims: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Check claims against evidence; flag unsupported statements.

    Args:
        llm: LLM instance.
        user_message: User's input question.
        draft: Draft answer.
        evidence: Evidence string.
        claims: Atomic claims extracted from the draft.

    Returns:
        Critic JSON result.
    """
    system = SystemMessage(
        content=(
            "You are a factuality critic. Verify the draft against the evidence list.\n"
            "If evidence is insufficient, mark as uncertain (fail unless the draft clearly hedges).\n"
            "Output strict JSON only: {\"pass\":bool,\"summary\":\"...\",\"unsupported_claims\":[...],\"suggestions\":[...]}"
        )
    )
    prompt = (
        f"User question:\n{user_message}\n\nDraft answer:\n{draft}\n\n"
        f"Evidence list:\n{evidence}\n\n"
        f"Atomic claims JSON:\n{json.dumps({'claims': claims}, ensure_ascii=False)}\n"
    )
    resp = await llm.ainvoke([system, {"role": "user", "content": prompt}])
    obj = _safe_json(getattr(resp, "content", str(resp)))
    return _normalize_critic(obj, fallback="factuality")


async def _critic_citations(llm: ChatOllama, draft: str, evidence: str) -> Dict[str, Any]:
    """Verify that citations are relevant and present when needed.

    Args:
        llm: LLM instance.
        draft: Draft answer.
        evidence: Evidence string.

    Returns:
        Critic JSON result.
    """
    system = SystemMessage(
        content=(
            "You are a citation critic. Check whether cited URLs (if any) are relevant to nearby claims.\n"
            "If the answer makes strong claims without citations when evidence exists, mark fail.\n"
            "Output strict JSON only: {\"pass\":bool,\"summary\":\"...\",\"issues\":[...],\"suggestions\":[...]}"
        )
    )
    resp = await llm.ainvoke(
        [
            system,
            {
                "role": "user",
                "content": f"Draft answer:\n{draft}\n\nEvidence list:\n{evidence}\n",
            },
        ]
    )
    obj = _safe_json(getattr(resp, "content", str(resp)))
    return _normalize_critic(obj, fallback="citations")


async def _critic_consistency(llm: ChatOllama, user_message: str, draft: str) -> Dict[str, Any]:
    """Detect internal contradictions or overconfident claims.

    Args:
        llm: LLM instance.
        user_message: User's input question.
        draft: Draft answer.

    Returns:
        Critic JSON result.
    """
    system = SystemMessage(
        content=(
            "You are a consistency critic. Look for contradictions, overconfidence, and logical gaps.\n"
            "Output strict JSON only: {\"pass\":bool,\"summary\":\"...\",\"issues\":[...],\"suggestions\":[...]}"
        )
    )
    resp = await llm.ainvoke([system, {"role": "user", "content": f"User question:\n{user_message}\n\nDraft answer:\n{draft}\n"}])
    obj = _safe_json(getattr(resp, "content", str(resp)))
    return _normalize_critic(obj, fallback="consistency")


def _normalize_critic(obj: Dict[str, Any], fallback: str) -> Dict[str, Any]:
    """Normalize critic output to a consistent JSON shape.

    Args:
        obj: Raw critic output.
        fallback: Label used when output is malformed.

    Returns:
        Normalized critic output.
    """
    if "pass" not in obj:
        obj["pass"] = False
    obj.setdefault("summary", f"{fallback}: no structured output")
    obj.setdefault("issues", [])
    obj.setdefault("suggestions", [])
    obj.setdefault("unsupported_claims", [])
    return obj


async def _cove_questions(llm: ChatOllama, user_message: str, draft: str) -> List[str]:
    """Generate verification questions for CoVe.

    Args:
        llm: LLM instance.
        user_message: User's input question.
        draft: Draft answer.

    Returns:
        List of verification questions.
    """
    system = SystemMessage(
        content=(
            "Generate verification questions that, if answered, would confirm the factual correctness of the draft.\n"
            "Output strict JSON only: {\"questions\":[\"...\", ...]}"
        )
    )
    resp = await llm.ainvoke([system, {"role": "user", "content": f"User question:\n{user_message}\n\nDraft:\n{draft}\n"}])
    obj = _safe_json(getattr(resp, "content", str(resp)))
    qs = obj.get("questions") or []
    return qs if isinstance(qs, list) else []


async def _cove_answer_questions(llm: ChatOllama, questions: List[str], evidence: str) -> Dict[str, str]:
    """Answer CoVe verification questions using evidence only.

    Args:
        llm: LLM instance.
        questions: Verification questions.
        evidence: Evidence string.

    Returns:
        Mapping of question to answer.
    """
    system = SystemMessage(
        content=(
            "Answer each verification question using ONLY the evidence list.\n"
            "If not answerable from evidence, say 'INSUFFICIENT_EVIDENCE'.\n"
            "Output strict JSON only: {\"answers\":{\"question\":\"answer\", ...}}"
        )
    )
    resp = await llm.ainvoke(
        [
            system,
            {"role": "user", "content": f"Evidence list:\n{evidence}\n\nQuestions:\n{json.dumps(questions, ensure_ascii=False)}\n"},
        ]
    )
    obj = _safe_json(getattr(resp, "content", str(resp)))
    answers = obj.get("answers") or {}
    return answers if isinstance(answers, dict) else {}


def _merge_feedback(
    factual: Dict[str, Any],
    cite: Dict[str, Any],
    cons: Dict[str, Any],
    claims: List[Dict[str, Any]],
    vqs: List[str],
    vqa: Dict[str, str],
) -> str:
    """Combine critic feedback and CoVe results into a single JSON string.

    Args:
        factual: Factuality critic output.
        cite: Citation critic output.
        cons: Consistency critic output.
        claims: Extracted claims list.
        vqs: CoVe questions.
        vqa: CoVe answers.

    Returns:
        JSON string used for revision prompting.
    """
    return json.dumps(
        {
            "factuality": factual,
            "citations": cite,
            "consistency": cons,
            "claims": claims,
            "verification_questions": vqs,
            "verification_answers": vqa,
        },
        ensure_ascii=False,
    )


async def _best_of_n_revisions(
    llm: ChatOllama,
    *,
    user_message: str,
    draft: str,
    evidence: str,
    feedback: str,
    n: int,
) -> List[str]:
    """Generate N candidate revisions with small temperature variations.

    Args:
        llm: LLM instance.
        user_message: User's input question.
        draft: Current draft.
        evidence: Evidence string.
        feedback: Combined critic feedback.
        n: Number of candidates to generate.

    Returns:
        A list of revised answers.
    """
    system = SystemMessage(
        content=(
            "Revise the draft using the feedback and evidence.\n"
            "Follow Stop-and-Go generation: write one section, then ensure claims in that section are supported by evidence.\n"
            "If evidence is insufficient, hedge or omit the claim.\n"
            "Include citations as URLs when referencing evidence.\n"
        )
    )
    prompt = (
        f"User question:\n{user_message}\n\n"
        f"Evidence list:\n{evidence}\n\n"
        f"Current draft:\n{draft}\n\n"
        f"Feedback JSON:\n{feedback}\n\n"
        "Return a revised answer.\n"
    )

    # generate N candidates with slight temperature variation
    temps = [0.1, 0.2, 0.3, 0.4]
    tasks = []
    for i in range(max(1, n)):
        t = temps[i % len(temps)]
        cand_llm = ChatOllama(model=llm.model, base_url=llm.base_url, temperature=t)
        tasks.append(cand_llm.ainvoke([system, {"role": "user", "content": prompt}]))
    resps = await asyncio.gather(*tasks)
    outs = [getattr(r, "content", str(r)) for r in resps]
    # stable dedupe
    seen = set()
    out: List[str] = []
    for o in outs:
        k = re.sub(r"\s+", " ", o).strip()
        if k and k not in seen:
            seen.add(k)
            out.append(o)
    return out


async def _select_best_candidate(
    llm: ChatOllama,
    user_message: str,
    evidence: str,
    candidates: List[str],
) -> Tuple[str, float, Dict[str, Any]]:
    """Score candidates and return the best one.

    Args:
        llm: LLM instance.
        user_message: User's input question.
        evidence: Evidence string.
        candidates: Candidate answers.

    Returns:
        Tuple of (best_answer, best_score, score_details).
    """
    if not candidates:
        return "", 0.0, {"scores": []}

    system = SystemMessage(
        content=(
            "Score each candidate answer for evidence-grounded factuality and citation correctness.\n"
            "Output strict JSON only: {\"scores\":[{\"idx\":0,\"score\":0.0,\"reason\":\"...\"}, ...],\"best_idx\":0}"
        )
    )
    prompt = (
        f"User question:\n{user_message}\n\nEvidence list:\n{evidence}\n\n"
        f"Candidates:\n{json.dumps(candidates, ensure_ascii=False)}\n"
    )
    resp = await llm.ainvoke([system, {"role": "user", "content": prompt}])
    obj = _safe_json(getattr(resp, "content", str(resp)))
    scores = obj.get("scores") or []
    best_idx = obj.get("best_idx", 0)
    if not isinstance(best_idx, int):
        best_idx = 0

    best_idx = max(0, min(best_idx, len(candidates) - 1))
    best_score = 0.0
    for s in scores:
        if isinstance(s, dict) and s.get("idx") == best_idx:
            try:
                best_score = float(s.get("score", 0.0))
            except Exception:
                best_score = 0.0
            break

    return candidates[best_idx], best_score, {"scores": scores, "best_idx": best_idx}

