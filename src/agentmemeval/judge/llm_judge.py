"""LLM judge for open-ended answers, prompt versions v1 and v2 (METHODOLOGY 1.5).

* v1 (baseline): lenient. CORRECT if the key information is there; dates may
  differ by one day.
* v2: v1 plus the refusal rules: for a "did not happen" gold the only key
  information is that it did not happen (background need not be repeated), so a
  "cannot find any record" answer is CORRECT; refusing and then answering anyway
  (including a hedged inference) is WRONG; listing related records clearly marked
  as not the answer is still CORRECT. And tighter leniency: an answer that gives
  both correct and incorrect information is WRONG.

Everything else is held fixed between versions (model, endpoint, temperature 0,
output format) so the prompt is the only variable. v2 never replaces v1: both are
reported side by side.

A failed call or an unparseable verdict raises `JudgeError`. It is never turned
into WRONG.
"""

from __future__ import annotations

import json

from agentmemeval.errors import InfraError, JudgeError
from agentmemeval.judge.base import Judge, JudgeInput, JudgeVerdict
from agentmemeval.llm.base import ChatMessage
from agentmemeval.llm.openai_compat import OpenAICompatLLM

_COMMON = (
    "You grade answers to questions about a team's group-chat history. Compare the ANSWER "
    "with the GOLD answer and reply in json: "
    '{"verdict": "CORRECT" or "WRONG", "reason": "<one short sentence>"}.\n'
    "Rules:\n"
    "1. CORRECT if the answer contains the key information of the gold answer, even if it is "
    "worded differently or adds extra detail.\n"
    "2. Dates may differ by at most one day.\n"
    "3. Otherwise WRONG.\n"
)
_V2_RULES = (
    "4. If the gold answer says that something did not happen, does not exist, or that a "
    "person did not take part, the ONLY key information is that it did not happen. Background "
    "in the gold answer does not need to be repeated. An answer that says it cannot find any "
    "relevant record, or cannot confirm it, is CORRECT.\n"
    "5. If the answer first refuses or says it cannot find the information, and then still "
    "gives a substantive answer (including a hedged guess or an inference), it is WRONG. "
    "Listing related records that are explicitly marked as for reference only, not as the "
    "answer, is still CORRECT.\n"
    "6. If the answer states both correct and incorrect information (for example several "
    "candidates, one of them wrong), it is WRONG.\n"
)
SYSTEM_PROMPTS = {"v1": _COMMON, "v2": _COMMON + _V2_RULES}


class LLMJudge(Judge):
    def __init__(self, llm: OpenAICompatLLM, version: str):
        if version not in SYSTEM_PROMPTS:
            raise ValueError(f"unknown judge prompt version {version!r}")
        self.llm = llm
        self.version = version

    @property
    def model(self) -> str:
        return self.llm.model

    @property
    def prompt_version(self) -> str:
        return f"llm-{self.version}"

    def judge(self, item: JudgeInput) -> JudgeVerdict:
        user = (
            f"QUESTION: {item.question}\nGOLD: {item.gold}\nANSWER: {item.answer}\n"
            "Return the json object."
        )
        try:
            reply = self.llm.complete(
                [ChatMessage("system", SYSTEM_PROMPTS[self.version]), ChatMessage("user", user)],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
        except InfraError as exc:
            raise JudgeError(f"{item.qid}: judge call failed: {exc}") from exc
        try:
            data = json.loads(reply.text)
            verdict = str(data["verdict"]).strip().upper()
            reason = str(data.get("reason", "")).strip()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise JudgeError(f"{item.qid}: unparseable judge reply {reply.text[:200]!r}") from exc
        if verdict not in ("CORRECT", "WRONG"):
            raise JudgeError(f"{item.qid}: judge returned verdict {verdict!r}")
        return JudgeVerdict(label=verdict, reason=reason, raw=reply.text)
