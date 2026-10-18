"""Exception hierarchy.

The split matters for outcome classification (METHODOLOGY 1.4): only
`InfraError` may be turned into `env_failure`. Anything else raised by the
system under test or by this framework is a bug and must propagate, never be
silently recorded as a wrong answer.
"""

from __future__ import annotations


class AgentMemEvalError(Exception):
    """Base class for errors raised deliberately by this package."""


class InfraError(AgentMemEvalError):
    """Infrastructure failure: service unreachable, timeout, rate limit, DB error.

    Counts as `env_failure`, never as a wrong answer.
    """


class LLMUnavailableError(InfraError):
    """The LLM endpoint could not produce a response."""


class MemoryServiceError(InfraError):
    """The memory service could not be reached or could not serve the request."""


class JudgeError(AgentMemEvalError):
    """The judge could not produce a verdict. Never recorded as WRONG (METHODOLOGY 1.5)."""


class DatasetError(AgentMemEvalError):
    """Input data is malformed or inconsistent. Refuse rather than guess (P2)."""


class StageError(AgentMemEvalError):
    """A pipeline stage file is missing, tampered with, or does not match the run."""


class IncomparableError(AgentMemEvalError):
    """Two results cannot be compared (different corpus, judge version, ...)."""
