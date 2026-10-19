"""pytest_agenteval: pytest plugin for evaluating memory systems and agents.

Test code imports the helpers from here; pytest loads `pytest_agenteval.plugin`
through the `pytest11` entry point.
"""

from pytest_agenteval.plugin import NOT_MEASURED_PREFIX, not_measured

__all__ = ["NOT_MEASURED_PREFIX", "not_measured"]
