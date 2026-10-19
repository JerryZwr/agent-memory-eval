from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _inner_ini(request: pytest.FixtureRequest) -> None:
    """Inner pytester sessions get a minimal ini (they don't see pyproject.toml).

    Without it pytest-asyncio warns about an unset loop scope, and the outer
    session's `filterwarnings = error` turns that warning into a failure.
    """
    if "pytester" in request.fixturenames:
        pytester: pytest.Pytester = request.getfixturevalue("pytester")
        pytester.makeini("[pytest]\nasyncio_default_fixture_loop_scope = function\n")
