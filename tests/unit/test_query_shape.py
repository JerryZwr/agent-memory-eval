"""Query-shape sensitivity experiment (METHODOLOGY 1.9), on the offline bm25 run."""

from __future__ import annotations

import json

import pytest

from agentmemeval.analysis.query_shape import (
    QUERY_SHAPE_FILE,
    SHAPES,
    queries,
    short_query,
    summary_ratios,
    write_query_shape,
)
from agentmemeval.pipeline.config import default_eval_config
from agentmemeval.pipeline.runner import load_run, run_pipeline

pytestmark = pytest.mark.unit


def test_short_query_keeps_the_first_content_words():
    q = "Which database is the primary database for the Heron scheduling service?"
    assert short_query(q) == "database primary heron scheduling service"
    assert short_query(q, n=2) == "database primary"


def test_three_shapes_per_question(mini):
    mc = mini.question_by_id["MA_C_Mini_001"]
    shapes = queries(mc)
    assert set(shapes) == set(SHAPES)
    assert "Recharts" in shapes["stem_options"] and "Recharts" not in shapes["stem"]
    open_q = mini.question_by_id["F_SH_Mini_001"]
    assert queries(open_q)["stem_options"] == open_q.question


def test_experiment_on_a_finished_run(tmp_path):
    run_pipeline(tmp_path, default_eval_config())
    path = write_query_shape(load_run(tmp_path))
    assert path.name == QUERY_SHAPE_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["short_query"].startswith("automatic")  # never presented as manual
    assert [s["shape"] for s in data["summary"]] == list(SHAPES)
    assert len(data["rows"]) == 13  # located-evidence questions only
    stem = summary_ratios(data)[0]
    assert stem["hits"].den == 13
    # the stem shape is what the pipeline searched with: its hit rate must match the run's
    run = load_run(tmp_path)
    from agentmemeval.analysis.breakdown import HitCriterion, two_by_two

    t = two_by_two(run.components.dataset.question_by_id, run.search_records,
                   run.judge_records, HitCriterion.MESSAGE)  # fmt: skip
    assert stem["hits"].num == t.hit_correct + t.hit_wrong
