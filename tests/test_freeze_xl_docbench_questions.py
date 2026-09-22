import hashlib
import json

import pytest

from scripts import freeze_xl_docbench_questions as freezer


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_frozen_question_lists_match_the_reviewed_runs() -> None:
    expected = {
        "cross_doc": (101, "8d7e70c6f055360276b44d5a7b422d7dacac5f00143e75ef357c3f02d6ece348"),
        "single_doc": (473, "616587fb9376a96733a025ae9cfdc94d10203dfd93027d3c244af4bc15f18c4d"),
    }

    for config, (count, digest) in expected.items():
        question_ids = freezer.FROZEN_QUESTION_IDS[config]
        assert len(question_ids) == count
        question_id_bytes = ("\n".join(question_ids) + "\n").encode()
        assert hashlib.sha256(question_id_bytes).hexdigest() == digest


def test_freeze_questions_preserves_source_order_and_omits_unselected_rows(tmp_path) -> None:
    source = tmp_path / "qa_cross_doc.jsonl"
    destination = tmp_path / "qa_cross_doc_frozen.jsonl"
    _write_jsonl(
        source,
        [
            {"question_id": "q-second", "question": "second"},
            {"question_id": "q-extra", "question": "extra"},
            {"question_id": "q-first", "question": "first"},
        ],
    )

    assert freezer.freeze_questions(source, destination, ("q-first", "q-second")) == 2

    rows = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert [row["question_id"] for row in rows] == ["q-second", "q-first"]


def test_freeze_questions_requires_every_frozen_id_once(tmp_path) -> None:
    source = tmp_path / "qa_cross_doc.jsonl"
    _write_jsonl(source, [{"question_id": "q-present"}])

    with pytest.raises(ValueError, match="q-missing"):
        freezer.freeze_questions(
            source,
            tmp_path / "qa_cross_doc_frozen.jsonl",
            ("q-present", "q-missing"),
        )
