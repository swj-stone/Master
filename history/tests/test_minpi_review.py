import pytest
import yaml

from experiments.review_minpi_study import snapshot, verify


def test_independent_decision_is_bound_to_the_current_paper_and_feedback(tmp_path):
    paper = tmp_path / "paper/minpi"
    paper.mkdir(parents=True)
    evidence = tmp_path / "results/minpi-study/raw-cost/rows.csv"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("5,6\n", encoding="utf-8")
    data = tmp_path / "data/revision-cohorts/mhealth-windows.npz"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"measurements")
    (paper / "main.tex").write_text(r"\input{section}", encoding="utf-8")
    (paper / "main.pdf").write_bytes(b"candidate")
    (paper / "section.tex").write_text("Initial result.", encoding="utf-8")
    (paper / "README.md").write_text("Check the result.", encoding="utf-8")
    report = tmp_path / "review.md"
    current = snapshot(paper)

    def decide(decision):
        record = dict(decision=decision, **current)
        report.write_text("---\n" + yaml.safe_dump(record) + "---\nChecked.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pending"):
        verify(paper, report)
    decide("revise")
    with pytest.raises(ValueError, match="returned"):
        verify(paper, report)
    decide("ready_for_external_review")
    assert verify(paper, report) == current
    for path in (paper / "section.tex", paper / "README.md", paper / "main.pdf", evidence, data):
        before = path.read_bytes()
        path.write_bytes(before + b"changed")
        with pytest.raises(ValueError, match="differs"):
            verify(paper, report)
        path.write_bytes(before)
