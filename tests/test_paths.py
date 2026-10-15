from pathlib import Path

from cdo import paths


def test_cluster_root_without_variable(monkeypatch):
    monkeypatch.delenv("CDO_DATA_ROOT", raising=False)
    assert paths.data_root() == Path("/project/nr_dream002")
    assert paths.collection("dream_runs") == Path("/project/nr_dream002/DREAM-runs/output")


def test_variable_sets_root_and_expands_home(monkeypatch):
    monkeypatch.setenv("CDO_DATA_ROOT", "~/Data/DREAM")
    root = Path.home() / "Data" / "DREAM"
    assert paths.data_root() == root
    assert paths.run_roots() == (root / "DREAM-runs/output", root / "DEMO_data_2026/runs")
    assert paths.run_folder("demo_jorek", "tauw_50ms_iso") == \
        root / "DEMO_data_2026/runs/tauw_50ms_iso"


def test_picker_skips_missing_roots(monkeypatch, tmp_path):
    from cdo.picker import survey_runs

    monkeypatch.setenv("CDO_DATA_ROOT", str(tmp_path / "nothing_here"))
    assert survey_runs() == []
