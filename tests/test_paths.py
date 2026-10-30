from pathlib import Path

from cdo import paths


def test_repo_data_root_without_variable(monkeypatch):
    monkeypatch.delenv("CDO_DATA_ROOT", raising=False)
    root = Path(paths.__file__).resolve().parent.parent / "data"
    assert paths.data_root() == root
    assert paths.collection("dream_runs") == root / "DREAM-runs/output"


def test_variable_sets_root_and_expands_home(monkeypatch):
    monkeypatch.setenv("CDO_DATA_ROOT", "~/Data/DREAM")
    root = Path.home() / "Data" / "DREAM"
    assert paths.data_root() == root
    assert paths.run_roots() == (root / "DREAM-runs/output", root / "DEMO_data_2026/runs")
    assert paths.run_folder("demo_jorek", "tauw_50ms_iso") == \
        root / "DEMO_data_2026/runs/tauw_50ms_iso"


def test_figure_root_from_variable_or_repo(monkeypatch):
    monkeypatch.delenv("CDO_FIGURE_ROOT", raising=False)
    assert paths.figure_root() == Path(paths.__file__).resolve().parent.parent / "plots"
    monkeypatch.setenv("CDO_FIGURE_ROOT", "~/plots")
    assert paths.figure_root() == Path.home() / "plots"


def test_picker_skips_missing_roots(monkeypatch, tmp_path):
    from cdo.picker import survey_runs

    monkeypatch.setenv("CDO_DATA_ROOT", str(tmp_path / "nothing_here"))
    assert survey_runs() == []


def _fake_run(folder: Path, legs: int = 1):
    import h5py
    import numpy as np

    folder.mkdir(parents=True)
    for i in range(legs):
        with h5py.File(folder / f"output_{i:04d}.h5", "w") as h:
            h["grid/t"] = np.array([0.0, 1e-3])


def test_survey_lists_campaign_runs_once_with_their_paths(monkeypatch, tmp_path):
    from cdo.picker import survey_runs

    monkeypatch.setenv("CDO_DATA_ROOT", str(tmp_path))
    dream = tmp_path / "DREAM-runs/output"
    demo = tmp_path / "DEMO_data_2026/runs"
    _fake_run(dream / "iter-nrt/run_a")
    _fake_run(dream / "run_a")  # old top-level copy
    _fake_run(dream / "archive-2026-07-23-precut/run_a")
    _fake_run(dream / "archive_2026-07-23_precut/run_a")  # archive twin
    (dream / "iter-nrt/logs").mkdir()
    _fake_run(demo / "tauw_inf_iso")
    _fake_run(demo / "fix-wrapped-bounce-orbits/tauw_inf_iso", legs=2)
    (demo / "tauw_inf_iso/seed").mkdir()  # no committed legs, not a run

    infos = survey_runs()
    assert [i.name for i in infos] == [
        "archive-2026-07-23-precut/run_a", "iter-nrt/run_a",
        "fix-wrapped-bounce-orbits/tauw_inf_iso", "tauw_inf_iso"]
    assert infos[2].legs == 2
    assert infos[2].label.startswith("fix-wrapped-bounce-orbits/tauw_inf_iso  |")


def test_survey_of_given_roots_looks_one_campaign_level_down(tmp_path):
    from cdo.picker import survey_runs

    _fake_run(tmp_path / "top")
    _fake_run(tmp_path / "campaign/inner")
    _fake_run(tmp_path / "campaign/inner/deeper")
    names = [i.name for i in survey_runs([tmp_path])]
    assert names == ["campaign/inner", "top"]
    assert [i.name for i in survey_runs([tmp_path], exclude=["campaign/*"])] == ["top"]
