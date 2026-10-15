"""Selection logic of the run picker, driven through its widgets without a browser."""

from pathlib import Path

import pytest

from cdo.picker import RunInfo, RunPicker

pytest.importorskip("ipywidgets")


def _infos(*names):
    return [RunInfo(Path(f"/runs/{n}"), legs=1, t_end_ms=1.0, commit="abc",
                    avalanche="fluid", unfinished_leg=False) for n in names]


def test_multiple_keeps_every_ticked_run_in_list_order():
    picker = RunPicker(_infos("a", "b", "c"), multiple=True)
    picker.widget()
    picker._checks[2].value = True
    picker._checks[0].value = True
    assert [p.name for p in picker.selected] == ["a", "c"]


def test_filter_hides_rows_but_keeps_the_choice():
    picker = RunPicker(_infos("tauw_a", "iter_b", "tauw_c"), multiple=True)
    box = picker.widget()
    picker._checks[1].value = True
    filt = box.children[0].children[0]
    filt.value = "tauw"
    assert picker._checks[1].layout.display == "none"
    assert [p.name for p in picker.selected] == ["iter_b"]


def test_status_line_follows_the_choice():
    picker = RunPicker(_infos("a", "b"), multiple=True)
    box = picker.widget()
    status = box.children[-1]
    assert "Selected (0)" in status.value
    picker._checks[1].value = True
    assert "Selected (1)" in status.value and "b" in status.value


def test_single_selects_exactly_one_and_survives_filtering():
    picker = RunPicker(_infos("tauw_a", "iter_b"), multiple=False)
    box = picker.widget()
    assert picker.selected == []
    picker._select.value = Path("/runs/iter_b")
    box.children[0].value = "tauw"  # the chosen run stays listed
    assert [p.name for p in picker.selected] == ["iter_b"]


def test_choose_by_number_without_widgets():
    picker = RunPicker(_infos("a", "b", "c"), multiple=True)
    assert [p.name for p in picker.choose(0, 2)] == ["a", "c"]
    single = RunPicker(_infos("a", "b"), multiple=False)
    with pytest.raises(ValueError):
        single.choose(0, 1)
