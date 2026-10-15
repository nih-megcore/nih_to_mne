from pathlib import Path

import pytest

from nih2mne.proc.derivative_review import (
    ProcessingScriptError,
    analyze_processing_script,
    evaluate_dataset,
)


SCRIPT = '''
project = "review"
low, high = 1, 40
conditions = ["face", "house"]
noise_run = "01"
recipe = FilterSpec(l_freq=low, h_freq=high, notch_freqs=(60,))
primary = get_bids_path_from_fname(dataset)
noise = primary.copy().update(task="EmptyRoom", run=noise_run)
template = MEGDerivativeTemplate(primary, project=project, filter_spec=recipe, noise=noise)
artifacts = template.evaluate()
ensure_dataset_description(artifacts["project_dataset_description"], project, logger)
result = load_or_make(artifacts["bem"], maker, saver, logger)
save_filtered(raw, artifacts["filtered_raw"], artifacts["filtered_raw_json"], False, logger)
for condition in conditions:
    native = template.artifact("stc", description=condition)
    save_if_requested(value, native, saver, False, logger)
    fsaverage = template.artifact("stc", description=condition, space="fsaverage")
    save_if_requested(value, fsaverage, saver, False, logger)
raise RuntimeError("this file must never execute")
'''


def test_analyzer_is_static_and_preserves_output_order(tmp_path):
    script = tmp_path / "task_v1.py"
    script.write_text(SCRIPT)

    spec = analyze_processing_script(script)

    assert spec.project == "review"
    assert spec.filter_kwargs == {"l_freq": 1, "h_freq": 40, "notch_freqs": (60,)}
    assert spec.noise_entities == {"task": "EmptyRoom", "run": "01"}
    assert [request.label for request in spec.outputs] == [
        "project_dataset_description",
        "bem",
        "filtered_raw",
        "filtered_raw_json",
        "stc: face",
        "stc: face / space-fsaverage",
        "stc: house",
        "stc: house / space-fsaverage",
    ]


def test_analyzer_rejects_legacy_and_dynamic_scripts(tmp_path):
    legacy = tmp_path / "legacy.py"
    legacy.write_text("output = BIDSPath(root='/tmp').fpath\n")
    with pytest.raises(ProcessingScriptError, match="FilterSpec"):
        analyze_processing_script(legacy)

    dynamic = tmp_path / "dynamic.py"
    dynamic.write_text(
        "project = get_project()\n"
        "recipe = FilterSpec(l_freq=1, h_freq=40)\n"
        "template = MEGDerivativeTemplate(primary, project=project, filter_spec=recipe)\n"
        "artifacts = template.evaluate()\n"
        "save_if_requested(value, artifacts['epochs'], saver, False, logger)\n"
    )
    with pytest.raises(ProcessingScriptError, match="project"):
        analyze_processing_script(dynamic)


def test_evaluate_dataset_uses_template_validation_for_surface_stc(tmp_path):
    script = tmp_path / "task_v1.py"
    script.write_text(
        "project = 'review'\n"
        "recipe = FilterSpec(l_freq=1, h_freq=40)\n"
        "template = MEGDerivativeTemplate(primary, project=project, filter_spec=recipe)\n"
        "record = template.artifact('stc', description='face')\n"
        "save_if_requested(value, record, saver, False, logger)\n"
    )
    raw = tmp_path / "sub-01" / "meg" / "sub-01_task-task_run-01_meg.ds"
    raw.mkdir(parents=True)
    spec = analyze_processing_script(script)

    first = evaluate_dataset(spec, raw)[0]
    assert first.record.status == "missing"
    first.record.component_paths[0].parent.mkdir(parents=True, exist_ok=True)
    first.record.component_paths[0].touch()

    partial = evaluate_dataset(spec, raw)[0]
    assert partial.record.status == "invalid"
    assert not partial.record.exists

    partial.record.component_paths[1].touch()
    complete = evaluate_dataset(spec, raw)[0]
    assert complete.record.exists

