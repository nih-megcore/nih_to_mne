"""Read-only inspection of outputs declared by NIH2MNE processing scripts.

Processing scripts are parsed with :mod:`ast`; they are never imported or
executed.  This is intentionally limited to scripts using ``FilterSpec`` and
``MEGDerivativeTemplate`` so the displayed paths have the same validation
semantics as the processing routine itself.
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from mne_bids import get_bids_path_from_fname

from nih2mne.bids_templates import ArtifactRecord, FilterSpec, MEGDerivativeTemplate
from nih2mne.proc.megcore_utils import find_emptyroom_dataset


class ProcessingScriptError(ValueError):
    """Raised when a processing script cannot be reviewed safely."""


@dataclass(frozen=True)
class ArtifactRequest:
    """One derivative output, in the order in which the script writes it."""

    key: str
    overrides: tuple[tuple[str, Any], ...] = ()

    @property
    def kwargs(self) -> dict[str, Any]:
        return dict(self.overrides)

    @property
    def label(self) -> str:
        details = []
        if "description" in self.kwargs:
            details.append(str(self.kwargs["description"]))
        if "space" in self.kwargs:
            details.append(f"space-{self.kwargs['space']}")
        return self.key if not details else f"{self.key}: {' / '.join(details)}"


@dataclass(frozen=True)
class ArtifactDebugSpec:
    """Static information used to reproduce one declared output."""

    variable: str
    calls: tuple[str, ...] = ()
    dependencies: tuple[ArtifactRequest, ...] = ()


@dataclass(frozen=True)
class ProcessingReviewSpec:
    """Static template settings and ordered output requests from one script."""

    script: Path
    project: str
    filter_kwargs: Mapping[str, Any]
    noise_entities: Mapping[str, Any] | None
    discover_emptyroom: bool
    space: str | None
    outputs: tuple[ArtifactRequest, ...]
    debug_steps: tuple[ArtifactDebugSpec, ...] = ()
    static_values: Mapping[str, Any] = field(default_factory=dict)

    @property
    def derivative_names(self) -> tuple[str, ...]:
        pipelines = {spec.key: spec.pipeline for spec in MEGDerivativeTemplate.ARTIFACT_SPECS}
        names = []
        for request in self.outputs:
            pipeline = pipelines.get(request.key)
            name = "preprocessing" if pipeline == "preprocessing" else self.project
            if pipeline in {"preprocessing", "project"} and name not in names:
                names.append(name)
        return tuple(names)


@dataclass(frozen=True)
class EvaluatedArtifact:
    """An ordered request paired with its evaluated artifact record."""

    request: ArtifactRequest
    record: ArtifactRecord


@dataclass(frozen=True)
class DebugScript:
    """Clipboard-ready reproduction script and whether it is step-isolated."""

    text: str
    isolated: bool


@dataclass(frozen=True)
class ProcessingFailureReport:
    """Error information attributable to one processing-script execution."""

    summary: str
    details: str
    application_log: Path | None = None
    execution_log: Path | None = None
    error_found: bool = False

    @property
    def display_text(self) -> str:
        lines = [self.summary]
        if self.application_log is not None:
            lines.extend(("", f"Application log: {self.application_log}"))
        if self.execution_log is not None:
            lines.append(f"Execution log: {self.execution_log}")
        if self.details:
            lines.extend(("", self.details.rstrip()))
        return "\n".join(lines)


_MISSING = object()


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _safe_value(node: ast.AST, values: Mapping[str, Any]) -> Any:
    """Evaluate the small literal subset used for generated script settings."""

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id, _MISSING)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_safe_value(item, values) for item in node.elts]
        if _MISSING in items:
            return _MISSING
        if isinstance(node, ast.Tuple):
            return tuple(items)
        if isinstance(node, ast.Set):
            return set(items)
        return items
    if isinstance(node, ast.Dict):
        keys = [_safe_value(item, values) for item in node.keys]
        vals = [_safe_value(item, values) for item in node.values]
        if _MISSING in keys or _MISSING in vals:
            return _MISSING
        return dict(zip(keys, vals))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _safe_value(node.operand, values)
        if value is _MISSING or isinstance(value, bool) or not isinstance(value, (int, float)):
            return _MISSING
        return -value if isinstance(node.op, ast.USub) else value
    return _MISSING


def _render_debug_string(node: ast.AST, values: Mapping[str, Any]) -> str | None:
    """Render a static or formatted provenance call without executing it."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if not isinstance(node, ast.JoinedStr):
        return None
    parts = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
            continue
        if not isinstance(value, ast.FormattedValue):
            return None
        resolved = _safe_value(value.value, values)
        if resolved is _MISSING:
            rendered = ast.unparse(value.value)
        elif value.conversion == ord("r"):
            rendered = repr(resolved)
        else:
            rendered = str(resolved)
        parts.append(rendered)
    return "".join(parts)


def _artifact_reference(
    node: ast.AST,
    values: Mapping[str, Any],
    records: Mapping[str, ArtifactRequest],
) -> ArtifactRequest | None:
    """Resolve an artifact path expression used by provenance inputs."""

    if isinstance(node, ast.Attribute) and node.attr in {"path", "fpath"}:
        node = node.value
    return _artifact_request(node, values, records)


def _literal_keywords(call: ast.Call, values: Mapping[str, Any], context: str) -> dict[str, Any]:
    result = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ProcessingScriptError(f"{context} cannot use **kwargs")
        value = _safe_value(keyword.value, values)
        if value is _MISSING:
            raise ProcessingScriptError(
                f"{context} keyword {keyword.arg!r} is not a static literal"
            )
        result[keyword.arg] = value
    return result


def _artifact_request(node: ast.AST, values: Mapping[str, Any], records: Mapping[str, ArtifactRequest]) -> ArtifactRequest | None:
    if isinstance(node, ast.Name):
        return records.get(node.id)
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "artifacts":
        key = _safe_value(node.slice, values)
        if isinstance(key, str):
            return ArtifactRequest(key)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "artifact" and node.args:
            key = _safe_value(node.args[0], values)
            if not isinstance(key, str):
                raise ProcessingScriptError("template.artifact key must be a static string")
            kwargs = _literal_keywords(node, values, "template.artifact")
            return ArtifactRequest(key, tuple(kwargs.items()))
    return None


@dataclass
class _Analyzer:
    values: dict[str, Any] = field(default_factory=dict)
    records: dict[str, ArtifactRequest] = field(default_factory=dict)
    filter_kwargs: dict[str, Any] | None = None
    template_kwargs: dict[str, Any] | None = None
    outputs: list[ArtifactRequest] = field(default_factory=list)
    debug_steps: list[ArtifactDebugSpec] = field(default_factory=list)

    def _bind(self, target: ast.AST, value: Any) -> None:
        if isinstance(target, ast.Name):
            self.values[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)):
            if len(target.elts) == len(value):
                for child, item in zip(target.elts, value):
                    self._bind(child, item)

    def process(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                self._assignment(statement)
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                self._output_call(statement.value)
            elif isinstance(statement, ast.For):
                self._for(statement)
            elif isinstance(statement, ast.If):
                # Both branches describe possible outputs. Generated routines
                # normally put the same artifact write in only one branch.
                self.process(statement.body)
                self.process(statement.orelse)
            elif isinstance(statement, ast.With):
                self.process(statement.body)
            elif isinstance(statement, ast.Try):
                self.process(statement.body)
                for handler in statement.handlers:
                    self.process(handler.body)
                self.process(statement.orelse)
                self.process(statement.finalbody)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == "main":
                self.process(statement.body)

    def _assignment(self, statement: ast.Assign | ast.AnnAssign) -> None:
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value_node = statement.value
        if value_node is None:
            return
        value = _safe_value(value_node, self.values)
        for target in targets:
            if value is not _MISSING:
                self._bind(target, value)

        if isinstance(value_node, ast.Call):
            name = _call_name(value_node.func)
            if name == "FilterSpec":
                self.filter_kwargs = _literal_keywords(value_node, self.values, "FilterSpec")
            elif name == "MEGDerivativeTemplate":
                self.template_kwargs = {}
                for keyword in value_node.keywords:
                    if keyword.arg not in {"project", "space", "noise_entities"}:
                        continue
                    value = _safe_value(keyword.value, self.values)
                    if value is _MISSING:
                        raise ProcessingScriptError(
                            f"MEGDerivativeTemplate keyword {keyword.arg!r} is not a static literal"
                        )
                    self.template_kwargs[keyword.arg] = value

            request = _artifact_request(value_node, self.values, self.records)
            if request is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        self.records[target.id] = request
            result_variable = next(
                (
                    target.id for target in targets
                    if isinstance(target, ast.Name)
                ),
                None,
            )
            self._output_call(value_node, result_variable=result_variable)

    def _for(self, statement: ast.For) -> None:
        iterable = _safe_value(statement.iter, self.values)
        if iterable is _MISSING or not isinstance(statement.target, ast.Name):
            output_names = {
                "ensure_dataset_description", "load_or_make", "save_filtered",
                "save_if_requested", "artifact",
            }
            contains_output = any(
                isinstance(node, ast.Call) and _call_name(node.func) in output_names
                for child in statement.body for node in ast.walk(child)
            )
            if contains_output:
                raise ProcessingScriptError(
                    "output-producing loops must use a static literal iterable"
                )
            return
        previous = self.values.get(statement.target.id, _MISSING)
        for item in iterable:
            self.values[statement.target.id] = item
            self.process(statement.body)
        if previous is _MISSING:
            self.values.pop(statement.target.id, None)
        else:
            self.values[statement.target.id] = previous
        self.process(statement.orelse)

    @staticmethod
    def _default_variable(request: ArtifactRequest) -> str:
        variables = {
            "fwd": "forward",
            "filtered_raw": "raw",
            "filtered_raw_json": "filtered_raw_metadata",
            "covariance": "data_cov",
            "lcmv": "filters",
            "noise_filtered_raw": "noise_raw",
            "noise_filtered_raw_json": "noise_filtered_raw_metadata",
            "noise_covariance": "noise_cov",
        }
        if request.key == "stc":
            description = str(request.kwargs.get("description", "result"))
            safe_description = re.sub(r"\W+", "_", description).strip("_")
            return f"stc_{safe_description or 'result'}"
        return variables.get(request.key, request.key)

    def _debug_spec(
        self,
        call: ast.Call,
        request: ArtifactRequest,
        result_variable: str | None,
    ) -> ArtifactDebugSpec:
        name = _call_name(call.func)
        variable = result_variable
        if name in {"save_filtered", "save_if_requested"} and call.args:
            value = call.args[0]
            if isinstance(value, ast.Name):
                variable = value.id
        if request.key.endswith("_json"):
            variable = self._default_variable(request)
        variable = variable or self._default_variable(request)

        provenance = next(
            (keyword.value for keyword in call.keywords
             if keyword.arg == "provenance"),
            None,
        )
        calls: list[str] = []
        dependencies: list[ArtifactRequest] = []
        if isinstance(provenance, ast.Call) and _call_name(provenance.func) == "derivative":
            calls_node = next(
                (keyword.value for keyword in provenance.keywords
                 if keyword.arg == "calls"),
                None,
            )
            if isinstance(calls_node, (ast.Tuple, ast.List)):
                for item in calls_node.elts:
                    rendered = _render_debug_string(item, self.values)
                    if rendered is not None:
                        calls.append(rendered)
            inputs_node = next(
                (keyword.value for keyword in provenance.keywords
                 if keyword.arg == "inputs"),
                None,
            )
            if isinstance(inputs_node, ast.Dict):
                for value in inputs_node.values:
                    dependency = _artifact_reference(
                        value, self.values, self.records
                    )
                    if dependency is not None and dependency not in dependencies:
                        dependencies.append(dependency)
        return ArtifactDebugSpec(variable, tuple(calls), tuple(dependencies))

    def _output_call(
        self, call: ast.Call, result_variable: str | None = None
    ) -> None:
        name = _call_name(call.func)
        indexes = {
            "ensure_dataset_description": (0,),
            "load_or_make": (0,),
            "save_filtered": (1, 2),
            "save_if_requested": (1,),
        }.get(name, ())
        for index in indexes:
            if len(call.args) <= index:
                continue
            request = _artifact_request(call.args[index], self.values, self.records)
            if request is None:
                raise ProcessingScriptError(
                    f"could not resolve derivative record passed to {name}"
                )
            self.outputs.append(request)
            self.debug_steps.append(
                self._debug_spec(call, request, result_variable)
            )


def _noise_configuration(
    tree: ast.AST,
    template_kwargs: Mapping[str, Any] | None,
    values: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, bool]:
    """Recover the statically declared noise-input strategy."""

    template_call = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node.func) == "MEGDerivativeTemplate":
            template_call = node
            break
    if template_call is None:
        return None, False
    noise_node = next((kw.value for kw in template_call.keywords if kw.arg == "noise"), None)
    if noise_node is None:
        return None, False
    if isinstance(noise_node, ast.Name):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == noise_node.id for target in node.targets):
                continue
            if isinstance(node.value, ast.Call) and _call_name(node.value.func) == "update":
                return (
                    _literal_keywords(node.value, values, "noise BIDSPath update"),
                    False,
                )
            if (
                isinstance(node.value, ast.Call)
                and _call_name(node.value.func) == "find_emptyroom_dataset"
            ):
                valid_primary = (
                    len(node.value.args) == 1
                    and isinstance(node.value.args[0], ast.Name)
                    and node.value.args[0].id == "primary"
                    and not node.value.keywords
                )
                if not valid_primary:
                    raise ProcessingScriptError(
                        "find_emptyroom_dataset must be called as "
                        "find_emptyroom_dataset(primary)"
                    )
                return None, True
    if template_kwargs and isinstance(template_kwargs.get("noise_entities"), Mapping):
        return dict(template_kwargs["noise_entities"]), False
    raise ProcessingScriptError(
        "the MEGDerivativeTemplate noise input must use a static BIDSPath update "
        "or find_emptyroom_dataset(primary)"
    )


def analyze_processing_script(script: str | Path) -> ProcessingReviewSpec:
    """Parse a template-based processing script without importing it."""

    path = Path(script).expanduser()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except OSError as error:
        raise ProcessingScriptError(f"could not read processing script: {error}") from error
    except SyntaxError as error:
        raise ProcessingScriptError(f"processing script has invalid Python syntax: {error}") from error

    analyzer = _Analyzer()
    analyzer.process(tree.body)
    if analyzer.filter_kwargs is None or analyzer.template_kwargs is None:
        raise ProcessingScriptError(
            "ProcReview supports scripts using FilterSpec and MEGDerivativeTemplate"
        )
    project = analyzer.template_kwargs.get("project")
    if not isinstance(project, str) or not project:
        raise ProcessingScriptError("MEGDerivativeTemplate project must be a static string")
    if not analyzer.outputs:
        raise ProcessingScriptError("no template-based derivative writes were found")
    space = analyzer.template_kwargs.get("space")
    if space is not None and not isinstance(space, str):
        raise ProcessingScriptError("MEGDerivativeTemplate space must be a string or None")
    noise_entities, discover_emptyroom = _noise_configuration(
        tree, analyzer.template_kwargs, analyzer.values
    )
    return ProcessingReviewSpec(
        script=path,
        project=project,
        filter_kwargs=dict(analyzer.filter_kwargs),
        noise_entities=noise_entities,
        discover_emptyroom=discover_emptyroom,
        space=space,
        outputs=tuple(analyzer.outputs),
        debug_steps=tuple(analyzer.debug_steps),
        static_values=dict(analyzer.values),
    )


def evaluate_dataset(spec: ProcessingReviewSpec, dataset: str | Path) -> tuple[EvaluatedArtifact, ...]:
    """Evaluate the ordered derivative records for one raw BIDS dataset."""

    primary = get_bids_path_from_fname(str(dataset), check=False)
    noise = None
    if spec.discover_emptyroom:
        noise = find_emptyroom_dataset(primary)
    elif spec.noise_entities is not None:
        noise = primary.copy().update(**dict(spec.noise_entities))
    template = MEGDerivativeTemplate(
        primary,
        project=spec.project,
        filter_spec=FilterSpec(**dict(spec.filter_kwargs)),
        noise=noise,
        space=spec.space,
    )
    records = []
    for request in spec.outputs:
        records.append(EvaluatedArtifact(request, template.artifact(request.key, **request.kwargs)))
    return tuple(records)


def _absolute_path(path: str | Path) -> Path:
    """Return a lexical absolute path without dereferencing BIDS symlinks."""

    return Path(os.path.abspath(Path(path).expanduser()))


def artifact_copy_text(record: ArtifactRecord) -> str:
    """Return the absolute path copied by a ProcReview CP button."""

    # MNE reads a surface STC from its common, extensionless stem.
    return str(_absolute_path(record.path))


def _default_debug_variable(request: ArtifactRequest) -> str:
    variables = {
        "fwd": "forward",
        "filtered_raw": "raw",
        "filtered_raw_json": "filtered_raw_metadata",
        "covariance": "data_cov",
        "lcmv": "filters",
        "noise_filtered_raw": "noise_raw",
        "noise_filtered_raw_json": "noise_filtered_raw_metadata",
        "noise_covariance": "noise_cov",
    }
    if request.key == "stc":
        description = str(request.kwargs.get("description", "result"))
        description = re.sub(r"\W+", "_", description).strip("_")
        return f"stc_{description or 'result'}"
    return variables.get(request.key, request.key)


def artifact_load_statement(
    request: ArtifactRequest,
    record: ArtifactRecord | None = None,
    *,
    path: str | Path | None = None,
    variable: str | None = None,
) -> str:
    """Build one directly runnable artifact-loading assignment."""

    if path is None:
        if record is None:
            raise ValueError("record or path is required")
        path = record.path
    path = _absolute_path(path)
    path_literal = repr(str(path))
    variable = variable or _default_debug_variable(request)
    key = request.key

    if key in {"raw", "noise_raw"}:
        lower_name = path.name.lower()
        if lower_name.endswith(".ds"):
            expression = (
                f"mne.io.read_raw_ctf({path_literal}, preload=True, "
                "system_clock='ignore', clean_names=True)"
            )
        elif lower_name.endswith((".fif", ".fif.gz")):
            expression = f"mne.io.read_raw_fif({path_literal}, preload=True)"
        elif lower_name.endswith((".con", ".sqd")):
            expression = f"mne.io.read_raw_kit({path_literal}, preload=True)"
        elif lower_name.endswith(".pdf"):
            expression = f"mne.io.read_raw_bti({path_literal}, preload=True)"
        else:
            expression = f"mne.io.read_raw({path_literal}, preload=True)"
    elif key in {
        "preprocessing_dataset_description", "project_dataset_description",
        "filtered_raw_json", "noise_filtered_raw_json",
    }:
        expression = (
            f"json.loads(Path({path_literal}).read_text(encoding='utf-8'))"
        )
    elif key in {"filtered_raw", "noise_filtered_raw"}:
        expression = f"mne.io.read_raw_fif({path_literal}, preload=True, verbose=None)"
    elif key in {"epochs", "noise_epochs"}:
        expression = f"mne.read_epochs({path_literal}, preload=True, verbose=None)"
    elif key in {"covariance", "noise_covariance"}:
        expression = f"mne.read_cov({path_literal}, verbose=None)"
    elif key == "bem":
        expression = f"mne.read_bem_solution({path_literal}, verbose=None)"
    elif key == "src":
        expression = f"mne.read_source_spaces({path_literal}, verbose=None)"
    elif key == "trans":
        expression = f"mne.read_trans({path_literal}, verbose=None)"
    elif key == "fwd":
        expression = f"mne.read_forward_solution({path_literal}, verbose=None)"
    elif key == "ica":
        expression = f"mne.preprocessing.read_ica({path_literal}, verbose=None)"
    elif key == "ica_dir":
        expression = f"Path({path_literal})"
    elif key == "lcmv":
        expression = f"mne.beamformer.read_beamformer({path_literal})"
    elif key == "stc":
        expression = f"mne.read_source_estimate({path_literal})"
    else:
        raise ValueError(f"no debug loader is registered for {key!r}")
    return f"{variable} = {expression}"


def _runnable_calls(
    request: ArtifactRequest, debug: ArtifactDebugSpec
) -> tuple[str, ...] | None:
    calls = list(debug.calls)
    if not calls:
        return None
    key = request.key
    variable = debug.variable
    if key == "bem" and len(calls) >= 2:
        calls[0] = f"model = {calls[0]}"
        calls[1] = f"{variable} = {calls[1]}"
    elif key == "epochs" and len(calls) >= 3:
        calls[0] = f"{variable} = {calls[0]}"
        calls[-1] = f"{variable}, reject_log = {calls[-1]}"
    elif key == "stc":
        if request.kwargs.get("space") == "fsaverage" and len(calls) >= 2:
            calls[0] = f"morph = {calls[0]}"
            calls[1] = f"{variable} = {calls[1]}"
        elif len(calls) >= 1 and calls[0].lstrip().startswith("apply_lcmv_epochs("):
            generator = calls[0]
            calls = [
                "stcs = [",
                "    stc.apply_baseline((None, 0))",
                f"    for stc in {generator}",
                "]",
                f"{variable} = stcs[0].copy()",
                "for stc in stcs[1:]:",
                f"    {variable} += stc",
                f"{variable} /= len(stcs)",
            ]
        else:
            return None
    elif key in {
        "src", "trans", "fwd", "covariance", "noise_covariance", "lcmv",
    }:
        calls[0] = f"{variable} = {calls[0]}"
    elif key in {"filtered_raw", "noise_filtered_raw"}:
        pass
    else:
        return None
    source = "\n".join(calls)
    try:
        ast.parse(source)
    except SyntaxError:
        return None
    return tuple(calls)


def _commented_save_statement(
    request: ArtifactRequest, record: ArtifactRecord, variable: str
) -> str | None:
    path_literal = repr(str(_absolute_path(record.path)))
    key = request.key
    if key == "bem":
        command = f"mne.write_bem_solution({path_literal}, {variable}, overwrite=False)"
    elif key == "src":
        command = f"{variable}.save({path_literal}, overwrite=False)"
    elif key == "trans":
        command = f"mne.write_trans({path_literal}, {variable}, overwrite=False)"
    elif key == "fwd":
        command = f"mne.write_forward_solution({path_literal}, {variable}, overwrite=False)"
    elif key in {"filtered_raw", "noise_filtered_raw", "epochs", "noise_epochs"}:
        command = f"{variable}.save({path_literal}, overwrite=False)"
    elif key in {"covariance", "noise_covariance"}:
        command = f"{variable}.save({path_literal}, overwrite=False)"
    elif key == "lcmv":
        command = f"{variable}.save({path_literal}, overwrite=False)"
    elif key == "stc":
        command = f"{variable}.save({path_literal}, ftype='stc', overwrite=False)"
    else:
        return None
    return f"# {command}"


def _full_script_fallback(
    spec: ProcessingReviewSpec,
    dataset: str | Path,
    *,
    mri_path: str | Path | None,
) -> DebugScript:
    command = f"%run {shlex.quote(str(_absolute_path(spec.script)))} "
    command += shlex.quote(str(_absolute_path(dataset)))
    if mri_path is not None:
        command += f" -mri {shlex.quote(str(_absolute_path(mri_path)))}"
        mri_note = ""
    else:
        mri_note = "\n# No QA-selected MRI was available; the script will use automatic T1 discovery."
    warning = (
        "# WARNING: an isolated recipe could not be generated for this step.\n"
        "# This reruns the full processing script from the beginning; cached steps\n"
        "# should be reused and execution is expected to proceed only to the failing step."
    )
    return DebugScript(f"{warning}{mri_note}\n{command}\n", isolated=False)


def build_debug_script(
    spec: ProcessingReviewSpec,
    dataset: str | Path,
    selected_index: int,
    evaluated_items: tuple[EvaluatedArtifact, ...],
    *,
    mri_path: str | Path | None = None,
    subjects_dir: str | Path | None = None,
) -> DebugScript:
    """Build a self-contained IPython block for one reviewed output."""

    if not (0 <= selected_index < len(evaluated_items)):
        raise IndexError("selected output index is out of range")
    if selected_index >= len(spec.debug_steps):
        return _full_script_fallback(spec, dataset, mri_path=mri_path)

    selected = evaluated_items[selected_index]
    selected_debug = spec.debug_steps[selected_index]
    calls = _runnable_calls(selected.request, selected_debug)
    save = _commented_save_statement(
        selected.request, selected.record, selected_debug.variable
    )
    if calls is None or save is None:
        return _full_script_fallback(spec, dataset, mri_path=mri_path)
    if any("anat" in call for call in calls) and mri_path is None:
        return _full_script_fallback(spec, dataset, mri_path=None)

    output_indexes = {
        request: index for index, request in enumerate(spec.outputs)
    }
    required: set[ArtifactRequest] = set()

    def add_dependencies(request: ArtifactRequest) -> None:
        index = output_indexes.get(request)
        if index is None or index >= len(spec.debug_steps):
            required.add(request)
            return
        for dependency in spec.debug_steps[index].dependencies:
            if dependency not in required:
                required.add(dependency)
                add_dependencies(dependency)

    for dependency in selected_debug.dependencies:
        required.add(dependency)
        add_dependencies(dependency)

    evaluated_by_request = {item.request: item for item in evaluated_items}
    dependency_lines: list[str] = []
    ordered = sorted(
        required,
        key=lambda request: output_indexes.get(request, -1),
    )
    for dependency in ordered:
        producer_index = output_indexes.get(dependency)
        variable = (
            spec.debug_steps[producer_index].variable
            if producer_index is not None and producer_index < len(spec.debug_steps)
            else _default_debug_variable(dependency)
        )
        item = evaluated_by_request.get(dependency)
        try:
            if item is not None:
                line = artifact_load_statement(
                    dependency, item.record, variable=variable
                )
            elif dependency.key == "raw":
                line = artifact_load_statement(
                    dependency, path=dataset, variable=variable
                )
            elif dependency.key == "noise_raw":
                primary = get_bids_path_from_fname(str(dataset), check=False)
                noise = find_emptyroom_dataset(primary)
                noise_path = noise.fpath if hasattr(noise, "fpath") else noise
                line = artifact_load_statement(
                    dependency, path=noise_path, variable=variable
                )
            else:
                return _full_script_fallback(
                    spec, dataset, mri_path=mri_path
                )
        except (OSError, TypeError, ValueError):
            return _full_script_fallback(spec, dataset, mri_path=mri_path)
        if line not in dependency_lines:
            dependency_lines.append(line)

    dataset_path = _absolute_path(dataset)
    try:
        primary = get_bids_path_from_fname(str(dataset_path), check=False)
        bids_root = _absolute_path(primary.root)
        fs_subject = f"sub-{primary.subject}"
    except Exception:
        return _full_script_fallback(spec, dataset, mri_path=mri_path)
    subjects_dir = _absolute_path(
        subjects_dir
        or bids_root / "derivatives" / "freesurfer" / "subjects"
    )
    setup = [
        "from pathlib import Path",
        "import json",
        "import mne",
        "import mne_bids",
        "import numpy as np",
        "from autoreject import AutoReject",
        "from mne.beamformer import apply_lcmv_epochs, make_lcmv",
        "from nih2mne.bids_templates import FilterSpec",
        "from nih2mne.proc.megcore_utils import normalize_beamfilt_ori",
        "",
        f"meg_path = Path({str(dataset_path)!r})",
        "primary = mne_bids.get_bids_path_from_fname(str(meg_path), check=False)",
        f"subjects_dir = Path({str(subjects_dir)!r})",
        f"fs_subject = {fs_subject!r}",
        "n_jobs = -1",
        f"recipe = FilterSpec(**{dict(spec.filter_kwargs)!r})",
    ]
    if mri_path is not None:
        setup.extend([
            f"mri_path = Path({str(_absolute_path(mri_path))!r})",
            "anat = mne_bids.get_bids_path_from_fname(str(mri_path), check=False)",
        ])
    if any("event_id" in call or "events" in call for call in calls):
        conditions = (
            spec.static_values.get("conds_OI")
            or spec.static_values.get("conditions")
        )
        if not isinstance(conditions, (list, tuple)) or not conditions:
            return _full_script_fallback(spec, dataset, mri_path=mri_path)
        setup.extend([
            f"conditions = {list(conditions)!r}",
            "events, available_event_id = mne.events_from_annotations(raw)",
            "event_id = {name: code for name, code in available_event_id.items() "
            "if name in conditions}",
        ])
    sections = [
        "\n".join(setup),
        "# Load required inputs\n" + "\n".join(dependency_lines),
        "# Reproduce the selected processing step\n" + "\n".join(calls),
        "# Uncomment only if you intend to write into the derivatives tree\n" + save,
    ]
    return DebugScript("\n\n".join(sections) + "\n", isolated=True)


_START_RE = re.compile(
    r"START\s+::\s*(?P<legacy>[0-9a-f]+)|"
    r"START\b[^\n]*?\bhash=(?P<current>[0-9a-f]+)",
    re.IGNORECASE,
)
_PRIMARY_RE = re.compile(r"Primary input:\s*(?P<path>.+?)\s*$", re.MULTILINE)
_SLURM_RE = re.compile(r"SLURM_LOGFILE_PATH\s*::\s*(?P<path>.+?)\s*$", re.MULTILINE)


def _script_hash(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def _same_path(first: str | Path, second: str | Path) -> bool:
    try:
        first_path = Path(first).expanduser().resolve(strict=False)
        second_path = Path(second).expanduser().resolve(strict=False)
        return first_path == second_path
    except (OSError, RuntimeError):
        return str(first) == str(second)


def _error_section(text: str) -> str:
    """Return the final complete traceback chain or ERROR log section."""

    traceback_header = "Traceback (most recent call last):"
    traceback_at = text.rfind(traceback_header)
    if traceback_at >= 0:
        chain_start = traceback_at
        chain_separator = re.compile(
            r"\n(?:The above exception was the direct cause of the following "
            r"exception|During handling of the above exception, another "
            r"exception occurred):\s*\Z"
        )
        while True:
            separator = chain_separator.search(text[:chain_start])
            if separator is None:
                break
            previous_traceback = text.rfind(
                traceback_header, 0, separator.start()
            )
            if previous_traceback < 0:
                break
            chain_start = previous_traceback
        return text[chain_start:].strip()
    lines = text.splitlines()
    error_lines = [
        index for index, line in enumerate(lines)
        if re.search(r"\bERROR\b", line)
    ]
    if error_lines:
        return "\n".join(lines[error_lines[-1]:]).strip()
    return ""


def _execution_error(attempt: str) -> tuple[Path | None, str]:
    matches = list(_SLURM_RE.finditer(attempt))
    if not matches:
        return None, ""
    stdout_path = Path(matches[-1].group("path").strip()).expanduser()
    candidates = []
    if stdout_path.suffix == ".o":
        candidates.append(stdout_path.with_suffix(".e"))
    candidates.append(stdout_path)
    last_read = None
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        last_read = candidate
        section = _error_section(text)
        if section:
            return candidate, section
        if text.strip() and candidate.suffix == ".e":
            return candidate, text.strip()
    return last_read, ""


def _log_modification_time(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def find_processing_failure(
    spec: ProcessingReviewSpec,
    dataset: str | Path,
    *,
    allow_legacy_fallback: bool = False,
) -> ProcessingFailureReport:
    """Find the latest logged failure for this script and exact BIDS dataset.

    Logs are only associated when their START marker contains the selected
    script's current Git blob hash.  Logs predating primary-input records may
    be used only when the caller knows the subject/task selection is unique.
    """

    dataset = Path(dataset).expanduser()
    try:
        primary = get_bids_path_from_fname(str(dataset), check=False)
        root = Path(primary.root)
        subject = f"sub-{primary.subject}"
    except (TypeError, ValueError, AttributeError) as error:
        return ProcessingFailureReport(
            "The dataset path could not be decoded, so its processing log "
            "cannot be located.",
            str(error),
        )

    try:
        process_hash = _script_hash(spec.script)
    except OSError as error:
        return ProcessingFailureReport(
            "The selected script could not be hashed, so an execution cannot "
            "be matched safely.",
            str(error),
        )

    log_dir = root / "derivatives" / spec.project / "logging" / subject
    attempts = []
    logfiles = (
        sorted(log_dir.glob("*.log"), key=_log_modification_time)
        if log_dir.is_dir()
        else ()
    )
    for logfile in logfiles:
        try:
            text = logfile.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        starts = list(_START_RE.finditer(text))
        for index, start in enumerate(starts):
            logged_hash = start.group("legacy") or start.group("current")
            if logged_hash.lower() != process_hash.lower():
                continue
            end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
            excerpt = text[start.start():end]
            primary_match = _PRIMARY_RE.search(excerpt)
            logged_primary = primary_match.group("path").strip() if primary_match else None
            attempts.append((logfile, excerpt, logged_primary))

    exact = [item for item in attempts if item[2] and _same_path(item[2], dataset)]
    if exact:
        candidates = exact
    else:
        identified = [item for item in attempts if item[2]]
        legacy = [item for item in attempts if item[2] is None]
        if identified:
            return ProcessingFailureReport(
                "Matching script executions were logged, but none were for this dataset.",
                f"Searched {log_dir} for script hash {process_hash}.",
            )
        if legacy and not allow_legacy_fallback:
            return ProcessingFailureReport(
                "A legacy execution was found, but its log does not identify the input run.",
                "This subject has multiple matching datasets, so displaying "
                "that error could attribute it to the wrong run.",
                application_log=legacy[-1][0],
            )
        candidates = legacy

    if not candidates:
        return ProcessingFailureReport(
            "No execution log matched this dataset and the selected version of the script.",
            f"Searched {log_dir} for script hash {process_hash}.",
        )

    application_log, attempt, _ = candidates[-1]
    finish_pattern = (
        rf"FINISHED(?:\s+::|\b[^\n]*?\bhash=)\s*"
        rf"{re.escape(process_hash)}"
    )
    finished = bool(re.search(finish_pattern, attempt, re.IGNORECASE))
    if finished:
        return ProcessingFailureReport(
            "The latest matching execution finished successfully, but this artifact is now missing or invalid.",
            "The derivative may have been removed, changed, or become incomplete after processing finished.",
            application_log=application_log,
        )

    execution_log, details = _execution_error(attempt)
    if not details:
        details = _error_section(attempt)
    if details:
        return ProcessingFailureReport(
            "The latest matching execution stopped before the pipeline completed.",
            details,
            application_log=application_log,
            execution_log=execution_log,
            error_found=True,
        )
    return ProcessingFailureReport(
        "The latest matching execution stopped, but no logged traceback was found.",
        "The job may have been interrupted before Python recorded an exception.",
        application_log=application_log,
        execution_log=execution_log,
    )


__all__ = [
    "ArtifactRequest",
    "EvaluatedArtifact",
    "ProcessingReviewSpec",
    "ProcessingScriptError",
    "ProcessingFailureReport",
    "analyze_processing_script",
    "evaluate_dataset",
    "find_processing_failure",
]
