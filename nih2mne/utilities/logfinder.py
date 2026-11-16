"""Find one task logfile using ``~/megcore/trigproc/{task}_logfind.yml``."""

from glob import glob
from pathlib import Path
import re

import yaml


_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]+\Z")


def _identifier(value, label):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must be a nonempty identifier without path or glob characters")
    return value


def _string_list(value, label):
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{label} must be a nonempty list of strings")
    return value


def logfind(task: str, subject: str) -> str:
    """Return the sole logfile matching a task's configured sources.

    ``subject`` is the MEG subject hash substituted for ``{subject_hash}`` in
    each pattern. Each source has ``directories`` and ``patterns`` lists and an
    optional ``recursive`` boolean. Missing, invalid, and ambiguous searches
    raise an exception instead of selecting an arbitrary file.
    """
    task = _identifier(task, "task")
    subject = _identifier(subject, "subject")
    config_path = Path.home() / "megcore" / "trigproc" / f"{task}_logfind.yml"
    try:
        with config_path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid logfile configuration: {config_path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError(f"Invalid logfile configuration: {config_path}")
    sources = config.get("log_sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{config_path}: log_sources must be a nonempty list")

    matches = set()
    for index, source in enumerate(sources, start=1):
        label = f"{config_path}: log_sources[{index}]"
        if not isinstance(source, dict):
            raise ValueError(f"{label} must be a mapping")
        directories = _string_list(source.get("directories"), f"{label}.directories")
        patterns = _string_list(source.get("patterns"), f"{label}.patterns")
        recursive = source.get("recursive", False)
        if not isinstance(recursive, bool):
            raise ValueError(f"{label}.recursive must be a boolean")
        for directory in directories:
            root = Path(directory).expanduser()
            if not root.is_absolute():
                root = config_path.parent / root
            for pattern in patterns:
                try:
                    relative_pattern = pattern.format(subject_hash=subject)
                except (KeyError, ValueError, IndexError) as exc:
                    raise ValueError(f"{label}: invalid pattern {pattern!r}") from exc
                if Path(relative_pattern).is_absolute() or ".." in Path(relative_pattern).parts:
                    raise ValueError(f"{label}: pattern must stay within its directory")
                search = f"**/{relative_pattern}" if recursive else relative_pattern
                for filename in glob(str(root / search), recursive=recursive):
                    path = Path(filename)
                    if path.is_file():
                        matches.add(str(path.resolve()))

    if not matches:
        raise FileNotFoundError(
            f"No logfile found for task {task!r}, subject {subject!r} using {config_path}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple logfiles found for task {task!r}, subject {subject!r}: "
            + ", ".join(sorted(matches))
        )
    return matches.pop()
