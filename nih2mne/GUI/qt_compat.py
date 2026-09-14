"""Select a supported PyQt binding for the :mod:`nih2mne.GUI` package.

PyQt6 is preferred, while PyQt5 5.15 remains supported for managed
environments that cannot migrate immediately.  ``QT_API`` may be set to
``pyqt6`` or ``pyqt5`` before importing a GUI module to select explicitly.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys


_BINDINGS = {"pyqt6": "PyQt6", "pyqt5": "PyQt5"}


def _loaded_binding() -> str | None:
    """Return the already imported binding, rejecting unsafe mixed use."""
    loaded = [api for api, package in _BINDINGS.items() if package in sys.modules]
    if len(loaded) > 1:
        raise RuntimeError("PyQt5 and PyQt6 are both already imported")
    return loaded[0] if loaded else None


def _select_binding() -> str:
    """Choose the requested, loaded, or first available supported binding."""
    requested = os.environ.get("QT_API", "").strip().lower() or None
    if requested is not None and requested not in _BINDINGS:
        choices = ", ".join(_BINDINGS)
        raise RuntimeError(f"Unsupported QT_API={requested!r}; choose one of: {choices}")

    loaded = _loaded_binding()
    if loaded is not None:
        if requested is not None and requested != loaded:
            raise RuntimeError(
                f"QT_API requests {requested}, but {_BINDINGS[loaded]} is already imported"
            )
        return loaded

    if requested is not None:
        package = _BINDINGS[requested]
        if importlib.util.find_spec(package) is None:
            raise ModuleNotFoundError(
                f"QT_API requests {requested}, but {package} is not installed"
            )
        return requested

    for api, package in _BINDINGS.items():
        if importlib.util.find_spec(package) is not None:
            return api

    raise ModuleNotFoundError(
        "No supported Qt binding is installed; install PyQt6 or PyQt5>=5.15"
    )


QT_API = _select_binding()
QT_BINDING = _BINDINGS[QT_API]
os.environ["QT_API"] = QT_API

QtCore = importlib.import_module(f"{QT_BINDING}.QtCore")
QtGui = importlib.import_module(f"{QT_BINDING}.QtGui")
QtTest = importlib.import_module(f"{QT_BINDING}.QtTest")
QtWidgets = importlib.import_module(f"{QT_BINDING}.QtWidgets")

if QT_API == "pyqt5":
    pyqt_version = tuple(int(part) for part in QtCore.PYQT_VERSION_STR.split(".")[:2])
    if pyqt_version < (5, 15):
        raise ImportError("The nih2mne PyQt5 compatibility path requires PyQt5>=5.15")

__all__ = ["QT_API", "QT_BINDING", "QtCore", "QtGui", "QtTest", "QtWidgets"]
