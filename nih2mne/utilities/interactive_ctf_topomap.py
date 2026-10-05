#!/usr/bin/env python3
"""Display a CTF raw browser linked to a single-sample MEG topomap.

Left-click in the main trace area of the MNE raw browser to update the
topomap.  Browser coordinates include ``raw.first_time``; MNE raw sample
indices are relative to the first sample.  The conversion between those two
time systems is kept explicit below and is covered by ``--test-linking``.

By default one channel-centered matrix is constructed and supplied unchanged
to both the raw browser and topomap. This empty-room recording has channel
offsets much larger than its temporal variation. Pass ``--keep-dc`` to use the
literal matrix in both views, or ``--vlim-pt`` to set their shared scale.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import mne
import numpy as np


DEFAULT_DATASET = Path(
    "/fast2/EMPTY/20260916/MEG_EmptyRoom_20260916_001.ds"
)


def prepare_raw(
    dataset: Path,
    *,
    resample_sfreq: float = 300.0,
    highpass: float | None = None,
    lowpass: float | None = 40.0,
    compensation_grade: int = 3,
    duration: float | None = None,
    n_jobs: int = -1,
) -> mne.io.BaseRaw:
    """Read and preprocess CTF data while compensation refs still exist."""
    raw = mne.io.read_raw_ctf(
        dataset,
        system_clock="ignore",
        preload=False,
    )

    if duration is not None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        # tmax is inclusive, hence subtract one original sampling interval.
        tmax = min(duration - 1.0 / raw.info["sfreq"], raw.times[-1])
        raw.crop(tmin=0.0, tmax=tmax)

    raw.load_data()
    raw.resample(resample_sfreq, n_jobs=n_jobs)
    raw.filter(highpass, lowpass, n_jobs=n_jobs)

    print(
        "Changing CTF compensation grade "
        f"{raw.compensation_grade} -> {compensation_grade}"
    )
    raw.apply_gradient_compensation(compensation_grade)
    if raw.compensation_grade != compensation_grade:
        raise RuntimeError(
            "Compensation change failed: "
            f"expected {compensation_grade}, got {raw.compensation_grade}"
        )

    # Compensation must happen before reference channels are discarded.
    picks = mne.pick_types(
        raw.info,
        meg=True,
        ref_meg=False,
        exclude="bads",
    )
    raw.pick(picks)

    channel_types = set(raw.get_channel_types())
    if channel_types != {"mag"}:
        raise RuntimeError(f"Expected only CTF MEG channels, got {channel_types}")
    return raw


@dataclass
class DrawState:
    """Most recently rendered time and data, also used by the link test."""

    time: float
    sample: int
    data_pt: np.ndarray


class LinkedTopomap:
    """Own the figures, data matrix, and browser-to-topomap callback."""

    def __init__(
        self,
        raw: mne.io.BaseRaw,
        *,
        remove_dc: bool = True,
        vlim_pt: float | None = None,
    ):
        self.raw = raw
        self.remove_dc = remove_dc

        self.data_t = raw.get_data()
        if remove_dc:
            # Match the raw browser's default intent: show variation rather
            # than the much larger per-channel static offsets.
            self.channel_dc_t = self.data_t.mean(axis=1, keepdims=True)
            display_t = self.data_t - self.channel_dc_t
        else:
            self.channel_dc_t = np.zeros((len(raw.ch_names), 1))
            display_t = self.data_t

        # MNE represents the CTF axial gradiometers as "mag" in tesla.
        self.display_data_t = display_t
        self.display_data_pt = self.display_data_t * 1e12

        # Give the browser the exact same matrix used by plot_topomap. This is
        # a separate Raw object so the input Raw remains untouched. No further
        # browser-side DC removal or projection is permitted in create_figures.
        self.browser_raw = raw.copy()
        self.browser_raw._data = self.display_data_t.copy()
        np.testing.assert_array_equal(
            self.browser_raw.get_data(),
            self.display_data_t,
        )

        if vlim_pt is None:
            stride = max(1, self.display_data_pt.shape[1] // 100_000)
            self.vlim_pt = float(
                np.nanpercentile(np.abs(self.display_data_pt[:, ::stride]), 99.5)
            )
        else:
            self.vlim_pt = float(vlim_pt)
        if not np.isfinite(self.vlim_pt) or self.vlim_pt <= 0:
            raise RuntimeError(f"Invalid topomap color limit: {self.vlim_pt}")

        self.raw_fig = None
        self.topo_fig = None
        self.topo_ax = None
        self.callback_id = None
        self.time_cursor = None
        self.last_draw: DrawState | None = None

    def create_figures(self, *, show: bool = True) -> None:
        """Create both figures and connect the browser click handler."""
        mne.viz.set_browser_backend("matplotlib")
        self.raw_fig = self.browser_raw.plot(
            butterfly=True,
            # The browser object already contains the final display matrix.
            remove_dc=False,
            proj=False,
            scalings={"mag": self.vlim_pt * 1e-12},
            clipping=None,
            title=f"Shared matrix; scale +/-{self.vlim_pt:g} pT",
            block=False,
            show=show,
        )
        self.topo_fig, self.topo_ax = plt.subplots(
            num="linked_topomap",
            figsize=(6, 5),
            layout="constrained",
        )
        self.callback_id = self.raw_fig.canvas.mpl_connect(
            "button_press_event", self.on_browser_click
        )
        self.time_cursor = self.raw_fig.mne.ax_main.axvline(
            self.raw.first_time,
            color="limegreen",
            linestyle="--",
            linewidth=1.5,
            zorder=20,
        )
        self.draw_topomap(0.0)
        if show:
            self.topo_fig.show()

    def time_to_sample(self, time_relative: float) -> tuple[float, int]:
        """Clip a relative time and convert it to a matrix column."""
        time_relative = float(
            np.clip(time_relative, self.raw.times[0], self.raw.times[-1])
        )
        sample = int(
            self.raw.time_as_index(time_relative, use_rounding=True)[0]
        )
        sample = int(np.clip(sample, 0, self.display_data_pt.shape[1] - 1))
        return time_relative, sample

    def draw_topomap(self, time_relative: float) -> DrawState:
        """Render exactly one column of the display matrix."""
        if self.topo_fig is None or self.topo_ax is None:
            raise RuntimeError("Call create_figures() before drawing")

        time_relative, sample = self.time_to_sample(time_relative)
        data_pt = self.display_data_pt[:, sample].copy()

        cursor_time = self.raw.first_time + time_relative
        if self.time_cursor is not None:
            self.time_cursor.set_xdata([cursor_time, cursor_time])
            self.raw_fig.canvas.draw_idle()

        # This assertion guards against accidental transposition or aliasing.
        np.testing.assert_array_equal(
            data_pt,
            self.display_data_pt[:, sample],
        )

        self.topo_ax.clear()
        mne.viz.plot_topomap(
            data_pt,
            self.raw.info,
            axes=self.topo_ax,
            show=False,
            sensors=True,
            contours=0,
            extrapolate="local",
            cmap="RdBu_r",
            vlim=(-self.vlim_pt, self.vlim_pt),
        )
        mode = "channel-centered" if self.remove_dc else "literal matrix"
        self.topo_ax.set_title(
            f"t = {time_relative:.3f} s, sample = {sample}\n"
            f"grade {self.raw.compensation_grade}, {mode}"
        )
        self.topo_fig.suptitle(f"CTF MEG; fixed scale +/-{self.vlim_pt:.3g} pT")
        self.topo_fig.canvas.draw_idle()
        self.topo_fig.canvas.flush_events()

        self.last_draw = DrawState(time_relative, sample, data_pt)
        peak = self.raw.ch_names[int(np.argmax(np.abs(data_pt)))]
        rms = float(np.sqrt(np.mean(data_pt**2)))
        print(
            f"[draw] t={time_relative:8.3f} s  sample={sample:7d}  "
            f"rms={rms:9.4f} pT  peak={peak}"
        )
        return self.last_draw

    def on_browser_click(self, event) -> None:
        """Convert a raw-browser click to relative time and redraw."""
        if self.raw_fig is None:
            return
        if (
            event.button != 1
            or event.inaxes is not self.raw_fig.mne.ax_main
            or event.xdata is None
        ):
            return

        # Browser traces are drawn at raw.times + raw.first_time. In contrast,
        # time_as_index() expects seconds relative to the first sample.
        time_relative = float(event.xdata - self.raw.first_time)
        self.draw_topomap(time_relative)

    def close(self) -> None:
        """Disconnect callbacks and close figures."""
        if self.raw_fig is not None and self.callback_id is not None:
            self.raw_fig.canvas.mpl_disconnect(self.callback_id)
        for fig in (self.raw_fig, self.topo_fig):
            if fig is not None:
                plt.close(fig)


def test_temporal_linking(viewer: LinkedTopomap) -> None:
    """Exercise the real callback and validate every time/sample/data link."""
    if viewer.raw_fig is None:
        raise RuntimeError("Figures must exist before testing")

    last_time = viewer.raw.times[-1]
    requested_times = np.array([0.0, 0.413, 1.237, 2.611])
    requested_times = requested_times[requested_times <= last_time]
    if len(requested_times) < 3:
        requested_times = np.linspace(0.0, last_time, 3)

    rendered_columns = []
    rendered_frames = []
    for requested_time in requested_times:
        browser_x = float(viewer.raw.first_time + requested_time)
        event = SimpleNamespace(
            button=1,
            inaxes=viewer.raw_fig.mne.ax_main,
            xdata=browser_x,
        )
        viewer.on_browser_click(event)

        state = viewer.last_draw
        if state is None:
            raise AssertionError("Click callback did not draw a topomap")
        expected_time, expected_sample = viewer.time_to_sample(requested_time)
        if state.time != expected_time:
            raise AssertionError(
                f"Time mismatch: rendered {state.time}, expected {expected_time}"
            )
        if state.sample != expected_sample:
            raise AssertionError(
                f"Sample mismatch: rendered {state.sample}, expected {expected_sample}"
            )
        cursor_x = np.asarray(viewer.time_cursor.get_xdata(), dtype=float)
        np.testing.assert_allclose(cursor_x, browser_x, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(
            state.data_pt,
            viewer.display_data_pt[:, expected_sample],
        )
        np.testing.assert_array_equal(
            state.data_pt,
            viewer.browser_raw.get_data(
                start=expected_sample,
                stop=expected_sample + 1,
            )[:, 0]
            * 1e12,
        )
        expected_title_prefix = (
            f"t = {expected_time:.3f} s, sample = {expected_sample}"
        )
        if not viewer.topo_ax.get_title().startswith(expected_title_prefix):
            raise AssertionError(
                "Topomap title does not show the linked time/sample: "
                f"{viewer.topo_ax.get_title()!r}"
            )

        # Force an Agg-compatible render and retain its RGBA pixels. This
        # verifies that changing matrix columns produces a visible update, not
        # merely a change in callback bookkeeping.
        viewer.topo_fig.canvas.draw()
        frame = np.asarray(viewer.topo_fig.canvas.buffer_rgba()).copy()
        rendered_columns.append(state.data_pt)
        rendered_frames.append(frame)
        print(
            f"[test] browser_x={browser_x:.6f} -> "
            f"relative_t={state.time:.6f} -> sample={state.sample}: PASS"
        )

    if not any(
        not np.array_equal(rendered_columns[0], column)
        for column in rendered_columns[1:]
    ):
        raise AssertionError("All tested topomap data columns were identical")
    if not any(
        not np.array_equal(rendered_frames[0], frame)
        for frame in rendered_frames[1:]
    ):
        raise AssertionError("All tested rendered topomap frames were identical")
    print(
        f"[test] Temporal linking passed for {len(requested_times)} times; "
        "the matrix columns and rendered canvas pixels both changed."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_positional",
        nargs="?",
        type=Path,
        default=None,
        metavar="DATASET",
        help="CTF .ds directory (positional form; --dataset is preferred)",
    )
    parser.add_argument(
        "--dataset",
        dest="dataset_option",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"CTF .ds directory (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--resample", type=float, default=300.0)
    parser.add_argument("--highpass", type=float, default=None)
    parser.add_argument("--lowpass", type=float, default=40.0)
    parser.add_argument("--compensation-grade", type=int, default=0)
    parser.add_argument(
        "--vlim-pt",
        type=float,
        default=None,
        metavar="PT",
        help=(
            "Shared symmetric browser/topomap limit in pT; for example, "
            "--vlim-pt 2 gives +/-2 pT (+/-2000 fT)"
        ),
    )
    parser.add_argument(
        "--keep-dc",
        action="store_true",
        help="Plot literal matrix values instead of channel-centered values",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optionally load only the first N seconds",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for resampling/filtering",
    )
    parser.add_argument(
        "--test-linking",
        action="store_true",
        help="Run headless-friendly callback/time/sample validation and exit",
    )
    args = parser.parse_args()
    if args.dataset_option is not None and args.dataset_positional is not None:
        parser.error("provide the dataset either positionally or with --dataset, not both")
    args.dataset = (
        args.dataset_option
        if args.dataset_option is not None
        else args.dataset_positional
        if args.dataset_positional is not None
        else DEFAULT_DATASET
    )
    return args


def main() -> None:
    args = parse_args()
    if not args.dataset.is_dir():
        raise FileNotFoundError(f"CTF dataset not found: {args.dataset}")

    # Keep a test invocation small unless an explicit duration was supplied.
    duration = args.duration
    if args.test_linking and duration is None:
        duration = 3.0

    raw = prepare_raw(
        args.dataset,
        resample_sfreq=args.resample,
        highpass=args.highpass,
        lowpass=args.lowpass,
        compensation_grade=args.compensation_grade,
        duration=duration,
        n_jobs=args.n_jobs,
    )
    viewer = LinkedTopomap(
        raw,
        remove_dc=not args.keep_dc,
        vlim_pt=args.vlim_pt,
    )
    print(
        f"Prepared {len(raw.ch_names)} channels x {raw.n_times} samples; "
        f"sfreq={raw.info['sfreq']:.3f} Hz; "
        f"compensation grade={raw.compensation_grade}; "
        f"topomap vlim=+/-{viewer.vlim_pt:.4g} pT"
    )

    if args.test_linking:
        viewer.create_figures(show=False)
        try:
            test_temporal_linking(viewer)
        finally:
            viewer.close()
        return

    viewer.create_figures(show=True)
    print("Left-click in the raw browser trace area to update the topomap.")
    plt.show(block=True)


if __name__ == "__main__":
    main()
