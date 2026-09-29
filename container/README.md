# Containers

The repository ships one OCI `Containerfile`. Podman builds it directly, and
SingularityCE or Apptainer converts the locally built Podman image into a SIF.
Both formats contain PyQt6, its Qt 6 runtime, MNE Qt Browser, pyqtgraph,
PyVista, Tk, and the Linux libraries needed for X11, Wayland, and OpenGL
visualization.

## Build

From the repository root:

```bash
make podman
make singularity
```

`make podman` creates `localhost/nih2mne:latest`. `make singularity` first
builds that Podman image, saves it to a temporary Docker-format archive, and
converts it to `build/nih2mne.sif`. The archive is removed afterward. Apptainer
is preferred when both `apptainer` and `singularity` are installed.

The names and executables are configurable:

```bash
make podman PODMAN_IMAGE=localhost/nih2mne:dev
make singularity SIF_IMAGE=/path/to/nih2mne.sif SINGULARITY=singularity
```

## Run a GUI on Linux

The GUI targets mount `DATA_DIR` at `/data` and run `bids_qa_gui.py` there.
They prefer Wayland when a usable Wayland socket is present and otherwise use
X11 with the host's Xauthority cookie.

```bash
make podman-gui DATA_DIR=/path/to/bids
make singularity-gui DATA_DIR=/path/to/bids
```

Select a backend or a different installed command when needed:

```bash
make podman-gui GUI_BACKEND=x11 DATA_DIR=/path/to/bids
make podman-gui CONTAINER_CMD='make_meg_bids_gui.py'
make singularity-gui CONTAINER_CMD='bids_qa_gui.py -num_rows 4 -num_cols 5'
```

Accepted `GUI_BACKEND` values are `auto`, `wayland`, and `x11`. If `/dev/dri`
exists, the launcher passes it through for local graphics acceleration. Only
the selected display socket, Xauthority file, and data directory are mounted;
the host home directory is not exposed wholesale.

## Run a GUI on macOS

Podman runs Linux containers in a Podman Machine VM on macOS, so it cannot use
the native Quartz API directly. The launcher uses XQuartz to integrate the
Linux window with the macOS desktop.

1. Install and start [XQuartz](https://www.xquartz.org/).
2. In **XQuartz Settings > Security**, enable **Allow connections from network
   clients**, then quit and restart XQuartz.
3. In an XQuartz terminal, authorize local clients:

   ```bash
   xhost +localhost
   ```

4. Start the GUI from the repository:

   ```bash
   make podman-gui DATA_DIR=/path/to/bids
   ```

The launcher sets the container display to `host.containers.internal:0` and
uses software OpenGL for XQuartz compatibility. It intentionally does not
change XQuartz access-control settings. Revoke the authorization when it is no
longer needed:

```bash
xhost -localhost
```

SingularityCE and Apptainer GUI targets are intended for Linux hosts. Images
are built for the host's native CPU architecture; an arm64 SIF built on Apple
Silicon will not run on an amd64 cluster.

## Build-time Qt verification

The `Containerfile` fails its build unless all of the following work:

- importing PyQt6, MNE Qt Browser, pyqtgraph, PyVista, and tkinter;
- creating an offscreen `QApplication`;
- selecting MNE's Qt browser backend;
- locating both Qt's XCB and Wayland platform plugins.

AFNI, FreeSurfer, cluster schedulers, and other licensed or site-specific
programs are not bundled. Mount any required data using `DATA_DIR`.
