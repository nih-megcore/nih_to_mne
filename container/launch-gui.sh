#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'EOF'
Usage: launch-gui.sh podman IMAGE COMMAND
       launch-gui.sh singularity IMAGE COMMAND

Environment:
  DATA_DIR       Host directory exposed as /data (default: current directory)
  GUI_BACKEND    auto, wayland, or x11 (default: auto)
  PODMAN         Podman executable (default: podman)
  SINGULARITY    Apptainer/Singularity executable (auto-detected)
EOF
}

die() {
    printf 'nih2mne container: %s\n' "$*" >&2
    exit 1
}

if [[ $# -ne 3 ]]; then
    usage
    exit 2
fi

engine=$1
image=$2
container_command=$3
backend=${GUI_BACKEND:-auto}
data_dir=${DATA_DIR:-$PWD}
host_os=$(uname -s)

[[ -d $data_dir ]] || die "DATA_DIR is not a directory: $data_dir"
data_dir=$(cd "$data_dir" && pwd -P)

case "$backend" in
    auto|wayland|x11) ;;
    *) die "GUI_BACKEND must be auto, wayland, or x11 (got: $backend)" ;;
esac

if [[ $host_os == Darwin ]]; then
    [[ $engine == podman ]] || die "Singularity/Apptainer GUI launch is supported on Linux; use Podman with XQuartz on macOS"
    [[ $backend != wayland ]] || die "Wayland is not available on macOS; use GUI_BACKEND=x11"
    backend=x11
    if [[ ! -d /Applications/Utilities/XQuartz.app && ! -x /opt/X11/bin/Xquartz ]]; then
        die "XQuartz is required. Install and start XQuartz, enable network clients, and authorize localhost as described in container/README.md"
    fi
elif [[ $host_os == Linux ]]; then
    if [[ $backend == auto ]]; then
        if [[ -n ${WAYLAND_DISPLAY:-} && -n ${XDG_RUNTIME_DIR:-} && -S ${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY} ]]; then
            backend=wayland
        else
            backend=x11
        fi
    fi
else
    die "unsupported host operating system: $host_os"
fi

display_env=("QT_API=pyqt6")
podman_display_args=()
singularity_display_args=()

if [[ $backend == wayland ]]; then
    [[ $host_os == Linux ]] || die "Wayland forwarding is only supported on Linux"
    [[ -n ${WAYLAND_DISPLAY:-} ]] || die "WAYLAND_DISPLAY is not set"
    [[ -n ${XDG_RUNTIME_DIR:-} ]] || die "XDG_RUNTIME_DIR is not set"
    wayland_socket=${XDG_RUNTIME_DIR}/${WAYLAND_DISPLAY}
    [[ -S $wayland_socket ]] || die "Wayland socket does not exist: $wayland_socket"

    display_env+=(
        "QT_QPA_PLATFORM=wayland"
        "WAYLAND_DISPLAY=$WAYLAND_DISPLAY"
        "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
    )
    podman_display_args+=(--volume "$wayland_socket:$wayland_socket")
    singularity_display_args+=(--bind "$wayland_socket:$wayland_socket")
else
    display_env+=("QT_QPA_PLATFORM=xcb" "QT_X11_NO_MITSHM=1")
    if [[ $host_os == Darwin ]]; then
        # Podman runs inside a VM on macOS, so the XQuartz server is reached
        # through Podman's host gateway rather than a Unix-domain socket.
        display_env+=("DISPLAY=host.containers.internal:0" "LIBGL_ALWAYS_SOFTWARE=1")
    else
        [[ -n ${DISPLAY:-} ]] || die "DISPLAY is not set and no usable Wayland socket was found"
        [[ -d /tmp/.X11-unix ]] || die "X11 socket directory does not exist: /tmp/.X11-unix"
        xauthority=${XAUTHORITY:-${HOME}/.Xauthority}
        [[ -r $xauthority ]] || die "Xauthority file is not readable: $xauthority"

        display_env+=("DISPLAY=$DISPLAY" "XAUTHORITY=/tmp/nih2mne.xauth")
        podman_display_args+=(
            --volume /tmp/.X11-unix:/tmp/.X11-unix:ro
            --volume "$xauthority:/tmp/nih2mne.xauth:ro"
        )
        singularity_display_args+=(
            --bind /tmp/.X11-unix:/tmp/.X11-unix:ro
            --bind "$xauthority:/tmp/nih2mne.xauth:ro"
        )
    fi
fi

if [[ $engine == podman ]]; then
    podman_bin=${PODMAN:-podman}
    command -v "$podman_bin" >/dev/null 2>&1 || die "Podman executable not found: $podman_bin"

    podman_args=(run --rm --security-opt label=disable --workdir /data --volume "$data_dir:/data:rw")
    if [[ -t 0 && -t 1 ]]; then
        podman_args+=(--interactive --tty)
    fi
    if [[ $host_os == Linux && -d /dev/dri ]]; then
        podman_args+=(--device /dev/dri)
    fi
    for assignment in "${display_env[@]}"; do
        podman_args+=(--env "$assignment")
    done
    podman_args+=("${podman_display_args[@]}" "$image" /bin/sh -lc "$container_command")
    exec "$podman_bin" "${podman_args[@]}"
fi

if [[ $engine == singularity ]]; then
    if [[ -n ${SINGULARITY:-} ]]; then
        singularity_bin=$SINGULARITY
    elif command -v apptainer >/dev/null 2>&1; then
        singularity_bin=apptainer
    else
        singularity_bin=singularity
    fi
    command -v "$singularity_bin" >/dev/null 2>&1 || die "Apptainer/Singularity executable not found: $singularity_bin"

    singularity_args=(exec --cleanenv --contain --pwd /data --bind "$data_dir:/data:rw")
    if [[ -d /dev/dri ]]; then
        singularity_args+=(--bind /dev/dri:/dev/dri)
    fi
    for assignment in "${display_env[@]}"; do
        singularity_args+=(--env "$assignment")
    done
    singularity_args+=("${singularity_display_args[@]}" "$image" /bin/sh -lc "$container_command")
    exec "$singularity_bin" "${singularity_args[@]}"
fi

die "unknown container engine: $engine"
