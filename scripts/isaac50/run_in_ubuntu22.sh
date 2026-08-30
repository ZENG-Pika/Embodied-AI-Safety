#!/usr/bin/env bash
set -euo pipefail

ROOTFS="${ISAAC_SIM_50_ROOTFS:-/home/zxw/isaacsim-5.0-rootfs}"
ISAAC_ROOT="${ISAAC_SIM_50_ROOT:-/home/zxw/isaacsim-5.0}"
DRIVER_ROOT="/opt/nvidia-driver"

if [[ ! -x "$(command -v bwrap)" ]]; then
    echo "bubblewrap (bwrap) is required" >&2
    exit 1
fi
if [[ ! -d "$ROOTFS" ]]; then
    echo "Isaac Sim 5.0 Ubuntu 22.04 rootfs not found: $ROOTFS" >&2
    exit 1
fi
if [[ $# -eq 0 ]]; then
    echo "usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi

uid="$(id -u)"
gid="$(id -g)"
args=(
    --unshare-user
    --uid "$uid"
    --gid "$gid"
    --bind "$ROOTFS" /
    --dev-bind /dev /dev
    --proc /proc
    --ro-bind /sys /sys
    --bind /home /home
    --bind /tmp /tmp
    --ro-bind /etc/resolv.conf /etc/resolv.conf
    --ro-bind /etc/passwd /etc/passwd
    --ro-bind /etc/group /etc/group
    --dir /opt
    --dir "$DRIVER_ROOT"
    --chdir "$PWD"
    --setenv HOME "$HOME"
    --setenv USER "${USER:-user}"
    --setenv LOGNAME "${LOGNAME:-user}"
    --setenv LC_ALL C.UTF-8
    --setenv LANG C.UTF-8
    --setenv PYTHONNOUSERSITE 1
    --setenv ISAAC_PYTHON "$ISAAC_ROOT/python.sh"
    --setenv INTERNDATA_ISAAC5_COMPAT 1
    --setenv OMNI_KIT_ACCEPT_EULA YES
    --setenv LD_LIBRARY_PATH "$DRIVER_ROOT"
    --setenv VK_ICD_FILENAMES "$DRIVER_ROOT/nvidia_icd.json"
    --setenv __EGL_VENDOR_LIBRARY_FILENAMES "$DRIVER_ROOT/10_nvidia.json"
    --setenv __GLX_VENDOR_LIBRARY_NAME nvidia
    --setenv __NV_PRIME_RENDER_OFFLOAD 1
)

if [[ -f /etc/machine-id ]]; then
    args+=(--ro-bind /etc/machine-id /etc/machine-id)
fi
if [[ -d /run/udev ]]; then
    args+=(--ro-bind /run/udev /run/udev)
fi
if [[ -n "${XDG_RUNTIME_DIR:-}" && -d "$XDG_RUNTIME_DIR" ]]; then
    args+=(--bind "$XDG_RUNTIME_DIR" "$XDG_RUNTIME_DIR" --setenv XDG_RUNTIME_DIR "$XDG_RUNTIME_DIR")
fi
if [[ -n "${DISPLAY:-}" ]]; then
    args+=(--setenv DISPLAY "$DISPLAY")
fi
if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    args+=(--setenv WAYLAND_DISPLAY "$WAYLAND_DISPLAY")
fi
if [[ -n "${XAUTHORITY:-}" && -f "$XAUTHORITY" ]]; then
    args+=(--setenv XAUTHORITY "$XAUTHORITY")
fi
if [[ -d /usr/local/cuda-11.8 ]]; then
    args+=(--ro-bind /usr/local/cuda-11.8 /usr/local/cuda-11.8)
fi
if [[ -x /usr/bin/nvidia-smi ]]; then
    args+=(--ro-bind /usr/bin/nvidia-smi /usr/bin/nvidia-smi)
fi

declare -A bound_driver_files=()
for file in \
    /usr/lib/x86_64-linux-gnu/libcuda.so* \
    /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
    /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* \
    /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so* \
    /usr/lib/x86_64-linux-gnu/libGLESv1_CM_nvidia.so* \
    /usr/lib/x86_64-linux-gnu/libGLESv2_nvidia.so* \
    /usr/lib/x86_64-linux-gnu/libnvcuvid.so* \
    /usr/lib/x86_64-linux-gnu/libnvoptix.so*; do
    [[ -e "$file" ]] || continue
    name="$(basename "$file")"
    [[ -n "${bound_driver_files[$name]:-}" ]] && continue
    bound_driver_files[$name]=1
    args+=(--ro-bind "$file" "$DRIVER_ROOT/$name")
done

for spec in \
    "/usr/share/vulkan/icd.d/nvidia_icd.json:nvidia_icd.json" \
    "/usr/share/vulkan/implicit_layer.d/nvidia_layers.json:nvidia_layers.json" \
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json:10_nvidia.json" \
    "/usr/share/nvidia/nvoptix.bin:nvoptix.bin"; do
    source_path="${spec%%:*}"
    target_name="${spec##*:}"
    if [[ -f "$source_path" ]]; then
        args+=(--ro-bind "$source_path" "$DRIVER_ROOT/$target_name")
    fi
done

exec bwrap "${args[@]}" "$@"
