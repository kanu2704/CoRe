#!/usr/bin/env bash
# Make MuJoCo render on the NVIDIA GPU when the container has no NVIDIA EGL library
# (check_setup.py then shows 'llvmpipe' and ~100 ms per frame).
# Downloads NVIDIA's driver package for the driver that is already running and copies ONLY its
# user-space EGL/OpenGL libraries; no kernel module, CUDA is untouched. Needs Internet and root.
#   bash kaggle/fix_gpu_render.sh
set -euo pipefail
VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
[ -n "$VER" ] || { echo "no NVIDIA GPU / driver visible in this session"; exit 1; }
LIBDIR=/usr/lib/x86_64-linux-gnu

if [ ! -e "$LIBDIR/libEGL_nvidia.so.$VER" ]; then
  RUN="NVIDIA-Linux-x86_64-$VER.run"
  cd /tmp
  if [ ! -s "$RUN" ]; then
    echo "downloading NVIDIA $VER driver package to extract its graphics libraries (~300 MB)..."
    curl -fsSL -o "$RUN" "https://us.download.nvidia.com/tesla/$VER/$RUN" \
      || curl -fsSL -o "$RUN" "https://us.download.nvidia.com/XFree86/Linux-x86_64/$VER/$RUN" \
      || { echo "could not download driver $VER from NVIDIA"; rm -f "$RUN"; exit 1; }
  fi
  rm -rf "/tmp/nvidia-$VER"
  sh "$RUN" --extract-only --target "/tmp/nvidia-$VER" >/dev/null
  n=0
  for lib in libEGL_nvidia libnvidia-eglcore libnvidia-glsi libnvidia-glcore libnvidia-tls libnvidia-gpucomp; do
    f="/tmp/nvidia-$VER/$lib.so.$VER"
    if [ -f "$f" ]; then cp "$f" "$LIBDIR/"; n=$((n + 1)); fi
  done
  echo "copied $n NVIDIA graphics libraries"
  ln -sf "$LIBDIR/libEGL_nvidia.so.$VER" "$LIBDIR/libEGL_nvidia.so.0"
  ldconfig
fi

mkdir -p /usr/share/glvnd/egl_vendor.d
printf '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}\n' \
  > /usr/share/glvnd/egl_vendor.d/10_nvidia.json
echo "NVIDIA EGL set up for driver $VER. Check again:  \$PY scripts/check_setup.py --render"
