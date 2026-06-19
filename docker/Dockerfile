# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG FROM_SCRIPTS_IMAGE=gitlab-master.nvidia.com:5005/dl/devops/build-scripts:latest
FROM ${FROM_SCRIPTS_IMAGE} AS build-scripts

FROM nvcr.io/nvidia/tensorrt:26.03-py3

RUN apt-get update && \
    apt-get install -y pkg-config && \
    apt-get install -y git && \
    apt-get install -y zlib1g && \
    apt-get install -y zlib1g-dev && \
    apt --only-upgrade -y install openssl && \
    apt --only-upgrade -y install libpmix2

# Codec-royalty mitigation (TAO-2183 / FF-4): the system `ffmpeg` apt package (Ubuntu GPL
# build) bundles libx264/libx265 (royalty-bearing H.264/H.265 encoders) and is intentionally
# NOT installed. A codec-disabled, LGPL-only ffmpeg is built from source below. Keep
# libsm6/libxext6 — OpenGL/X11 runtime deps for OpenCV, not codecs.
RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install libsm6 libxext6 -y

# ---------------------------------------------------------------------------
# Codec-disabled, LGPL-only FFmpeg (TAO-2183 / FF-4) — identical allow-list to the
# tao-dataservices / tao-pytorch images. NO software H.264/H.265/AAC codec ships: H.264
# *input* decode is hardware-only via h264_cuvid (NVDEC); royalty-free VP9 (libvpx-vp9) +
# mjpeg encode. libavformat/libavutil retained per Legal guidance (2026-06-12).
# NVIDIA_DRIVER_CAPABILITIES must include `video` for the h264_cuvid runtime path.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
ARG FFMPEG_VERSION=8.1
ARG FFMPEG_SHA256=b072aed6871998cce9b36e7774033105ca29e33632be5b6347f3206898e0756a
ARG NV_CODEC_HEADERS_VERSION=n12.1.14.0
# Mirror build ARGs into ENV so the quoted (<<'EOF') heredoc reads them from the shell env.
ENV FFMPEG_VERSION=${FFMPEG_VERSION} \
    FFMPEG_SHA256=${FFMPEG_SHA256} \
    NV_CODEC_HEADERS_VERSION=${NV_CODEC_HEADERS_VERSION}
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential yasm nasm pkg-config git curl ca-certificates xz-utils libvpx-dev libgnutls28-dev && \
    rm -rf /var/lib/apt/lists/*
RUN <<'EOF'
#!/usr/bin/env bash
set -euxo pipefail
mkdir -p /tmp/ffmpeg-build /usr/share/oss-sources/source-builds && cd /tmp/ffmpeg-build
git clone --depth 1 --branch "${NV_CODEC_HEADERS_VERSION}" https://github.com/FFmpeg/nv-codec-headers.git
tar -C /tmp/ffmpeg-build --exclude='.git' \
    -czf "/usr/share/oss-sources/source-builds/nv-codec-headers-${NV_CODEC_HEADERS_VERSION}.tar.gz" nv-codec-headers
make -C nv-codec-headers -j"$(nproc)" && make -C nv-codec-headers install
curl -fsSL "https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.xz" -o ffmpeg.tar.xz
echo "${FFMPEG_SHA256}  ffmpeg.tar.xz" | sha256sum -c -
cp ffmpeg.tar.xz "/usr/share/oss-sources/source-builds/ffmpeg-${FFMPEG_VERSION}.tar.xz"
tar -xf ffmpeg.tar.xz
cd "ffmpeg-${FFMPEG_VERSION}"
./configure \
  --prefix=/usr/local \
  --disable-debug --disable-doc --disable-static --enable-shared --enable-pic \
  --enable-gnutls \
  --disable-nonfree --disable-gpl \
  --disable-encoders --disable-decoders --disable-everything \
  --enable-libvpx \
  --enable-encoder=libvpx_vp9 --enable-encoder=mjpeg \
  --enable-decoder=h264_cuvid --enable-decoder=vp8 --enable-decoder=vp9 --enable-decoder=mjpeg \
  --enable-nvenc --enable-nvdec \
  --enable-protocol=file \
  --enable-muxer=webm --enable-muxer=image2 \
  --enable-demuxer=mov --enable-demuxer=matroska --enable-demuxer=image2 \
  --enable-parser=h264 --enable-parser=vp8 --enable-parser=vp9 \
  --enable-bsf=h264_mp4toannexb --enable-bsf=vp9_superframe \
  --enable-filter=fps --enable-filter=scale --enable-filter=format \
  --extra-cflags="-I/usr/local/include" --extra-ldflags="-L/usr/local/lib"
make -j"$(nproc)"
make install
ldconfig
rm -rf /tmp/ffmpeg-build
# ---- build-time codec assertions (fail the build, not the OSRB audit) ----
ffmpeg -hide_banner -version | head -n 3
! ffmpeg -buildconf | grep -E "enable-gpl|enable-nonfree"
for forbidden in libx264 libopenh264 libx265 h264 hevc aac; do
  if ffmpeg -hide_banner -encoders 2>/dev/null | awk '{print $2}' | grep -Fx "$forbidden" >/dev/null; then
    echo "ERROR: forbidden encoder '$forbidden' leaked into the image (GPL/patent-encumbered)" >&2; exit 1; fi
done
# Required royalty-free encoders (runtime name is libvpx-vp9, hyphenated).
for enc in libvpx-vp9 mjpeg; do
  ffmpeg -hide_banner -encoders 2>/dev/null | awk '{print $2}' | grep -Fx "$enc" >/dev/null \
    || { echo "ERROR: expected encoder '$enc' to be enabled" >&2; exit 1; }
done
for dec in h264_cuvid vp8 vp9 mjpeg; do
  ffmpeg -hide_banner -decoders 2>/dev/null | awk '{print $2}' | grep -Fx "$dec" >/dev/null \
    || { echo "ERROR: expected decoder '$dec' to be enabled" >&2; exit 1; }
done
# Only h264_cuvid may appear in the H.264 decoder family — no SW h264 decoder.
h264_decoders="$(ffmpeg -hide_banner -decoders 2>/dev/null | awk '$2 ~ /^h264/ {print $2}' | sort -u)"
forbidden_h264_decoders="$(printf '%s\n' "${h264_decoders}" | awk 'NF && $0 != "h264_cuvid"')"
if [[ -n "${forbidden_h264_decoders}" ]]; then
  echo "ERROR: forbidden H.264 decoder(s) leaked into the image: ${forbidden_h264_decoders}" >&2; exit 1; fi
for forbidden in hevc aac; do
  if ffmpeg -hide_banner -decoders 2>/dev/null | awk '{print $2}' | grep -Fx "$forbidden" >/dev/null; then
    echo "ERROR: forbidden decoder '$forbidden' leaked into the image" >&2; exit 1; fi
done
ffmpeg -hide_banner -muxers   2>/dev/null | awk '{print $2}' | grep -Fx webm   >/dev/null || { echo "ERROR: muxer webm missing"   >&2; exit 1; }
ffmpeg -hide_banner -muxers   2>/dev/null | awk '{print $2}' | grep -Fx image2 >/dev/null || { echo "ERROR: muxer image2 missing" >&2; exit 1; }
ffmpeg -hide_banner -demuxers 2>/dev/null | awk '{print $2}' | grep -E '(^|,)matroska(,|$)' >/dev/null || { echo "ERROR: demuxer matroska,webm missing" >&2; exit 1; }
ffmpeg -hide_banner -demuxers 2>/dev/null | awk '{print $2}' | grep -E '(^|,)mov(,|$)'      >/dev/null || { echo "ERROR: demuxer mov missing" >&2; exit 1; }
ffmpeg -hide_banner -protocols 2>/dev/null | awk '{print $1}' | grep -Fx file             >/dev/null || { echo "ERROR: protocol file missing" >&2; exit 1; }
ffmpeg -hide_banner -bsfs      2>/dev/null | awk '{print $1}' | grep -Fx h264_mp4toannexb  >/dev/null || { echo "ERROR: bsf h264_mp4toannexb missing" >&2; exit 1; }
# vp9_superframe is required to mux libvpx-vp9 output into WebM/Matroska (the FF-1 path).
ffmpeg -hide_banner -bsfs      2>/dev/null | awk '{print $1}' | grep -Fx vp9_superframe    >/dev/null || { echo "ERROR: bsf vp9_superframe missing (needed to mux VP9 into WebM)" >&2; exit 1; }
for filt in fps scale format; do
  ffmpeg -hide_banner -filters 2>/dev/null | awk '{print $2}' | grep -Fx "$filt" >/dev/null \
    || { echo "ERROR: filter '$filt' missing" >&2; exit 1; }
done
echo "FFmpeg codec allow-list verified: no libx264/libx265/openh264/SW-h264/hevc/aac; VP9+mjpeg encode, h264_cuvid+vp8/vp9/mjpeg decode."
EOF
RUN apt-get purge -y --auto-remove yasm nasm && rm -rf /var/lib/apt/lists/*
# upgrade pip
RUN pip install --upgrade pip

WORKDIR /workspace/
COPY docker/requirements-dev.txt /workspace/
RUN pip install -r requirements-dev.txt && \
    rm -rf requirements-dev.txt

# TAO-CORE microservice dependencies
COPY docker/requirements-apt.txt /workspace/
RUN apt-get update && apt-get install --no-install-suggests --no-install-recommends --yes $(cat /workspace/requirements-apt.txt)
RUN rm -rf /workspace/requirements-apt.txt

# ---------------------------------------------------------------------------
# OpenCV rebuilt WITH_FFMPEG=OFF (TAO-2183 / FF-3)
# ---------------------------------------------------------------------------
# Overrides the opencv-python 4.13.0.92 wheel pulled by requirements-dev.txt, whose bundled
# libav* (inside opencv_python.libs/) carries the royalty-bearing H.264/H.265/AAC codecs.
# Uninstall every opencv flavor, scrub the leftover cv2/ + .libs/, then build from sdist with
# codec backends disabled (headless: no GUI in containers). The embedded python check fails
# the build if cv2 reports FFMPEG: YES or links any libav*.
#
# VERSION NOTE: base wheel 4.13.0.92 is wheel-only (no public sdist), so we build the latest
# source-available release 4.12.0.88, mirroring paidf-auto-labeling. Immaterial API delta for
# tao-deploy's cv2 usage; revisit if DL-frameworks ships a libav-free 4.13 wheel (FF-3).
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake libjpeg-dev libpng-dev libtiff-dev libwebp-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*
RUN <<'EOF'
#!/usr/bin/env bash
set -euxo pipefail
# Neutralize any base global PIP_CONSTRAINT; every install below is exact-pinned.
export PIP_CONSTRAINT=""
python -m pip uninstall -y \
    opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true
SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
rm -rf "${SP}"/cv2 \
       "${SP}"/opencv_python*.libs \
       "${SP}"/opencv_python*.dist-info \
       "${SP}"/opencv_contrib_python*.dist-info \
       "${SP}"/opencv_contrib_python*.libs
# --ignore-installed: the tensorrt base ships some build tools (e.g. wheel) via apt/debian
# with no RECORD file, which pip cannot uninstall when upgrading; install over them instead.
python -m pip install --no-cache-dir --ignore-installed \
    "setuptools>=78.1.1" "wheel>=0.46.2" "scikit-build>=0.18,<1" "cmake>=3.27,<4"
export ENABLE_CONTRIB=0
export ENABLE_HEADLESS=1
export CMAKE_ARGS="\
    -DWITH_FFMPEG=OFF \
    -DWITH_GSTREAMER=OFF \
    -DWITH_V4L=OFF \
    -DWITH_LIBV4L=OFF \
    -DWITH_QT=OFF \
    -DWITH_GTK=OFF \
    -DOPENCV_ENABLE_NONFREE=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF"
# Build opencv-python from sdist. 4.12.0.88 is the latest version with a public sdist
# (4.13.0.92 is wheel-only). ENABLE_HEADLESS=1 + GUI disabled => no X/Qt deps.
python -m pip install --no-cache-dir --no-build-isolation \
    --no-binary=opencv-python \
    "opencv-python==4.12.0.88"
python - <<'PY'
import re, subprocess
from pathlib import Path
import cv2
info = cv2.getBuildInformation()
if re.search(r"FFMPEG:\s+YES", info):
    raise SystemExit("ERROR: OpenCV unexpectedly built WITH FFmpeg")
cv2_pkg_dir = Path(cv2.__file__).resolve().parent
cv2_sos = sorted(cv2_pkg_dir.rglob("cv2*.so"))
if not cv2_sos:
    raise SystemExit(f"ERROR: could not locate cv2 compiled extension under {cv2_pkg_dir}")
forbidden = ("libavcodec", "libavdevice", "libavfilter",
             "libavformat", "libavutil", "libswresample", "libswscale")
for cv2_so in cv2_sos:
    ldd = subprocess.check_output(["ldd", str(cv2_so)], text=True)
    for lib in forbidden:
        if lib in ldd:
            raise SystemExit(f"ERROR: {cv2_so.name} still links {lib}")
print("opencv", cv2.__version__, "FFmpeg disabled, no libav* linkage in", [p.name for p in cv2_sos])
PY
leftovers="$(find "${SP}" \( \
    -path '*opencv_python*.libs/libavcodec*' -o \
    -path '*opencv_python*.libs/libavdevice*' -o \
    -path '*opencv_python*.libs/libavfilter*' -o \
    -path '*opencv_python*.libs/libavformat*' -o \
    -path '*opencv_python*.libs/libavutil*' -o \
    -path '*opencv_python*.libs/libswresample*' -o \
    -path '*opencv_python*.libs/libswscale*' \
\) -print 2>/dev/null || true)"
if [ -n "${leftovers}" ]; then echo "ERROR: bundled libav* leaked back in:"; echo "${leftovers}"; exit 1; fi
EOF
# NOTE: build-essential/cmake intentionally NOT purged (tensorrt:26.03 may need a runtime
# toolchain for TRT/torch JIT). Only build-only codec assemblers (yasm/nasm) were purged above.

# Create user that will run commands
ARG user_id=1000
ARG user_name=developer
ARG groups=developer:1000
ARG home=/home/developer
RUN echo "ALL   ALL = (ALL) NOPASSWD: ALL" > /etc/sudoers \
    && mkdir -p "$(dirname $home)"
WORKDIR /workspace/