#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sudo pacman -U --needed \
  "${ROOT}"/packages/python-rendercanvas-*.pkg.tar.zst \
  "${ROOT}"/packages/python-pylinalg-*.pkg.tar.zst \
  "${ROOT}"/packages/python-wgpu-*.pkg.tar.zst \
  "${ROOT}"/packages/python-pygfx-*.pkg.tar.zst
