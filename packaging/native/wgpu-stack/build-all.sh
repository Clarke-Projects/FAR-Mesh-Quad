#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sudo pacman -S --needed \
  python-pip \
  python-installer \
  python-cffi \
  python-pycparser \
  python-jinja \
  python-markupsafe \
  python-numpy \
  python-freetype-py \
  python-hsluv \
  python-uharfbuzz

order=(
  python-rendercanvas
  python-pylinalg
  python-wgpu
  python-pygfx
)

for name in "${order[@]}"; do
  echo
  echo "== Building ${name} =="
  cd "${ROOT}/${name}"
  rm -rf pkg src *.pkg.tar.zst
  makepkg -Cf
done

mkdir -p "${ROOT}/packages"
cp "${ROOT}"/*/*.pkg.tar.zst "${ROOT}/packages/"

echo
echo "Built packages:"
ls -lh "${ROOT}/packages"
