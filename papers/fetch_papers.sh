#!/usr/bin/env bash
# Fetch the public reference PDFs into papers/ and verify checksums.
# The two 2016–2017 sources are local copies (see README.md); this script only
# checks that they are present and unchanged. Add public papers as
# "name|url|sha256" lines when the literature pass (E5.2) reads them.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
ua="Mozilla/5.0 (fetch_papers.sh; research use)"

public=(
)

local_only=(
  "martin-dissertation-2016-rbf-fd-interfaces.pdf|a658c8b94547eb085eed68e1bd70cca00ed9c49eba8911d8412576d9945e96f2"
  "martin-fornberg-2017-rbf-fd-heat-equilibrium-eabe-submitted.pdf|e1ef8560939a6504f1a9df0311b473ab5dc07bce0f406b63607162ad16fbe78b"
)

sha() { shasum -a 256 "$1" | cut -d' ' -f1; }

for entry in ${public[@]+"${public[@]}"}; do
  IFS='|' read -r name url want <<<"$entry"
  dest="$here/$name"
  if [[ ! -f "$dest" ]]; then
    echo "fetching $name"
    curl -fsSL -A "$ua" -o "$dest" "$url"
  fi
  got="$(sha "$dest")"
  if [[ "$got" != "$want" ]]; then
    echo "WARNING $name: sha256 $got != recorded $want (upstream revised?)" >&2
  else
    echo "ok       $name"
  fi
done

for entry in "${local_only[@]}"; do
  IFS='|' read -r name want <<<"$entry"
  dest="$here/$name"
  if [[ ! -f "$dest" ]]; then
    echo "MISSING  $name (local copy; see papers/README.md)" >&2
    continue
  fi
  got="$(sha "$dest")"
  if [[ "$got" != "$want" ]]; then
    echo "WARNING $name: sha256 differs from the copy indexed 2026-09-20" >&2
  else
    echo "ok       $name"
  fi
done
