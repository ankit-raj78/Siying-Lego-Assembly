#!/usr/bin/env bash
# Download the Act-FIGNet real-world episode pack (Zenodo 10.5281/zenodo.22802802, CC-BY-4.0).
set -euo pipefail
mkdir -p data
curl -L -o data/real_world_datasets_pack.zip \
  "https://zenodo.org/api/records/22802802/files/real_world_datasets_pack.zip/content"
unzip -q -o data/real_world_datasets_pack.zip -d data
echo "Data ready in data/real_world_datasets_pack"
