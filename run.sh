#!/bin/bash

# ORB-SLAM3 TUM-VI Monocular Example - Corridor1 Dataset
# This script runs orbslam3 with Boost Program Options argument parsing

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

BUILD_TYPE=Release
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-type) BUILD_TYPE="$2"; shift 2 ;;
    *) shift ;;
  esac
done
BIN=".build/${BUILD_TYPE}/orbslam3_app/orbslam3_app"

echo "Running orbslam3 on corridor1 dataset (positional arguments)..."
$BIN \
  Vocabulary/ORBvoc.txt \
  config/TUM-VI.yaml \
  .data/tumvi/dataset-corridor1_512_16/mav0/cam0/data \
  --slam-type mono \
  --output-folder .data/tumvi/dataset-corridor1_512_16/orbslam3 \
  --frames-take 0 \
  --output trajectory \
  --save-ply

