#!/bin/bash

# ORB-SLAM3 TUM-VI Monocular Example - Corridor1 Dataset
# This script runs orbslam3 with Boost Program Options argument parsing

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ==============================================================================
# Option 1: Using positional arguments (original style)
# ==============================================================================
# echo "Running orbslam3 on corridor1 dataset (positional arguments)..."
# .build/Release/orbslam3 \
#   Vocabulary/ORBvoc.txt \
#   Examples/Monocular/TUM-VI.yaml \
#   .data/tumvi/dataset-corridor1_512_16/mav0/cam0/data \
#   Examples/Monocular/TUM_TimeStamps/dataset-corridor1_512.txt \
#   --output-folder .data/tumvi/dataset-corridor1_512_16/orbslam3 \
#   --frames-take 0 \
#   --output trajectory \
#   --save-ply

# .build/Release/orbslam3 \
#   Vocabulary/ORBvoc.txt \
#   Examples/Monocular/reaper.yaml \
#   .data/reaper/testoutdoor5/testoutdoor5 \
#   .data/reaper/testoutdoor5/utc_timestamps.txt \
#   --timestamps-type utc \
#   --output-folder .data/reaper/testoutdoor5/orbslam3 \
#   --frames-skip 0 \
#   --frames-take 0 \
#   --output trajectory \
#   --save-ply

# .build/Release/orbslam3 \
#   Vocabulary/ORBvoc.txt \
#   Examples/Monocular/hilti.yaml \
#   .data/hilti/2022/exp21_outside_building/alphasense/cam0/image_raw \
#   --output-folder .data/hilti/2022/exp21_outside_building/orbslam3 \
#   --frames-skip 0 \
#   --frames-take 0 \
#   --output trajectory \
#   --save-ply

# ==============================================================================
# Option 2: Using named arguments
# ==============================================================================
# Uncomment to use named arguments instead:
echo "Running orbslam3 on corridor1 dataset (named arguments)..."
.build/Debug/orbslam3 \
  --vocab Vocabulary/ORBvoc.txt \
  --settings Examples/Monocular/TUM-VI.yaml \
  --sequence ./.data/tumvi/dataset-corridor1_512_16/mav0/cam0/data \
  --sequence Examples/Monocular/TUM_TimeStamps/dataset-corridor1_512.txt \
  --output corridor1_trajectory

# ==============================================================================
# Option 3: Running on multiple sequences (corridor1 + corridor2)
# ==============================================================================
# Uncomment to run on both corridor sequences:
# echo "Running orbslam3 on corridor1 + corridor2 datasets..."
# .build/Release/orbslam3 \
#   Vocabulary/ORBvoc.txt \
#   Examples/Monocular/TUM-VI.yaml \
#   .data/tumvi/dataset-corridor1_512_16/mav0/cam0/data \
#   Examples/Monocular/TUM_TimeStamps/dataset-corridor1_512.txt \
#   .data/tumvi/dataset-corridor2_512_16/mav0/cam0/data \
#   Examples/Monocular/TUM_TimeStamps/dataset-corridor2_512.txt \
#   --output corridor12_trajectory
