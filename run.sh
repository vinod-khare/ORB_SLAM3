#!/bin/bash

# ORB-SLAM3 TUM-VI Monocular Example - Corridor1 Dataset
# This script runs mono_tum_vi with Boost Program Options argument parsing

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ==============================================================================
# Option 1: Using positional arguments (original style)
# ==============================================================================
echo "Running mono_tum_vi on corridor1 dataset (positional arguments)..."
./Examples/Monocular/mono_tum_vi \
  Vocabulary/ORBvoc.txt \
  Examples/Monocular/TUM-VI.yaml \
  .data/tumvi/dataset-corridor1_512_16/mav0/cam0/data \
  Examples/Monocular/TUM_TimeStamps/dataset-corridor1_512.txt \
  --output corridor1_trajectory

echo "Done! Trajectory saved to:"
echo "  - f_corridor1_trajectory.txt (camera trajectory)"
echo "  - kf_corridor1_trajectory.txt (keyframe trajectory)"

# ==============================================================================
# Option 2: Using named arguments
# ==============================================================================
# Uncomment to use named arguments instead:
# echo "Running mono_tum_vi on corridor1 dataset (named arguments)..."
# ./Examples/Monocular/mono_tum_vi \
#   --vocab Vocabulary/ORBvoc.txt \
#   --settings Examples/Monocular/TUM-VI.yaml \
#   --sequence .data/tumvi/dataset-corridor1_512_16/mav0/cam0/data \
#   --sequence Examples/Monocular/TUM_TimeStamps/dataset-corridor1_512.txt \
#   --output corridor1_trajectory

# ==============================================================================
# Option 3: Running on multiple sequences (corridor1 + corridor2)
# ==============================================================================
# Uncomment to run on both corridor sequences:
# echo "Running mono_tum_vi on corridor1 + corridor2 datasets..."
# ./Examples/Monocular/mono_tum_vi \
#   Vocabulary/ORBvoc.txt \
#   Examples/Monocular/TUM-VI.yaml \
#   .data/tumvi/dataset-corridor1_512_16/mav0/cam0/data \
#   Examples/Monocular/TUM_TimeStamps/dataset-corridor1_512.txt \
#   .data/tumvi/dataset-corridor2_512_16/mav0/cam0/data \
#   Examples/Monocular/TUM_TimeStamps/dataset-corridor2_512.txt \
#   --output corridor12_trajectory
