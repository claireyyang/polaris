#!/bin/bash
# Example script to run interactive instruction updates with visualization

# Make sure your policy server is running first!
# Example: python serve_policy.py --port 8000

uv run scripts/eval.py \
    --environment DROID-OrganizeTools \
    --policy.client DroidJointPos \
    --policy.host localhost \
    --policy.port 8000 \
    --policy.open_loop_horizon 8 \
    --run_folder ./runs/interactive_test \
    --interactive True \
    --headless False \
    --rollouts 3

# During execution:
# 1. Watch the simulation in the Isaac Sim window
# 2. Type a new instruction in the terminal and press Enter
# 3. Watch the robot immediately switch behavior!
