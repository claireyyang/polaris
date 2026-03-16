# Interactive Instruction Updates

This feature allows you to pause execution during policy evaluation and provide new instructions that are paired with the current observation.

## Overview

The interactive instruction update feature enables real-time control over the robot's behavior by allowing you to:
- Pause execution at any step during an episode
- Provide a new language instruction
- Have the policy immediately requery with the new instruction and current observation
- Discard any remaining actions in the current action chunk

## How It Works

### Architecture

1. **Background Listener Thread**: A separate thread monitors for user input without blocking the main execution loop
2. **Instruction Queue**: Thread-safe queue for passing new instructions from the listener to the main loop
3. **Action Chunk Discard**: When a new instruction is provided, any remaining actions in the current chunk are discarded
4. **Immediate Requery**: The policy is immediately queried with the new instruction and current observation

### Key Design Decisions

- **Policy state is NOT reset** when instructions change mid-episode (only reset at episode boundaries)
- **Action chunks are discarded** immediately when a new instruction is provided
- **Instructions reset** to the original at the start of each new episode

## Usage

### In eval.py

**IMPORTANT**: To enable interactive mode, use `--interactive True`. To see the simulation visually (highly recommended), also use `--headless False`:

```bash
python scripts/eval.py \
    --environment DROID-OrganizeTools \
    --policy.client DroidJointPos \
    --interactive True \
    --headless False \
    ...
```

Without `--interactive True`, the script runs in normal (non-interactive) mode.
Without `--headless False`, the simulation runs in the background and you won't see what's happening!

During execution:
1. Simply type your new instruction in the terminal
2. Press Enter
3. The policy will immediately discard remaining actions and requery with the new instruction on the next step

Example:
```
INTERACTIVE MODE ENABLED
Type a new instruction and press Enter to update it during execution
The instruction will be applied immediately on the next policy query
================================================================

>>> Initial instruction: 'pick up the cup'

[execution running...]

place the cup on the table  # User types new instruction + Enter

✓ Instruction updated to: 'place the cup on the table'
>>> Will apply on next policy query

>>> Action chunk discarded, requerying policy with new instruction <<<
```

### In example_script.py

The example script also supports interactive instructions (though it uses random actions):

```bash
python example_script.py
```

Same interaction pattern as above.

## Visualization

### Making the Simulation Visible

By default, `eval.py` runs in **headless mode** (no GUI), which means you can't see what's happening. For interactive instruction updates, you'll want to see the simulation!

**Solution**: Add `--interactive True` and `--headless False` to your command:

```bash
python scripts/eval.py \
    --environment DROID-OrganizeTools \
    --policy.client DroidJointPos \
    --policy.host localhost \
    --policy.port 8000 \
    --run_folder ./runs/interactive_test \
    --interactive True \
    --headless False
```

This will open the Isaac Sim GUI window where you can:
- See the robot and environment in 3D
- Watch the robot execute actions in real-time
- Observe the effects of your instruction changes
- Better judge when to intervene with new instructions

### Example Workflow

1. Start the script with `--interactive True --headless False`
2. Watch the robot attempt the initial instruction
3. When you see it struggling or want to change behavior, type a new instruction + Enter
4. The robot will immediately discard remaining actions and switch to the new behavior

## Implementation Details

### Files Modified

1. **`src/polaris/policy/abstract_client.py`**
   - Added `discard_action_chunk()` method to base class

2. **`src/polaris/policy/droid_jointpos_client.py`**
   - Implemented `discard_action_chunk()` to reset chunk state

3. **`scripts/eval.py`**
   - Added `instruction_listener()` function for background input monitoring
   - Modified main loop to check for instruction updates
   - Added instruction queue and threading setup

4. **`example_script.py`**
   - Added same interactive instruction support

### Code Flow

```
Main Loop:
1. Check instruction queue for updates
2. If new instruction found:
   - Update current_instruction
   - Call policy_client.discard_action_chunk()
   - Print confirmation
3. Call policy_client.infer(obs, current_instruction)
4. Execute action
5. If episode ends:
   - Reset instruction to original
   - Call policy_client.reset()
```

## Testing

Run the test suite to verify core functionality:

```bash
python test_interactive_instruction.py
```

This tests:
- Instruction queue operations
- Main loop logic
- Policy client discard functionality (if dependencies available)

## Limitations

- Input is via terminal only (no GUI)
- Requires manual typing (no voice input)
- Thread-based input may not work in all terminal environments
- The listener thread is daemon, so it will terminate when the main program exits

## Quick Reference

### Command Line Arguments

```bash
python scripts/eval.py \
    --environment DROID-OrganizeTools \           # Environment to use
    --policy.client DroidJointPos \               # Policy client type
    --policy.host localhost \                     # Policy server host
    --policy.port 8000 \                          # Policy server port
    --policy.open_loop_horizon 8 \                # Action chunk size
    --run_folder ./runs/my_run \                  # Output folder
    --interactive True \                          # ENABLE INTERACTIVE MODE (required!)
    --headless False \                            # SHOW VISUALIZATION (important!)
    --rollouts 5                                  # Number of episodes
```

### Interactive Commands

- **Type instruction + Enter**: Update to new instruction (applied on next policy query)
- The terminal accepts input at any time during execution
- No need to pause - just type and press Enter

### Troubleshooting

**Q: I can't see the simulation!**
A: Add `--headless False` to your command

**Q: The input prompt doesn't appear**
A: There's no prompt - just type your instruction directly and press Enter. Make sure you added `--interactive True` to enable interactive mode.

**Q: Instructions aren't updating**
A: Make sure you pressed Enter after typing your instruction. Also verify `--interactive True` is set.

**Q: The script seems to hang or the robot doesn't move**
A: If you're not using interactive mode, make sure you didn't accidentally enable it. Run without `--interactive` flag for normal operation.

**Q: Robot behavior doesn't change immediately**
A: This is expected if you're mid-action-chunk. The change takes effect on the next policy query.

## Future Enhancements

Potential improvements:
- GUI-based instruction input
- Voice-to-text instruction input
- Network-based remote instruction updates
- Instruction history and undo
- Pre-defined instruction shortcuts
- Visual indicator in simulation when paused
