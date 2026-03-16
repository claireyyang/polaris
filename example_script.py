import torch
import argparse
import gymnasium as gym
import threading
import queue
from isaaclab.app import AppLauncher
# This must be done before importing anything with dependency on Isaaclab
# >>>> Isaac Sim App Launcher <<<<
parser = argparse.ArgumentParser()
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
# >>>> Isaac Sim App Launcher <<<<

import polaris.environments  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from polaris.environments.manager_based_rl_splat_environment import ManagerBasedRLSplatEnv  # noqa: E402
from polaris.utils import load_eval_initial_conditions  # noqa: E402


def instruction_listener(instruction_queue: queue.Queue, stop_event: threading.Event):
    """
    Background thread that listens for user input to update instructions.
    Press 'p' to pause and provide a new instruction.
    """
    print("\n" + "="*60)
    print("INTERACTIVE MODE: Press 'p' + Enter to pause and update instruction")
    print("="*60 + "\n")
    
    while not stop_event.is_set():
        try:
            user_input = input()
            if user_input.lower() == 'p':
                print("\n>>> PAUSED: Enter new instruction (or press Enter to resume): ")
                new_instruction = input("New instruction: ")
                if new_instruction.strip():
                    instruction_queue.put(new_instruction)
                    print(f">>> Instruction updated to: '{new_instruction}'")
                else:
                    print(">>> No instruction provided, resuming with current instruction")
        except EOFError:
            break


env_cfg = parse_env_cfg(
    "DROID-OrganizeTools",
    device="cuda",
    num_envs=1,
    use_fabric=True,
)
# FoodBussing
env: ManagerBasedRLSplatEnv = gym.make("DROID-OrganizeTools", cfg=env_cfg)   # type: ignore
language_instruction, initial_conditions = load_eval_initial_conditions(env.usd_file)
obs, info = env.reset(object_positions = initial_conditions[0])

# Setup interactive instruction updates
instruction_queue = queue.Queue()
stop_event = threading.Event()
listener_thread = threading.Thread(
    target=instruction_listener, 
    args=(instruction_queue, stop_event),
    daemon=True
)
listener_thread.start()

current_instruction = language_instruction
print(f">>> Initial instruction: '{current_instruction}' <<<")

while True:
    # Check for instruction updates (example uses random actions, but instruction is tracked)
    try:
        new_instruction = instruction_queue.get_nowait()
        current_instruction = new_instruction
        print("\n>>> Instruction updated (note: this example uses random actions) <<<\n")
    except queue.Empty:
        pass
    
    action = torch.tensor(env.action_space.sample())
    obs, rew, term, trunc, info = env.step(action, expensive=True)

    if term[0] or trunc[0]:
        print(f"Episode Finished. Success: {info['rubric']['success']}, Progress: {info['rubric']['progress']}")
        obs, info = env.reset(object_positions=initial_conditions[0])
        current_instruction = language_instruction
        print(f">>> Reset to initial instruction: {current_instruction} <<<")
