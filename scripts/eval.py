import tyro
import mediapy

# import wandb
import tqdm
import gymnasium as gym
import torch
import argparse
import pandas as pd
import threading
import queue
import sys
import select


from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import EvalArgs


def instruction_listener(instruction_queue: queue.Queue, stop_event: threading.Event):
    """
    Background thread that listens for user input to update instructions.
    Simply type a new instruction and press Enter - it will be applied on the next step.
    """
    print("\n" + "="*60)
    print("INTERACTIVE MODE ENABLED")
    print("Type a new instruction and press Enter to update it during execution")
    print("The instruction will be applied immediately on the next policy query")
    print("TIP: Use --headless False to see the simulation visually")
    print("="*60 + "\n")
    
    while not stop_event.is_set():
        try:
            # Check if input is available (non-blocking on Unix)
            if sys.platform != 'win32':
                # Unix-like systems: use select
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    line = sys.stdin.readline().strip()
                    if line:
                        instruction_queue.put(line)
                        print(f"\n✓ Instruction updated to: '{line}'")
                        print(">>> Will apply on next policy query\n")
            else:
                # Windows: fall back to blocking input with timeout
                # This is less ideal but works
                import msvcrt
                if msvcrt.kbhit():
                    line = input().strip()
                    if line:
                        instruction_queue.put(line)
                        print(f"\n✓ Instruction updated to: '{line}'")
                        print(">>> Will apply on next policy query\n")
                else:
                    threading.Event().wait(0.1)
        except (EOFError, KeyboardInterrupt):
            break
        except Exception:
            # Fallback: just use blocking input
            try:
                line = input().strip()
                if line:
                    instruction_queue.put(line)
                    print(f"\n✓ Instruction updated to: '{line}'")
                    print(">>> Will apply on next policy query\n")
            except (EOFError, KeyboardInterrupt):
                break


def main(eval_args: EvalArgs):
    # This must be done before importing anything from IsaacLab
    # Inside main function to avoid launching IsaacLab in global scope
    # >>>> Isaac Sim App Launcher <<<<
    parser = argparse.ArgumentParser()
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = eval_args.headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    # >>>> Isaac Sim App Launcher <<<<

    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from polaris.environments.manager_based_rl_splat_environment import (
        ManagerBasedRLSplatEnv,
    )
    from polaris.utils import load_eval_initial_conditions
    from polaris.policy import InferenceClient
    # from real2simeval.autoscoring import TASK_TO_SUCCESS_CHECKER

    env_cfg = parse_env_cfg(
        eval_args.environment,
        device="cuda",
        num_envs=1,
        use_fabric=True,
    )
    env: ManagerBasedRLSplatEnv = gym.make(eval_args.environment, cfg=env_cfg)  # type: ignore

    language_instruction, initial_conditions = load_eval_initial_conditions(
        usd=env.usd_file,
        initial_conditions_file=eval_args.initial_conditions_file,
        rollouts=eval_args.rollouts,
    )
    rollouts = len(initial_conditions)
    # Resume CSV logging
    run_folder = Path(eval_args.run_folder)
    run_folder.mkdir(parents=True, exist_ok=True)
    csv_path = run_folder / "eval_results.csv"
    if csv_path.exists():
        episode_df = pd.read_csv(csv_path)
    else:
        episode_df = pd.DataFrame(
            {
                "episode": pd.Series(dtype="int"),
                "episode_length": pd.Series(dtype="int"),
                "success": pd.Series(dtype="bool"),
                "progress": pd.Series(dtype="float"),
            }
        )
    episode = len(episode_df)
    if episode >= rollouts:
        print("All rollouts have been evaluated. Exiting.")
        env.close()
        simulation_app.close()
        return

    policy_client: InferenceClient = InferenceClient.get_client(eval_args.policy)

    # Setup interactive instruction updates (only if enabled)
    instruction_queue = queue.Queue()
    stop_event = threading.Event()
    if eval_args.interactive:
        listener_thread = threading.Thread(
            target=instruction_listener, 
            args=(instruction_queue, stop_event),
            daemon=True
        )
        listener_thread.start()

    video = []
    horizon = env.max_episode_length
    bar = tqdm.tqdm(range(horizon))
    obs, info = env.reset(
        object_positions=initial_conditions[episode % len(initial_conditions)]
    )
    policy_client.reset()
    current_instruction = language_instruction
    print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")
    print(f" >>> Initial instruction: '{current_instruction}' <<< ")
    
    while True:
        # Check for instruction updates
        try:
            new_instruction = instruction_queue.get_nowait()
            current_instruction = new_instruction
            policy_client.discard_action_chunk()
            print("\n>>> Action chunk discarded, requerying policy with new instruction <<<\n")
        except queue.Empty:
            pass
        
        action, viz = policy_client.infer(obs, current_instruction)
        if viz is not None:
            video.append(viz)
        obs, rew, term, trunc, info = env.step(
            torch.tensor(action).reshape(1, -1), expensive=policy_client.rerender
        )

        bar.update(1)
        if term[0] or trunc[0] or bar.n >= horizon:
            policy_client.reset()

            # Save video and metadata
            filename = run_folder / f"episode_{episode}.mp4"
            mediapy.write_video(filename, video, fps=5)

            # Log episode results to CSV
            episode_data = {
                "episode": episode,
                "episode_length": bar.n,
                "success": info["rubric"]["success"],
                "progress": info["rubric"]["progress"],
            }
            episode_df = pd.concat(
                [episode_df, pd.DataFrame([episode_data])], ignore_index=True
            )
            episode_df.to_csv(csv_path, index=False)

            bar.close()
            print(f"Episode {episode} finished. Episode length: {bar.n}")
            bar = tqdm.tqdm(range(horizon))
            obs, info = env.reset(
                object_positions=initial_conditions[episode % len(initial_conditions)]
            )
            
            # Reset instruction to original for new episode
            current_instruction = language_instruction
            print(f" >>> Starting episode {episode + 1} with instruction: '{current_instruction}' <<< ")

            episode += 1
            video = []
            if episode >= rollouts:
                break

    stop_event.set()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    main(args)
