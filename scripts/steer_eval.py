"""
Evaluate the policy with interactive steering mid-rollout.

A browser-based GUI shows the current viz frame each time a new action
chunk begins.  The sim loop **blocks** until the user either continues
or types a new steering instruction and submits it.  Because the loop is
paused while the user looks at the frame, the steering command is
guaranteed to apply to the very next inference call — no lag or race
conditions.

To use remotely, SSH-tunnel the GUI port to your local machine:

    ssh -L 7860:localhost:7860 user@remote

Then open http://localhost:7860 in your browser.
"""

import json
import io
import tyro
import mediapy

import numpy as np
import tqdm
import gymnasium as gym
import torch
import argparse
import pandas as pd
import threading
import logging

from flask import Flask, Response, request, render_template_string
from PIL import Image
from pathlib import Path
from isaaclab.app import AppLauncher

from polaris.config import SteerEvalArgs


# ── Steering GUI (browser-based) ─────────────────────────────────────

_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>Polaris – Steering</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: #1a1a2e; color: #eee;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh;
  }
  .container { text-align: center; max-width: 900px; width: 100%; padding: 24px; }
  h1 { font-size: 1.4rem; margin-bottom: 12px; color: #00ffc8; }
  #meta { color: #aaa; font-size: 0.9rem; margin-bottom: 8px; min-height: 1.2em; }
  .frame-box {
    border: 2px solid #333; border-radius: 8px; overflow: hidden;
    display: inline-block; margin-bottom: 16px; background: #000;
  }
  .frame-box img { display: block; max-width: 100%; height: auto; }
  #waiting { color: #888; padding: 80px 0; font-style: italic; display: none; }
  #controls { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; min-height: 42px; }
  input[type="text"] {
    width: 400px; padding: 10px 14px; border-radius: 6px;
    border: 1px solid #555; background: #16213e; color: #eee;
    font-size: 1rem; outline: none;
  }
  input[type="text"]:focus { border-color: #00ffc8; }
  button {
    padding: 10px 24px; border-radius: 6px; border: none;
    font-size: 1rem; cursor: pointer; font-weight: 600;
  }
  .btn-continue { background: #00ffc8; color: #1a1a2e; }
  .btn-continue:hover { background: #00e6b5; }
  .btn-pause { background: #f0a500; color: #fff; }
  .btn-pause:hover { background: #d08f00; }
  .btn-steer { background: #e94560; color: #fff; }
  .btn-steer:hover { background: #d13550; }
  .hint { color: #666; font-size: 0.8rem; margin-top: 10px; min-height: 2.5em; }
</style>
</head>
<body>
<div class="container">
  <h1>Polaris – Steering</h1>
  <div id="meta"></div>
  <div class="frame-box"><img id="viz-frame" src="" alt="viz frame" style="display:none" /></div>
  <div id="waiting">Waiting for the next visualization frame…</div>
  <div id="controls"></div>
  <div id="hint" class="hint"></div>
</div>
<script>
  let lastStep = null;
  let paused = false;

  function doAction(action, instruction) {
    const body = new URLSearchParams({ action, new_instruction: instruction || '' });
    fetch('/submit', { method: 'POST', body });
  }

  function renderControls(state) {
    const ctrl = document.getElementById('controls');
    const hint = document.getElementById('hint');
    if (!state.frame_ready) {
      ctrl.innerHTML = '';
      hint.textContent = '';
      return;
    }
    if (state.paused) {
      ctrl.innerHTML = `
        <input type="text" id="instr-input" placeholder="New instruction..." />
        <button class="btn-steer" onclick="doAction('steer', document.getElementById('instr-input').value)">Steer</button>
        <button class="btn-continue" onclick="doAction('continue', '')">Cancel &amp; Continue</button>`;
      hint.textContent = 'Enter a new instruction and click Steer, or just Continue.';
      // Only focus if we just became paused
      if (!paused) document.getElementById('instr-input').focus();
    } else {
      ctrl.innerHTML = `<button class="btn-pause" onclick="doAction('pause', '')">Pause to Correct</button>`;
      hint.innerHTML = 'The rollout will continue automatically...<br/>Press \"Pause to Correct\" to provide new input.';
    }
  }

  function poll() {
    fetch('/state')
      .then(r => r.json())
      .then(state => {
        paused = state.paused;
        const img = document.getElementById('viz-frame');
        const waiting = document.getElementById('waiting');
        const meta = document.getElementById('meta');

        if (state.frame_ready) {
          waiting.style.display = 'none';
          img.style.display = 'block';
          meta.textContent = 'Step ' + state.step + ' \u00a0|\u00a0 Instruction: ' + state.instruction;
          // Update image only when step changes to avoid flicker
          if (state.step !== lastStep) {
            img.src = '/frame?t=' + state.step;
            lastStep = state.step;
          }
        } else {
          waiting.style.display = 'block';
          img.style.display = 'none';
          meta.textContent = '';
        }

        renderControls(state);
      })
      .catch(() => {})
      .finally(() => setTimeout(poll, 500));
  }

  poll();
</script>
</body>
</html>
"""


class SteeringGUIWeb:
    """Browser-based GUI for interactive policy steering.

    Replaces the OpenCV ``SteeringGUI`` so steering works over SSH
    tunnels.  The interface is identical to the caller:

    *   ``show(frame_rgb, step)`` blocks up to 2 seconds, unless paused.
    *   Returns the new instruction string, or ``None`` if the user
        just continued.

    Internally a Flask server runs on a background thread.  The sim
    thread publishes a frame and waits; the Flask request handler
    unblocks it when the user submits the form.
    """

    def __init__(self, initial_instruction: str, port: int = 7860) -> None:
        self.instruction = initial_instruction
        self.port = port

        self._frame_jpeg: bytes | None = None
        self._step: int | None = None

        # Sim thread waits on this; Flask handler sets it.
        self._user_event = threading.Event()
        self._user_response: str | None = None
        self._user_action: str | None = None
        self._paused: bool = False

        # Signals that a new frame is ready for the browser to show.
        self._frame_ready = threading.Event()

        self._app = self._build_app()
        self._server_thread = threading.Thread(
            target=self._run_server, daemon=True
        )
        self._server_thread.start()

    def _build_app(self) -> Flask:
        app = Flask(__name__)

        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        @app.route("/")
        def index():
            return render_template_string(_HTML_TEMPLATE)

        @app.route("/frame")
        def frame():
            if self._frame_jpeg is None:
                return Response(status=204)
            return Response(self._frame_jpeg, mimetype="image/jpeg")

        @app.route("/state")
        def state():
            from flask import jsonify
            return jsonify({
                "frame_ready": self._frame_ready.is_set(),
                "step": self._step,
                "instruction": self.instruction,
                "paused": self._paused,
            })

        @app.route("/submit", methods=["POST"])
        def submit():
            action = request.form.get("action", "continue")
            new_text = request.form.get("new_instruction", "").strip()

            if action == "pause":
                self._paused = True
                self._user_action = "pause"
                self._user_event.set()
                return "", 204

            if action == "steer" and new_text:
                self.instruction = new_text
                self._user_response = new_text
                self._user_action = "steer"
            else:
                self._user_response = None
                self._user_action = "continue"

            self._frame_ready.clear()
            self._paused = False
            self._user_event.set()
            return render_template_string(_HTML_TEMPLATE, step=None, instruction=None, paused=False)

        return app

    def _run_server(self) -> None:
        print(
            f"\n  *** Steering GUI is live at http://localhost:{self.port} ***\n"
            f"  (SSH tunnel: ssh -L {self.port}:localhost:{self.port} user@remote)\n"
        )
        self._app.run(
            host="0.0.0.0", port=self.port, threaded=True, use_reloader=False,
        )

    def show(self, frame_rgb: np.ndarray, step: int) -> str | None:
        """Display *frame_rgb* and wait for input up to 2 seconds.
        If user pauses, wait indefinitely.

        Returns the new instruction string if the user entered one,
        otherwise ``None``.
        """
        img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        self._frame_jpeg = buf.getvalue()
        self._step = step

        # Buffer check: If an action was submitted before show was reached
        if self._user_action == "steer":
            ans = self._user_response
            self._user_action = None
            self._user_response = None
            self._paused = False
            return ans

        if self._user_action == "continue":
            self._user_action = None
            self._user_response = None
            self._paused = False
            return None

        self._user_event.clear()
        self._user_action = None
        self._user_response = None
        
        self._frame_ready.set()

        if self._paused:
            self._user_event.wait()
            if self._user_action == "steer":
                return self._user_response
            return None

        triggered = self._user_event.wait(timeout=0.5) # changed to 0.5 seconds

        if triggered:
            if self._user_action == "pause":
                self._user_event.clear()
                self._user_event.wait()
                if self._user_action == "steer":
                    return self._user_response
                return None
            elif self._user_action == "steer":
                return self._user_response
            else:
                return None
        else:
            return None

    def close(self) -> None:
        pass


# ── Main eval loop ───────────────────────────────────────────────────


def main(eval_args: SteerEvalArgs):
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

    gui: SteeringGUIWeb | None = None
    if eval_args.interactive:
        gui = SteeringGUIWeb(language_instruction, port=eval_args.gui_port)

    video = []
    scene_state_log = []
    steering_log: list[dict] = []
    horizon = env.max_episode_length
    bar = tqdm.tqdm(range(horizon))
    obs, info = env.reset(
        object_positions=initial_conditions[episode % len(initial_conditions)]
    )
    policy_client.reset()
    print(f" >>> Starting eval job from episode {episode + 1} of {rollouts} <<< ")
    while True:
        action, viz = policy_client.infer(obs, language_instruction)
        if viz is not None:
            video.append(viz)

            step_state = env.get_scene_state(cam_name="viz_cam")
            step_state["step"] = bar.n
            step_state["time"] = len(video) / 5.0
            scene_state_log.append(step_state)

            if gui is not None:
                new_instruction = gui.show(viz, step=bar.n)
                if new_instruction is not None:
                    language_instruction += new_instruction
                    policy_client.flush_actions()
                    steering_log.append(
                        {"step": bar.n, "instruction": language_instruction}
                    )
                    tqdm.tqdm.write(
                        f" [STEER] Instruction updated to:"
                        f" {language_instruction!r}"
                    )
                    action, viz = policy_client.infer(obs, language_instruction)
                    if viz is not None:
                        video[-1] = viz

        obs, rew, term, trunc, info = env.step(
            torch.tensor(action).reshape(1, -1), expensive=policy_client.rerender
        )

        bar.update(1)
        if term[0] or trunc[0] or bar.n >= horizon:
            policy_client.reset()

            # Save video and metadata
            filename = run_folder / f"episode_{episode}.mp4"
            mediapy.write_video(filename, video, fps=5)

            # Save per-step scene state log
            scene_state_path = run_folder / f"episode_{episode}_scene_state.json"
            with open(scene_state_path, "w") as f:
                json.dump(scene_state_log, f, indent=2)

            # Save steering log (empty list if no steering occurred)
            steer_path = run_folder / f"episode_{episode}_steering.json"
            with open(steer_path, "w") as f:
                json.dump(steering_log, f, indent=2)

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

            episode += 1
            video = []
            scene_state_log = []
            steering_log = []
            if episode >= rollouts:
                break

    if gui is not None:
        gui.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    args: SteerEvalArgs = tyro.cli(SteerEvalArgs)
    main(args)
