import numpy as np
from openpi_client import websocket_client_policy, image_tools
from polaris.policy.abstract_client import InferenceClient, PolicyArgs


# Franka Panda modified DH parameters: [a, d, alpha]
_PANDA_DH = np.array([
    [0.0,     0.333,  0.0],
    [0.0,     0.0,   -np.pi / 2],
    [0.0,     0.316,  np.pi / 2],
    [0.0825,  0.0,    np.pi / 2],
    [-0.0825, 0.384, -np.pi / 2],
    [0.0,     0.0,    np.pi / 2],
    [0.088,   0.0,    np.pi / 2],
])

_PANDA_FLANGE_OFFSET = np.array([
    [0.7071, -0.7071, 0, 0],
    [0.7071,  0.7071, 0, 0],
    [0,       0,      1, 0.107],
    [0,       0,      0, 1],
])

_PANDA_JOINT_LIMITS_LOW = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
)
_PANDA_JOINT_LIMITS_HIGH = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
)


def _dh_matrix(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d],
        [0,   0,        0,       1],
    ])


def panda_fk(joint_positions: np.ndarray) -> np.ndarray:
    """Forward kinematics: 7 joint angles -> 4x4 EEF pose."""
    T = np.eye(4)
    for i in range(7):
        a, d, alpha = _PANDA_DH[i]
        T = T @ _dh_matrix(a, d, alpha, joint_positions[i])
    T = T @ _PANDA_FLANGE_OFFSET
    return T


def _numerical_jacobian(joint_positions: np.ndarray, eps=1e-6) -> np.ndarray:
    """6x7 geometric Jacobian via finite differences."""
    J = np.zeros((6, 7))
    T0 = panda_fk(joint_positions)
    p0 = T0[:3, 3]
    R0 = T0[:3, :3]
    for i in range(7):
        q_perturbed = joint_positions.copy()
        q_perturbed[i] += eps
        T1 = panda_fk(q_perturbed)
        J[:3, i] = (T1[:3, 3] - p0) / eps
        dR = T1[:3, :3] @ R0.T
        # Extract angular velocity from skew-symmetric part of dR
        J[3, i] = (dR[2, 1] - dR[1, 2]) / (2 * eps)
        J[4, i] = (dR[0, 2] - dR[2, 0]) / (2 * eps)
        J[5, i] = (dR[1, 0] - dR[0, 1]) / (2 * eps)
    return J


def ik_delta_to_joint_targets(
    current_joints: np.ndarray,
    eef_delta: np.ndarray,
    damping: float = 0.05,
) -> np.ndarray:
    """Convert a 6-D EEF delta (dx,dy,dz,droll,dpitch,dyaw) to absolute joint targets.

    Uses damped least-squares (Levenberg-Marquardt) on the Jacobian.
    """
    J = _numerical_jacobian(current_joints)
    JJT = J @ J.T + (damping ** 2) * np.eye(6)
    dq = J.T @ np.linalg.solve(JJT, eef_delta)
    return current_joints + dq


# ──────────────────────────────────────────────────────────────────────
# Tunable constants — adjust these to match your specific Bridge VLA
# ──────────────────────────────────────────────────────────────────────
EEF_DELTA_SCALE = 1.0       # Scale factor applied to raw EEF deltas before IK
IK_DAMPING = 0.05           # Damped least-squares damping factor
IMAGE_SIZE = 256             # Resize target; Bridge models often expect 256x256


@InferenceClient.register(client_name="SteerablePolicies")
class SteerablePoliciesClient(InferenceClient):
    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        self.open_loop_horizon = args.open_loop_horizon or 1
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        exterior_image_viz = None

        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            curr_obs = self._extract_observation(obs)
            self.actions_from_chunk_completed = 0

            exterior_image = image_tools.resize_with_pad(
                curr_obs["right_image"], IMAGE_SIZE, IMAGE_SIZE
            )
            wrist_image = image_tools.resize_with_pad(
                curr_obs["wrist_image"], IMAGE_SIZE, IMAGE_SIZE
            )

            # ── Build the request for your Bridge VLA server ──
            # TODO: adjust these keys to match your server's expected input schema
            request_data = {
                "image": exterior_image,
                "wrist_image": wrist_image,
                "instruction": instruction,
            }
            server_response = self.client.infer(request_data)

            # Bridge VLAs typically return 7-D per step:
            #   [dx, dy, dz, droll, dpitch, dyaw, gripper]
            # Some return a single step, others return a chunk.
            raw_actions = np.asarray(server_response["actions"])
            if raw_actions.ndim == 1:
                raw_actions = raw_actions[np.newaxis, :]

            joint_targets = []
            current_joints = curr_obs["joint_position"].copy()
            for step_action in raw_actions:
                eef_delta = step_action[:6] * EEF_DELTA_SCALE
                gripper = step_action[6] if step_action.shape[0] > 6 else 0.0

                target_joints = ik_delta_to_joint_targets(
                    current_joints, eef_delta, damping=IK_DAMPING
                )
                target_joints = np.clip(
                    target_joints, _PANDA_JOINT_LIMITS_LOW, _PANDA_JOINT_LIMITS_HIGH
                )

                full_action = np.concatenate([target_joints, [gripper]])
                joint_targets.append(full_action)
                current_joints = target_joints

            self.pred_action_chunk = np.array(joint_targets)
            exterior_image_viz = curr_obs["viz_camera"]

        if return_viz and exterior_image_viz is None:
            curr_obs = self._extract_observation(obs)
            exterior_image_viz = curr_obs["viz_camera"]

        if self.pred_action_chunk is None:
            raise ValueError("No action chunk predicted")

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # Binarize gripper: Bridge uses continuous [-1, 1] or [0, 1] for gripper
        if action[-1] > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        return action, exterior_image_viz

    def _extract_observation(self, obs_dict):
        right_image = obs_dict["splat"]["external_cam"]
        wrist_image = obs_dict["splat"]["wrist_cam"]
        viz_camera = obs_dict["splat"]["viz_cam"]

        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()[0]
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()[0]

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "viz_camera": viz_camera,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
