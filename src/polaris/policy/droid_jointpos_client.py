import numpy as np
from openpi_client import websocket_client_policy, image_tools
from polaris.policy.abstract_client import InferenceClient, PolicyArgs


# Joint Position Client for DROID
@InferenceClient.register(client_name="DroidJointPos")
class DroidJointPosClient(InferenceClient):
    def __init__(self, args: PolicyArgs) -> None:
        self.args = args
        if args.open_loop_horizon is None:
            raise ValueError("open_loop_horizon must be set for DroidJointPosClient")

        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None
        self.open_loop_horizon = args.open_loop_horizon

    @property
    def rerender(self) -> bool:
        return (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        )

    def visualize(self, request: dict):
        """
        Return the camera views how the model sees it
        """
        curr_obs = self._extract_observation(request)
        # base_img = image_tools.resize_with_pad(curr_obs["right_image"], 896, 896)
        viz_img = image_tools.resize_with_pad(curr_obs["viz_camera"], 896, 896)
        # You can combine multiple views if needed:
        # combined = np.concatenate([base_img, viz_img], axis=1)
        return viz_img  # or return combined

    def reset(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def discard_action_chunk(self):
        self.actions_from_chunk_completed = 0
        self.pred_action_chunk = None

    def infer(
        self, obs: dict, instruction: str, return_viz: bool = False
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Infer the next action from the policy in a server-client setup
        """
        exterior_image_viz = None
        if (
            self.actions_from_chunk_completed == 0
            or self.actions_from_chunk_completed >= self.open_loop_horizon
        ):
            curr_obs = self._extract_observation(obs)

            self.actions_from_chunk_completed = 0
            exterior_image = image_tools.resize_with_pad(
                curr_obs["right_image"], 224, 224
            )
            wrist_image = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
            request_data = {
                "observation/exterior_image_1_left": exterior_image,
                "observation/wrist_image_left": wrist_image,
                "observation/joint_position": curr_obs["joint_position"],
                "observation/gripper_position": curr_obs["gripper_position"],
                "prompt": instruction,
            }
            server_response = self.client.infer(request_data)
            self.pred_action_chunk = server_response["actions"]
            
            exterior_image_viz = image_tools.resize_with_pad(
                curr_obs["viz_camera"], 896, 896
            )

        if return_viz and exterior_image_viz is None:
            curr_obs = self._extract_observation(obs)
            exterior_image_viz = image_tools.resize_with_pad(curr_obs["viz_camera"], 896, 896)

        if self.pred_action_chunk is None:
            raise ValueError("No action chunk predicted")

        action = self.pred_action_chunk[self.actions_from_chunk_completed]
        self.actions_from_chunk_completed += 1

        # binarize gripper action
        if action[-1].item() > 0.5:
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])

        return action, exterior_image_viz

    def _extract_observation(self, obs_dict):
        # Assign images
        right_image = obs_dict["splat"]["external_cam"]
        wrist_image = obs_dict["splat"]["wrist_cam"]
        viz_camera = obs_dict["splat"]["viz_cam"]

        # Capture proprioceptive state
        robot_state = obs_dict["policy"]
        joint_position = robot_state["arm_joint_pos"].clone().detach().cpu().numpy()[0]
        gripper_position = robot_state["gripper_pos"].clone().detach().cpu().numpy()[0]

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "viz_camera": viz_camera,  # Add to returned dict
            "joint_position": joint_position,
            "gripper_position": gripper_position,
        }
