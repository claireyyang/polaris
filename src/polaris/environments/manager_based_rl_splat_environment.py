import torch
import cv2
from pathlib import Path
import numpy as np

from isaaclab.sensors.camera.camera import Camera
import isaaclab.utils.math as math
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaacsim.core.prims import GeometryPrim
from isaacsim.core.utils.stage import get_current_stage
from pxr import Semantics

from polaris.splat_renderer import SplatRenderer
from polaris.environments.rubrics import Rubric


class ManagerBasedRLSplatEnv(ManagerBasedRLEnv):
    rubric: Rubric | None = None
    _task_name: str | None = None

    def __init__(
        self,
        cfg: ManagerBasedRLEnvCfg,
        *args,
        rubric: Rubric | None = None,
        usd_file: str | None = None,
        **kwargs,
    ):
        # do dynamic setup here maybe
        if usd_file is not None:
            self.usd_file = usd_file
            cfg.dynamic_setup(usd_file)

        super().__init__(cfg=cfg, *args, **kwargs)
        self.setup_splat_world_and_robot_views()
        self.setup_splat_robot()
        self.rubric = rubric

    def _evaluate_rubric(self) -> dict:
        """Evaluate rubric and return results for info dict."""
        if self.rubric is None:
            return {
                "rubric": {
                    "success": False,
                    "progress": -1.0,
                    "metrics": {},
                }
            }

        result = self.rubric.evaluate(self)
        return {
            "rubric": {
                "success": result.success,
                "progress": result.progress,
                "metrics": result.metrics,
            }
        }

    def _get_camera_intrinsics(self, cam_name: str) -> np.ndarray:
        """Build 3x3 intrinsics matrix K for the named camera sensor."""
        cam = self.scene[cam_name]
        h_aperture = cam._sensor_prims[0].GetHorizontalApertureAttr().Get()
        v_aperture = cam._sensor_prims[0].GetVerticalApertureAttr().Get()
        focal_length = cam._sensor_prims[0].GetFocalLengthAttr().Get()
        H, W = cam.image_shape
        fx = focal_length * W / h_aperture
        fy = focal_length * H / v_aperture
        cx, cy = W / 2.0, H / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    def _world_to_pixel(
        self, points_world: np.ndarray, cam_pos: np.ndarray,
        cam_rot: np.ndarray, K: np.ndarray,
    ) -> np.ndarray:
        """Project Nx3 world points to Nx2 pixel coordinates (u, v).

        Uses OpenCV convention: camera looks along +Z, +X right, +Y down.
        ``cam_rot`` is the 3x3 rotation matrix of the camera in world frame
        (columns = camera axes in world coords, derived from
        ``quat_w_world`` via ``math.matrix_from_quat``).
        """
        R_cam_world = cam_rot.T
        t = -R_cam_world @ cam_pos
        pts_cam = (R_cam_world @ points_world.T).T + t
        # Avoid division by zero for points behind the camera
        z = pts_cam[:, 2:3].clip(min=1e-6)
        uv = (K @ (pts_cam / z).T).T[:, :2]
        return uv

    def get_scene_state(self, cam_name: str = "viz_cam") -> dict:
        """Return world-frame poses for the EE, all objects, and the
        specified camera, along with pixel-space projections of the EE
        and objects into that camera.

        Returns a JSON-serialisable dict.
        """
        state: dict = {}

        # --- End effector ---
        ee_pos = self.scene["ee_frame"].data.target_pos_w[0].detach().cpu().numpy()
        state["ee"] = {"pos_world": ee_pos.tolist()}

        # --- Rigid objects ---
        state["objects"] = {}
        for obj_name in self.scene.rigid_objects:
            obj_pos = self.scene[obj_name].data.root_pos_w[0].detach().cpu().numpy()
            obj_quat = self.scene[obj_name].data.root_quat_w[0].detach().cpu().numpy()
            state["objects"][obj_name] = {
                "pos_world": obj_pos.tolist(),
                "quat_world": obj_quat.tolist(),
            }

        # --- Camera extrinsics + intrinsics ---
        cam = self.scene[cam_name]
        cam_pos = cam.data.pos_w[0].detach().cpu().numpy()
        cam_quat = cam.data.quat_w_world[0]
        cam_rot = math.matrix_from_quat(cam_quat).detach().cpu().numpy()
        K = self._get_camera_intrinsics(cam_name)
        H, W = cam.image_shape

        state["camera"] = {
            "name": cam_name,
            "pos_world": cam_pos.tolist(),
            "quat_world": cam_quat.detach().cpu().tolist(),
            "rotation_matrix": cam_rot.tolist(),
            "intrinsics": K.tolist(),
            "resolution": [int(H), int(W)],
        }

        # --- Project EE and objects into camera pixel space ---
        all_points = [ee_pos]
        all_labels = ["ee"]
        for obj_name in self.scene.rigid_objects:
            all_points.append(
                self.scene[obj_name].data.root_pos_w[0].detach().cpu().numpy()
            )
            all_labels.append(obj_name)

        points_world = np.stack(all_points, axis=0)  # (N, 3)
        pixels = self._world_to_pixel(points_world, cam_pos, cam_rot, K)

        projections: dict = {}
        projections_pct: dict = {}
        for label, uv in zip(all_labels, pixels):
            u, v = float(uv[0]), float(uv[1])
            in_frame = 0 <= u < W and 0 <= v < H
            projections[label] = {"u": u, "v": v, "in_frame": in_frame}
            projections_pct[label] = {
                "u_pct": max(0.0, min(100.0, u / W * 100)),
                "v_pct": max(0.0, min(100.0, v / H * 100)),
            }
        state["projections"] = projections
        state["projections_percentages"] = projections_pct

        return state

    def reset(self, object_positions: dict = {}, expensive=True, *args, **kwargs):
        """
        Reset the environment

        Parameters
        ----------
        object_positions : dict
            A dictionary mapping object names to their desired poses (position and orientation).
        expensive : bool
            Whether to perform expensive (splat) rendering operations.
        """
        obs, info = super().reset(*args, **kwargs)

        # Reset rubric state
        if self.rubric:
            self.rubric.reset()

        # Following predefined initial conditions
        for obj, pose in object_positions.items():
            print(f"Setting initial condition for {obj} to {pose}")
            pose = torch.tensor(pose)[None]
            self.scene[obj].write_root_pose_to_sim(pose)
        self.sim.render()
        self.scene.update(0)
        obs = (
            self.observation_manager.compute()
        )  # update observation after setting ICs if needed
        obs["splat"] = self.custom_render(expensive, transform_static=True)

        # Evaluate rubric and add to info
        info.update(self._evaluate_rubric())

        return obs, info

    def step(self, action, expensive=True):
        """
        Steps the environment

        Parameters
        ----------
        action: torch.Tensor
            The action to take in the environment.
        expensive : bool
            Whether to perform expensive (splat) rendering operations.
        """
        obs, rew, done, trunc, info = super().step(action)
        obs["splat"] = self.custom_render(expensive)
        # obs["splat"] = {cam: self.get_robot_from_sim()[cam]["rgb"] for cam in self.get_robot_from_sim()}

        # Evaluate rubric and add to info
        info.update(self._evaluate_rubric())

        return obs, rew, done, trunc, info

    def custom_render(self, expensive: bool, transform_static: bool = False):
        """
        Render the environment
        """
        if expensive:
            self.transform_sim_to_splat(transform_static=transform_static)
            rgb = self.render_splat()
            mask_and_rgb = self.get_robot_from_sim()
            for cam in mask_and_rgb:
                og_img = (
                    rgb[cam] if cam in rgb else np.zeros_like(mask_and_rgb[cam]["rgb"])
                )
                mask = mask_and_rgb[cam]["mask"]
                sim_img = mask_and_rgb[cam]["rgb"]
                new_img = np.where(mask, sim_img, og_img)
                rgb[cam] = new_img
        else:
            rgb = {}
            for cam in self.scene.sensors:
                if isinstance(self.scene.sensors[cam], Camera):
                    rgb[cam] = (
                        self.scene[cam].data.output["rgb"][0].detach().cpu().numpy()
                    )
        return rgb

    def setup_splat_world_and_robot_views(self):
        splats = {}
        self.views = {}
        stage = get_current_stage()

        # Allocate splats for all rigid objects in the scene and raytrace semantic tags
        for name in self.scene.rigid_objects:
            path = Path(self.usd_file).parent / "assets" / name / "splat.ply"
            if path.exists():
                splats[name] = path
            else:
                # apply semantic tags
                prim = stage.GetPrimAtPath(f"/World/envs/env_0/scene/{name}")
                semantic_type = "class"
                semantic_value = "raytraced"
                instance_name = f"{semantic_type}_{semantic_value}"
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                sem.CreateSemanticTypeAttr()
                sem.CreateSemanticDataAttr()
                sem.GetSemanticTypeAttr().Set(semantic_type)
                sem.GetSemanticDataAttr().Set(semantic_value)

        # Setup splat cameras with intrinsics and resolution from sim cameras
        camera_cfg = {}
        for name in self.scene.sensors:
            if not isinstance(self.scene.sensors[name], Camera):
                continue
            resolution = self.scene.sensors[name].image_shape
            h_aperture = (
                self.scene[name]._sensor_prims[0].GetHorizontalApertureAttr().Get()
            )
            v_aperture = (
                self.scene[name]._sensor_prims[0].GetVerticalApertureAttr().Get()
            )
            f = self.scene[name]._sensor_prims[0].GetFocalLengthAttr().Get()
            fovx = 2 * np.arctan(h_aperture / (2 * f))
            fovy = 2 * np.arctan(v_aperture / (2 * f))
            camera_cfg[name] = {
                "res": resolution,
                "fovx": fovx,
                "fovy": fovy,
            }
        self.splat_renderer = SplatRenderer(splats=splats, device=self.device)
        self.splat_renderer.init_cameras(camera_cfg)

    def setup_splat_robot(self):
        # Allocate robot splats and views on robot links to track
        more_splats = {}
        robot_asset_path = Path(self.cfg.scene.robot.spawn.usd_path).parent
        for ply in sorted(list(robot_asset_path.glob("SEGMENTED/*.ply"))):
            more_splats[ply.stem] = ply
            sim_path = ply.stem.replace("-", "/")
            view = GeometryPrim(
                prim_paths_expr=f"/World/envs/env_0/robot/{sim_path}",
                reset_xform_properties=False,
            )
            print(f"/World/envs/env_0/robot/{sim_path}")
            self.views[ply.stem] = view
        self.splat_renderer.add_splats(more_splats)

    def get_robot_from_sim(self):
        # TODO: comment this. does this get only robot? objects too?
        ret = {}
        for cam in self.scene.sensors:
            if not isinstance(self.scene.sensors[cam], Camera):
                continue
            base_cam = self.scene[cam]
            mask = (
                base_cam.data.output["semantic_segmentation"][0].detach().cpu().numpy()
            )
            img = base_cam.data.output["rgb"][0].detach().cpu().numpy()
            mask = np.where(mask >= 2, 1, 0)

            ret[cam] = {"rgb": img, "mask": mask}

        return ret

    def transform_sim_to_splat(self, transform_static=False):
        """
        Update splat renderer transforms from simulation

        Parameters
        ----------
        transform_static : bool
            Whether to also transform static objects (like environment).
        """
        all_transforms = {}

        # rigid bodies
        for name in self.scene.rigid_objects:
            path = Path(self.usd_file).parent / "assets" / name / "splat.ply"
            if (
                "static" not in name or transform_static
            ) and path.exists():  # splat exists
                pos = self.scene[name].data.root_state_w[0, :3]
                quat = self.scene[name].data.root_state_w[0, 3:7]
                all_transforms[name] = (pos, quat)

        #  robot - this will only fire if setup_splat_robot has been called otherwise views will be empty
        for v_name in self.views:
            view = self.views[v_name]
            pos, quat = view.get_world_poses(usd=False)
            pos, quat = pos.squeeze(), quat.squeeze()
            all_transforms[v_name] = (pos, quat)

        if len(all_transforms) > 0:  # only transform if there is something to transform
            self.splat_renderer.transform_many(all_transforms)

        # set all cameras so that static cameras are set
        if transform_static:
            cam_extrinsics_dict = {}
            for name in self.splat_renderer.cameras:
                pos = self.scene[name].data.pos_w[0].detach().cpu().numpy()
                quat = self.scene[name].data.quat_w_world[0]

                rot = math.matrix_from_quat(quat).detach().cpu().numpy()
                cam_extrinsics_dict[name] = {"pos": pos, "rot": rot}

            if len(self.splat_renderer.pcds) > 0:
                self.splat_renderer.render(cam_extrinsics_dict)

    def render_splat(self):
        # get camera extrinsics
        cam_extrinsics_dict = {}
        for name in self.splat_renderer.cameras:
            if "wrist" in name:
                pos = self.scene[name].data.pos_w[0].detach().cpu().numpy()
                quat = self.scene[name].data.quat_w_world[0]

                rot = math.matrix_from_quat(quat).detach().cpu().numpy()
                cam_extrinsics_dict[name] = {"pos": pos, "rot": rot}

        # perform splat rendering
        if len(self.splat_renderer.pcds) > 0:
            rgb = self.splat_renderer.render(cam_extrinsics_dict)
        else:
            rgb = {
                name: torch.zeros(
                    (
                        self.splat_renderer.cameras[name].image_height,
                        self.splat_renderer.cameras[name].image_width,
                        3,
                    )
                )
                for name in cam_extrinsics_dict
            }

        # process output
        for k, v in rgb.items():
            rgb[k] = v.detach().cpu().numpy()
            rgb[k] = np.clip(rgb[k], 0, 1)
            rgb[k] = (rgb[k] * 255).astype(np.uint8)

            # TODO: why is there a resize?
            rgb[k] = cv2.resize(rgb[k], (rgb[k].shape[1] // 2, rgb[k].shape[0] // 2))
            rgb[k] = cv2.resize(rgb[k], (rgb[k].shape[1] * 2, rgb[k].shape[0] * 2))

        return rgb
