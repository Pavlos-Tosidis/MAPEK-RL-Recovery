"""Webots TurtleBot navigation environment with persistent sensor-coordinate faults.

Derived from the final realistic-fault environment supplied with the IEEE Access
recovery implementation. Requires its original Webots world/PROTOS/assets.
"""
from controller import Supervisor, Display
import random
import math
import torch
from transforms3d import quaternions

from ultralytics import YOLO

from abc import ABC
import numpy as np

import cv2
from numpy import inf

from gymnasium import spaces
import gymnasium as gym


LIVE_IMAGE_SHAPES = ("rectangle", "ellipse", "polygon", "blob", "strip")
LIVE_IMAGE_VARIANTS = ("black", "gray", "random")
LIVE_SEVERITIES = (0.05, 0.10, 0.20, 0.35)
LIVE_LIDAR_DISTANCES_M = (0.05, 0.10, 0.20, 0.30, 0.50)


def _rescale_mask_to_target(mask, target, max_iter=6):
    h, w = mask.shape
    out = (mask > 0).astype(np.uint8)
    for _ in range(max_iter):
        actual = int(out.sum())
        if actual <= 0:
            break
        ratio = actual / float(target)
        if 0.90 <= ratio <= 1.10:
            break
        ys, xs = np.where(out > 0)
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        crop = out[y0:y1, x0:x1]
        scale = float(np.sqrt(target / float(actual)))
        new_w = max(1, min(w, int(round(crop.shape[1] * scale))))
        new_h = max(1, min(h, int(round(crop.shape[0] * scale))))
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        cx = (x0 + x1 - 1) / 2.0
        cy = (y0 + y1 - 1) / 2.0
        nx0 = int(round(cx - new_w / 2.0))
        ny0 = int(round(cy - new_h / 2.0))
        nx0 = int(np.clip(nx0, 0, max(0, w - new_w)))
        ny0 = int(np.clip(ny0, 0, max(0, h - new_h)))
        candidate = np.zeros_like(out)
        candidate[ny0:ny0+new_h, nx0:nx0+new_w] = resized
        out = candidate
    return out


def _random_image_fault(h, w):
    severity = float(random.choice(LIVE_SEVERITIES))
    shape = str(random.choice(LIVE_IMAGE_SHAPES))
    variant = str(random.choice(LIVE_IMAGE_VARIANTS))
    if variant == "black":
        color = (0, 0, 0)
    elif variant == "gray":
        color = (127, 127, 127)
    else:
        color = tuple(int(x) for x in np.random.randint(0, 256, size=3))

    target = max(1, int(round(h * w * severity)))
    mask = np.zeros((h, w), dtype=np.uint8)
    aspect = float(np.exp(np.random.uniform(np.log(0.45), np.log(2.2))))
    base_area = target * 4.0 / math.pi if shape == "ellipse" else target
    bw = int(np.clip(round(math.sqrt(base_area * aspect)), 3, w - 2))
    bh = int(np.clip(round(base_area / max(1, bw)), 3, h - 2))
    cx = random.randint(max(1, bw // 2), max(bw // 2 + 1, w - bw // 2 - 1))
    cy = random.randint(max(1, bh // 2), max(bh // 2 + 1, h - bh // 2 - 1))

    if shape == "rectangle":
        x0 = int(np.clip(cx - bw // 2, 0, w - 1))
        y0 = int(np.clip(cy - bh // 2, 0, h - 1))
        x1 = int(np.clip(x0 + bw, 1, w))
        y1 = int(np.clip(y0 + bh, 1, h))
        mask[y0:y1, x0:x1] = 1
    elif shape == "ellipse":
        cv2.ellipse(
            mask, (cx, cy), (max(1, bw // 2), max(1, bh // 2)),
            float(np.random.uniform(0, 180)), 0, 360, 1, -1
        )
    elif shape in {"polygon", "blob"}:
        npts = random.randint(5, 9 if shape == "polygon" else 13)
        angles = np.sort(np.random.uniform(0, 2 * np.pi, npts))
        radial = np.random.uniform(0.65 if shape == "polygon" else 0.35, 1.0, npts)
        pts = []
        for a, r in zip(angles, radial):
            x = int(round(cx + math.cos(a) * (bw / 2) * r))
            y = int(round(cy + math.sin(a) * (bh / 2) * r))
            pts.append([int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))])
        pts = np.asarray(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 1)
        if shape == "blob":
            k = max(3, int(round(min(bw, bh) * 0.08)) | 1)
            blurred = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
            mask = (blurred > 0.35).astype(np.uint8)
    elif shape == "strip":
        length = int(np.clip(max(bw, bh, int(math.sqrt(target) * 2.5)), 10, min(w, h) * 2))
        thickness = int(np.clip(round(target / max(1, length)), 3, min(h, w) // 2))
        angle = float(np.random.uniform(0, 180))
        rect = ((float(cx), float(cy)), (float(length), float(thickness)), angle)
        pts = cv2.boxPoints(rect).astype(np.int32)
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [pts], 1)

    mask = _rescale_mask_to_target(mask, target)
    actual = int(mask.sum())
    iters = 0
    while actual < 0.90 * target and iters < 12:
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        actual = int(mask.sum())
        iters += 1

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("Generated empty image anomaly mask.")
    bbox = [
        int(xs.min()), int(ys.min()),
        int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    ]
    spec = {
        "shape": shape,
        "color": list(color),
        "severity_target": severity,
        "actual_area": int(mask.sum()),
        "actual_fraction": float(mask.mean()),
        "bounding_rect": bbox,
    }
    return mask.astype(bool), spec


def _random_lidar_fault(raw_len):
    severity = float(random.choice(LIVE_SEVERITIES))
    width = max(1, min(raw_len, int(round(raw_len * severity))))
    start = random.randrange(raw_len)
    distance_m = float(random.choice(LIVE_LIDAR_DISTANCES_M))
    value = float(1.0 - distance_m / 12.5)
    return {
        "start": int(start),
        "width": int(width),
        "distance_m": distance_m,
        "value": value,
        "noise_std": 0.001,
        "severity": severity,
    }



class Env(gym.Env, ABC):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed=1, action_space='discrete', observation_space=['image', 'lidar', 'world_info'],
                 kb_control=False, inference_mode=False):
        super(Env, self).__init__()

        self.verbose = False
        # Webots Environment
        # Seed
        self.seed = seed
        # continuous or discrete actionspace:
        self.a_space = action_space
        self.o_space = observation_space
        # Set seeds
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Webots supervisor, robot and devices
        self.supervisor = Supervisor()
        self.robot = self.supervisor.getSelf()
        # Motors
        self.left_motor = self.supervisor.getDevice("left wheel motor")
        self.right_motor = self.supervisor.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)
        # Robot rotation and position
        self.rotation = self.robot.getField('rotation')
        self.position = self.robot.getField('translation')
        self.timestep = int(self.supervisor.getBasicTimeStep())
        # Root and children of world structure
        self.root = self.supervisor.getRoot()
        self.children = self.root.getField('children')

        # Robot camera compass, gyro, lidar and touchsensor
        self.camera = self.supervisor.getDevice('camera')
        self.camera.enable(self.timestep)
        self.display = self.supervisor.getDevice('display')
        self.compass = self.supervisor.getDevice('compass')
        self.compass.enable(self.timestep)
        self.gyro = self.supervisor.getDevice('gyro')
        self.gyro.enable(self.timestep)
        self.lidar = self.supervisor.getDevice('LDS-01')
        self.lidar.enable(self.timestep)
        self.sensor = self.supervisor.getDevice('touch sensor')
        self.sensor.enable(self.timestep)
        self.lidar_main_motor = self.supervisor.getDevice('LDS-01_main_motor')
        self.lidar_secondary_motor = self.supervisor.getDevice('LDS-01_secondary_motor')
        self.lidar_main_motor.setPosition(float('inf'))
        self.lidar_secondary_motor.setPosition(float('inf'))
        self.lidar_main_motor.setVelocity(30.0)
        self.lidar_secondary_motor.setVelocity(60.0)

        self.startingpoint = random.randint(0, 1080)
        self.leng = random.randint(50, 500)

        if kb_control:  # enable keyboard control
            self.supervisor.keyboard.enable(self.timestep)
        # ----------------------------- GYM STUFF ----------------------------- #
        # Action Spaces
        if action_space == 'discrete':
            self.action_space = spaces.Discrete(6)
        if action_space == 'continuous':
            self.action_space = spaces.Box(-1.0, 1.0, (3,))

        # Observation Spaces
        self.padding = 60  # padding for lidar data
        self.vector_size = self.lidar.getHorizontalResolution() + 2 * self.padding
        self.img_size = (270, 480, 3)
        self.framestack = 8

        if observation_space == ['image']:
            self.observation_space = spaces.Box(low=0, high=255, shape=self.img_size, dtype=np.uint8)
        else:
            observation_dict = {}
            if 'image' in observation_space:
                observation_dict['image'] = spaces.Box(0, 255, self.img_size, dtype=np.uint8)
            if 'lidar' in observation_space:
                observation_dict['lidar'] = spaces.Box(0, 1, (self.framestack, self.vector_size,), dtype=np.float64)
            if 'world_info' in observation_space:
                observation_dict['world_info'] = spaces.Box(0, 1, (1 * 3,), dtype=np.float64)
            if 'image_health' in observation_space:
                observation_dict['image_health'] = spaces.Box(0, 1, (1,), dtype=np.float64)
            if 'lidar_health' in observation_space:
                observation_dict['lidar_health'] = spaces.Box(0, 1, (1,), dtype=np.float64)

            self.observation_space = spaces.Dict(
                spaces=observation_dict
            )
        self.lidar_obs = []
        self.world_obs = []
        self.obs = {}
        self.reward_range = [-100, 100]
        self.metadata = None
        self.step_counter = 0
        self.timesteps = 0
        self.frameskip = 4
        self.info = None
        # for reward
        self.distance, self.prev_distance, self.angle, self.prev_angle, self.angle_pi = \
            100.0, 100.0, 100.0, 100.0, 100.0
        self.target_distance = 1.0
        self.just_reset = False
        self.min_distance = 100.0
        self.min_angle = 10.0
        self.motor_speeds = [0, 0]
        self.keyboard_control = kb_control
        self.distance_to_target = 100
        self.cumulative_reward = 0
        self.person_detector, self.filtered = None, None
        self.load_models()
        self.found = False
        self.human = None
        self.closest_robot_entrance, self.closest_human_entrance = None, None
        self.next_target = None
        self.target = None
        self.rooms = [[-6, 6], [-6, 0], [-6, -6],
                      [0, 6], [0, 0], [0, -6],
                      [6, 6], [6, 0], [6, -6]]

        self.ano_x = random.randint(0, 240)
        self.ano_y = random.randint(0, 135)
        self.ano_width = random.randint(10, 240)
        self.ano_height = random.randint(10, 135)
        self.ano_col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.save_counter = 0
        self.img_name = 0
        self.d_reward = 0
        self.a_reward = 0
        self.lidar_anomaly = False
        self.image_anomaly = False
        self.prev_lidarData = None

        self.total_env_steps = 0
        self.ano_ramp_steps = 2_000_000
        self.p_partial_max = 0.15  # 15% partial corruption at end of ramp
        self.p_full_max = 0.25  # 25% full single-sensor failure at end of ramp
        self.p_img_vs_lidar = 0.5  # 50/50 image vs lidar
        self.ano_mode = "normal"  # "normal"|"partial_image"|"partial_lidar"|"full_image"|"full_lidar"
        self.image_health = np.array([1.0], dtype=np.float32)
        self.lidar_health = np.array([1.0], dtype=np.float32)
        self.anomalies_enabled = True
        self.gate_debug = {}
        self.inference_mode = inference_mode

        # MAPE-K adaptation state. In real deployment this is written by the
        # Analyze phase after a 360-degree scan. During finetuning/evaluation the
        # environment reuses this exact corruption instead of resampling it.
        self.forced_anomaly_mode = None
        self.forced_image_region = None
        self.forced_lidar_region = None
        self.forced_image_mask = None
        self.forced_image_spec = None
        self.forced_lidar_obstruction = None

    def _sample_live_partial_fault_descriptor(self):
        """Sample the physical/live fault once per episode in sensor coordinates.

        Random inference faults use the same families used by the detector dataset:
        arbitrary persistent image masks and close-range persistent LiDAR sectors.
        """
        # Clear descriptors from a previous randomly sampled episode. Experiment-
        # forced faults are handled separately through forced_anomaly_mode.
        self.forced_image_region = None
        self.forced_lidar_region = None
        self.forced_image_mask = None
        self.forced_image_spec = None
        self.forced_lidar_obstruction = None

        if self.ano_mode in {"partial_image", "partial_both"}:
            mask, spec = _random_image_fault(self.img_size[0], self.img_size[1])
            self.forced_image_mask = mask
            self.forced_image_spec = spec
            self.ano_col = tuple(int(c) for c in spec["color"])
            x, y, w, h = spec["bounding_rect"]
            self.ano_x = int(x)
            self.ano_y = int(y)
            self.ano_width = int(w)
            self.ano_height = int(h)

        if self.ano_mode in {"partial_lidar", "partial_both"}:
            raw_len = int(self.lidar.getHorizontalResolution())
            spec = _random_lidar_fault(raw_len)
            self.forced_lidar_obstruction = spec
            self.startingpoint = int(spec["start"])
            self.leng = int(spec["width"])

    def load_models(self):
        self.person_detector = YOLO('yolo11n.pt')

    def reset(self, seed=None):
        # Image anomaly
        self.ano_x = random.randint(0, 240)
        self.ano_y = random.randint(0, 135)
        self.ano_width = random.randint(10, 240)
        self.ano_height = random.randint(10, 135)
        self.ano_col = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        self.save_counter = 0
        self.startingpoint = random.randint(0, 1080)
        self.leng = random.randint(400, 1000)
        self.rooms = [[-6, 6], [-6, 0], [-6, -6],
                      [0, 6], [0, 0], [0, -6],
                      [6, 6], [6, 0], [6, -6]]

        self.just_reset = True
        self.closest_robot_entrance, self.closest_human_entrance = None, None
        self.next_target = None
        self.min_distance = 100.0
        self.min_angle = 10.0
        self.found = False
        (self.distance, self.prev_distance, self.angle, self.prev_angle,
         self.distance_to_target, self.angle_pi) = 100.0, 100.0, 100.0, 100.0, 100.0, 100.0
        self.supervisor.simulationReset()
        self.supervisor.simulationResetPhysics()
        self.supervisor.step(self.timestep)
        self.replace_humans()
        self.initialize_world()
        self.motor_speeds = [0, 0]
        self.rooms = [[-6, 6], [-6, 0], [-6, -6],
                      [0, 6], [0, 0], [0, -6],
                      [6, 6], [6, 0], [6, -6]]
        room = self.get_random_room()
        x = random.uniform(room[0] - 2.5, room[0] + 2.5)
        y = random.uniform(room[1] - 2.5, room[1] + 2.5)
        self.position.setSFVec3f([x, y, -0.0004])
        self.define_target()
        if self.anomalies_enabled:
            self._sample_anomaly_mode()
            if self.forced_anomaly_mode is None:
                self._sample_live_partial_fault_descriptor()
        if self.forced_anomaly_mode is not None:
            self._apply_forced_anomaly()
        for i in range(random.randint(2, 3)):
            self.left_motor.setPosition(float('inf'))  # NOQA
            self.right_motor.setPosition(float('inf'))  # NOQA
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
            # Step world forward
            self.supervisor.step(self.timestep)
            self.get_reward()
        if self.sensor.getValue() != 0.0 or self.distance < 0.2:
            self.reset()
        self.cumulative_reward = 0
        self.prev_lidarData = None
        # Grab new observation
        obs = self.get_obs()
        self.get_reward()
        self.step_counter = 0
        self.image_health = np.array([1.0], dtype=np.float32)
        self.lidar_health = np.array([1.0], dtype=np.float32)

        return obs, {}

    def step(self, action=0):
        color_map = {
            "normal": 0x00ff00,
            "partial_lidar": 0xffff00,
            "partial_image": 0xffff00,
            "full_lidar": 0xff0000,
            "full_image": 0xff0000,
            "partial_both": 0xff8800,
            "full_both": 0xff0000,
        }

        label_color = color_map.get(self.ano_mode, 0xffffff)
        info = {"TimeLimit.truncated": False}
        self.step_counter += 1
        self.total_env_steps += 1
        self.timesteps += 1
        done = False
        truncated = False
        collided = False
        reward = 0
        self.found = False

        if self.keyboard_control:
            key = self.supervisor.keyboard.getKey()
            action = self.apply_key(key)

        for i in range(self.frameskip):
            self.apply_action(action)
            self.supervisor.step(self.timestep)
            reward += self.get_reward()

        #penalize backwards movement
        if self.a_space == 'continuous':
            if self.motor_speeds[0] < 0.0 and self.motor_speeds[1] < 0.0 and reward > 0:
                reward *= -1

        if self.a_space == 'discrete':
            if action == 1 and reward > 0:
                reward *= -1

        # Grab new observation
        obs = self.get_obs()

        # TimeLimit end of an episode
        if self.step_counter == 500:
            reward += 0
            # done = True
            info = {"TimeLimit.truncated": True}
            info['truncated'] = True
            truncated = True

        # Collision end of an episode
        if self.sensor.getValue() != 0.0 or self.distance_to_target < 0.2:
            done = True
            reward = -10
            collided = True
            info['collided'] = True

        if action == 4:  # and self.found:
            if 0.3 <= self.distance <= 3.0 and abs(self.angle_pi) < 20.0:
                done = True
                reward = 100
                info["success"] = True
            else:
                reward -= 0.01 * self.frameskip

        if action == 5 or action == 1:
            reward -= 0.01 * self.frameskip

        self.cumulative_reward += reward
        self.supervisor.setLabel(0, 'Episode Reward: ' + str(round(self.cumulative_reward, 2)), 0, 0, 0.1, 0xff0000, 0,
                                 "Arial")

        self.supervisor.setLabel(1, 'Anomaly: ' +
                                 str(self.ano_mode), 0, 0.1, 0.1, 0xff0000, 0, "Arial")

        # if done or truncated:  # and self.verbose:
        #     print('Episode reward is {} Steps: {}'.format(self.cumulative_reward, self.step_counter))
        #     print('Truncated and collision: ', truncated, collided)

        info["anomaly_type"] = self.ano_mode

        gate_text = "\n".join(
            [f"{k}: {v:.3f}" for k, v in self.gate_debug.items()]
        )

        self.supervisor.setLabel(
            2,
            gate_text,
            0, 0.2, 0.08,
            label_color,
            0,
            "Arial"
        )

        return obs, reward, done, truncated, info

    def apply_action(self, action):  # CLEANED
        if self.a_space == 'discrete':
            self.motor_speeds = [0.0, 0.0]
            position = self.position.getSFVec3f()
            # print('Action taken: ', action)
            if action == 0:  # Move forward
                ori = np.array(self.robot.getOrientation()).reshape([3, 3])
                b = np.array([0.01, 0, 0])
                new_p = ori.dot(b) + self.position.getSFVec3f()
                new_p[2] = position[2]
                self.position.setSFVec3f(list(new_p))
            elif action == 1:  # Move Backwards
                ori = np.array(self.robot.getOrientation()).reshape([3, 3])
                b = np.array([-0.01, 0, 0])
                new_p = ori.dot(b) + self.position.getSFVec3f()
                new_p[2] = position[2]
                self.position.setSFVec3f(list(new_p))
            elif action == 2:  # turn left
                rotation = self.rotation.getSFRotation()
                q1 = quaternions.axangle2quat(rotation[0:3], rotation[3])
                q2 = quaternions.axangle2quat([0, 0, 1], 1 * 3.14 / 180)
                q = quaternions.qmult(q1, q2)
                vec, angle = quaternions.quat2axangle(q)
                new_rotation = [vec[0], vec[1], vec[2], angle]
                # new_rotation = [0, 0, 1, angle]
                self.rotation.setSFRotation(new_rotation)
            elif action == 3:  # turn right
                rotation = self.rotation.getSFRotation()
                q1 = quaternions.axangle2quat(rotation[0:3], rotation[3])
                q2 = quaternions.axangle2quat([0, 0, 1], -1 * 3.14 / 180)
                q = quaternions.qmult(q1, q2)
                vec, angle = quaternions.quat2axangle(q)
                new_rotation = [vec[0], vec[1], vec[2], angle]
                # new_rotation = [0, 0, 1, angle]
                self.rotation.setSFRotation(new_rotation)
            elif action == 4:  # apply person detection
                # pass
                self.get_obs(run_dl=True)
            elif action == 5:  # Noop
                pass

        if self.a_space == 'continuous':
            self.motor_speeds[0] = action[0]
            self.motor_speeds[1] = action[1]
            self.set_velocity(self.motor_speeds[0], self.motor_speeds[1])
            if action[2] > 0.0:
                self.get_obs(run_dl=False)

    def apply_key(self, key):
        if key == 315:  # getforward
            if self.a_space == 'discrete':
                action = 0
            else:
                action = [1.0, 1.0, 1.0]
        elif key == 317:  # backward
            if self.a_space == 'discrete':
                action = 1
            else:
                action = [-1.0, -1.0, 1.0]
        elif key == 314:  # left
            if self.a_space == 'discrete':
                action = 2
            else:
                action = [-1.0, 1.0, 1.0]
        elif key == 316:  # right
            if self.a_space == 'discrete':
                action = 3
            else:
                action = [1.0, -1.0, 1.0]
        elif key == 32:
            if self.a_space == 'discrete':
                action = 4
            else:
                action = [0.0, 0.0, 1.0]
        else:
            if self.a_space == 'discrete':
                action = 5
            else:
                action = [0.0, 0.0, 1.0]
        return action

    def get_obs(self, run_dl=False):
        self.save_counter += 1
        if 'image' in self.o_space:
            cameraData = self.camera.getImage()
            if cameraData:
                frame = np.frombuffer(cameraData, np.uint8).reshape(
                    (self.camera.getHeight(), self.camera.getWidth(), 4))
                frame = frame[:, :, :3]
                # if self.save_counter % 30 == 0:
                #     self.img_name += 1
                #     frame_to_save = frame.astype(np.uint8).copy()
                #     cv2.imwrite('/media/pavlos/2tb/Turtlebot_and_drone/turtlebot_anomalies/image_dataset/'
                #                 + str(self.img_name) + '.jpg', frame_to_save)
                if run_dl:
                    self.deploy_dl(frame)
                # if self.image_anomaly: # Camera anomaly
                #     frame = frame.astype(np.uint8).copy()
                #     frame = np.full_like(frame, 127, dtype=np.uint8)  # mid-gray, not black
                if self.ano_mode in {"full_image", "full_both"}:
                    frame = frame.astype(np.uint8).copy()
                    frame[:] = 0.0  # complete failure (black)
                    self.image_health = np.array([0.0], dtype=np.float32)
                elif self.ano_mode in {"partial_image", "partial_both"}:
                    # partial: rectangle occlusion (you already have apply_camera_anomaly)
                    frame = self.apply_camera_anomaly(frame, rectangle=True, crop=False, show=False)
                    rnd = random.uniform(0.4, 0.8)
                    self.image_health = np.array([rnd], dtype=np.float32)
                frame_disp = frame[..., ::-1].copy()
                ir = self.display.imageNew(frame_disp.tobytes(), Display.RGB, 480, 270)
                self.display.imagePaste(ir, 0, 0, False)
                self.display.imageDelete(ir)
                if 'image' in self.o_space:
                    self.obs['image'] = frame

        if 'lidar' in self.o_space:
            # Lidar data is an array, 0 is behind the robot incrementing clockwise. Front of robot is Lidarpoints/2
            # Grab Lidar Data
            lidarData = self.lidar.getRangeImage()
            lidarData = np.asarray(lidarData)
            # Normalize lidar data
            lidarData[lidarData == inf] = 12.5
            lidarData = lidarData / 12.5
            lidarData = 1 - lidarData
            # LIDAR ANOMALY
            # --- LiDAR corruption ---
            if self.ano_mode in {"full_lidar", "full_both"}:
                lidarData[:] = 1.0  # complete failure (max "close" in your encoding)
                self.lidar_health = np.array([0.0], dtype=np.float32)
            elif self.ano_mode in {"partial_lidar", "partial_both"}:
                # Persistent partial LiDAR fault.
                # IMPORTANT: do not resample the sector on every get_obs() call.
                # The active-localization stage assumes a sensor-fixed corrupted
                # sector that remains stable while the robot rotates. The region is
                # sampled once at reset through self.startingpoint/self.leng, or is
                # explicitly forced by Knowledge through set_detected_anomaly().
                descriptor = self.forced_lidar_region if self.forced_lidar_region is not None else self.forced_lidar_obstruction
                if descriptor is not None:
                    start = int(descriptor["start"])
                    width = int(descriptor["width"])
                    if "value" in descriptor:
                        blocked_value = float(descriptor["value"])
                    else:
                        distance_m = float(descriptor.get("distance_m", 0.05))
                        blocked_value = 1.0 - distance_m / 12.5
                    noise_std = float(descriptor.get("noise_std", 0.0))
                else:
                    start = int(self.startingpoint)
                    width = int(self.leng)
                    blocked_value = 1.0
                    noise_std = 0.0

                n_lidar = len(lidarData)
                start = start % n_lidar
                width = max(1, min(width, n_lidar))
                self.startingpoint = start
                self.leng = width

                rnd = random.uniform(0.4, 0.8)
                self.lidar_health = np.array([rnd], dtype=np.float32)
                lidarData = lidarData.copy()
                indices = (start + np.arange(width)) % n_lidar
                if noise_std > 0.0:
                    vals = np.clip(blocked_value + np.random.normal(0.0, noise_std, size=width), 0.0, 1.0)
                else:
                    vals = np.full(width, blocked_value, dtype=np.float64)
                lidarData[indices] = vals

                # # mild noise everywhere else
                # lidarData += np.random.normal(0.0, 0.02, size=lidarData.shape)
                # lidarData = np.clip(lidarData, 0.0, 1.0)
            # Apply Circular Padding
            lidarData_left = lidarData[:self.padding]
            lidarData_right = lidarData[-self.padding:]
            lidarData = np.concatenate((lidarData, lidarData_left), axis=0)
            lidarData = np.concatenate((lidarData_right, lidarData), axis=0)
            # fill 1st framestacked observation
            if self.just_reset:
                self.lidar_obs = []
                for i in range(self.framestack):
                    self.lidar_obs.append(lidarData)
            else:
                self.lidar_obs = self.lidar_obs[1:]
                self.lidar_obs.append(lidarData)
            lidar_obs = np.asarray(self.lidar_obs)
            # print('LIDAROBS ', lidar_obs.shape)
            # Changed here
            # lidar_obs = lidar_obs.flatten()
            # lidar_obs = lidarData
            self.obs['lidar'] = lidar_obs

        if 'world_info' in self.o_space:
            worldData = [1.0 - round(self.distance_to_target / 29, 6),  # + np.random.normal(0.0, 1, 1) ,
                         round(self.angle / np.pi, 6),  # + np.random.normal(0.0, 1, 1),
                         round(1 - (self.angle_pi / 180), 6)]
            if self.ano_mode in {"full_lidar", "full_both"}:
                worldData = [0.0, 0.0, 0.0]  # compact state unavailable under complete LiDAR failure
            elif self.ano_mode in {"partial_lidar", "partial_both"}:
                worldData = np.asarray(worldData)
                worldData += np.random.normal(0.0, 0.02, size=worldData.shape)
                worldData = np.clip(worldData, 0.0, 1.0)

            world_obs = np.asarray(worldData)
            self.obs['world_info'] = world_obs
        self.just_reset = False

        if 'image_health' in self.o_space:
            self.obs['image_health'] = self.image_health

        if 'lidar_health' in self.o_space:
            self.obs['lidar_health'] = self.lidar_health

        if self.o_space == ['image']:
            return frame

        return self.obs

    def apply_camera_anomaly(self, frame, rectangle=True, crop=False, show=False):
        frame = frame.astype(np.uint8).copy()
        if self.forced_image_mask is not None:
            mask = np.asarray(self.forced_image_mask, dtype=bool)
            if mask.shape != frame.shape[:2]:
                raise ValueError(f"Image anomaly mask shape {mask.shape} does not match frame {frame.shape[:2]}")
            frame[mask] = np.asarray(self.ano_col, dtype=np.uint8)
        elif rectangle:
            cv2.rectangle(frame,
                          (self.ano_x, self.ano_y), (self.ano_x + self.ano_width, self.ano_y + self.ano_height),
                          self.ano_col,
                          -1)
        if crop:
            height = int(270 * 0.5625)
            crop_img = frame[int(self.camera.getHeight() / 2) - int(height / 2):int(self.camera.getHeight() / 2)
                                                                                + int(height / 2), 201:]
            frame = cv2.resize(crop_img, (480, 270), interpolation=cv2.INTER_AREA)
        if show:
            cv2.imshow('Anomaly Input', frame)
            cv2.waitKey(1)
        return frame

    def get_reward(self):
        if self.prev_angle != self.angle_pi:
            self.prev_angle = self.angle_pi
        if self.prev_distance != self.distance:
            self.prev_distance = self.distance
        reward = 0
        robot_pos = np.array(self.position.getSFVec3f())  # robot position
        human_pos = np.array(self.human_position.getSFVec3f())
        human_pos[2] = 0.0004
        target_vector = np.array(robot_pos - human_pos)
        self.distance_to_target = np.linalg.norm(target_vector)
        # distance to next target
        robot_next_target = np.array(robot_pos - self.next_target)
        distance = np.linalg.norm(target_vector)
        robot_ntarget_vector = robot_next_target / distance  # unit vector from drone to next target

        # drone orientation
        ori_drone = np.array(self.robot.getOrientation()).reshape([3, 3])

        # angles between vector drone-next_target and drone-forward_direction_of_drone
        b = np.array([1, 0, 0])
        new_p = ori_drone.dot(b) + robot_pos
        drone_forward = robot_pos - new_p  # unit vector drone-forward_direction

        dot_ntarget_drone = drone_forward.dot(robot_ntarget_vector)
        dot_ntarget_drone = np.clip(dot_ntarget_drone, -1.0, 1.0)
        cross_ntarget_drone = np.cross(drone_forward, robot_ntarget_vector)
        angle = np.arctan2(cross_ntarget_drone, dot_ntarget_drone)[2]
        angle_pi = np.arccos(dot_ntarget_drone)
        distance = round(distance, 2)
        angle_pi = round(angle_pi * (180 / np.pi))
        # compute individual rewards
        d_reward = self.get_d_reward(distance)
        reward += d_reward  # + self.get_a_reward(angle)
        # print(angle_pi) # maybe here add angle reward only when close

        a_reward = self.get_a_reward(angle_pi, distance)
        if distance < 3.0:
            reward += a_reward
        # lidar_reward = self.get_lidar_reward()
        # reward += lidar_reward
        # if lidar_reward != 0:
        #     print(d_reward, a_reward, lidar_reward)
        self.angle_pi = angle_pi
        self.distance = distance
        self.angle = angle
        return reward

    def get_a_reward(self, angle, distance):
        if self.a_space == 'discrete':
            # a_reward = 2 * (self.prev_angle - angle)
            a_reward = (self.prev_angle - angle) / 10
        if self.a_space == 'continuous':
            a_reward = 2 * round((abs(round(self.prev_angle, 1)) - abs(round(angle, 1))), 2)

        if abs(angle) < self.min_angle:
            self.min_angle = abs(angle)
        # if distance > 0.5:
        #     self.a_reward = round(a_reward /10 , 1)
        # else:
        #     self.a_reward = 0.0
        # self.supervisor.setLabel(1, 'Angle Reward: ' + str(self.a_reward), 0,0,0.1,0xff0000,0,"Arial")
        self.a_reward = round(a_reward, 2)
        return a_reward

    def get_d_reward(self, distance):
        if self.a_space == 'discrete':
            if distance > self.target_distance:
                d_reward = self.prev_distance - distance
            else:
                d_reward = distance - self.prev_distance
            # if distance < self.min_distance:
            #     d_reward = self.min_distance - distance
            # else:
            #     d_reward = 0
        if self.a_space == 'continuous':
            d_reward = 5 * (round(self.prev_distance, 4) - round(distance, 4))
        # self.prev_distance = distance
        if distance < self.min_distance:
            self.min_distance = distance
        self.d_reward = round(10 * d_reward, 1)
        return round(10 * d_reward, 1)

    def get_lidar_reward(self):
        lidarData = self.lidar.getRangeImage()
        lidarData = np.asarray(lidarData)
        # Normalize lidar data
        lidarData[lidarData == inf] = 12.5
        lidarData = lidarData / 12.5
        lidarData = 1 - lidarData  # 1 is close,0 far
        # if self.prev_lidarData is None:
        #     self.prev_lidarData = lidarData
        mask = (lidarData > 0.96)  # | (self.prev_lidarData > 0.96)
        prev_mask = self.prev_lidarData > 0.96 if self.prev_lidarData is not None else mask
        # diff = lidarData[mask] - self.prev_lidarData[mask]
        # mean = round(diff.mean(), 3) if diff.size > 0 else 0.0
        if mask.sum() > 100 and mask.sum() - prev_mask.sum() > 0:
            lidar_reward = -0.01
        else:
            lidar_reward = 0.0
        if self.lidar_anomaly:
            lidar_reward = 0.0
        self.prev_lidarData = lidarData
        return lidar_reward

    def get_efficient_reward(self, angle, distance):
        turn_penalty = 0.1 * (abs(angle - self.prev_angle))
        # distance_penalty = 0.1 * (abs(distance - self.prev_distance))
        efficency_reward = turn_penalty  # + distance_penalty
        return efficency_reward

    def define_target(self):
        self.next_target = np.array(self.human_position.getSFVec3f())
        self.next_target[2] = 0.0004
        self.target = self.next_target
        self.supervisor.step(self.timestep)

    def set_velocity(self, v_left, v_right):
        """
        Sets the two motor velocities.
        :param v_left: velocity value for left motor
        :type v_left: float
        :param v_right: velocity value for right motor
        :type v_right: float
        """
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(v_left * 6.0)
        self.right_motor.setVelocity(v_right * 6.0)

    def filter_persons(self, frame):
        """
        Filters the class person in yolo
        """
        filtered_results = self.filtered.infer(frame)
        return filtered_results

    def bbox_to_np(self, bbox):
        bbox_np = np.asarray(
            [bbox.left, bbox.top, bbox.left + bbox.width, bbox.top + bbox.height, bbox.confidence, bbox.name],
            dtype=object)
        return bbox_np

    def deploy_dl(self, frame):
        results = self.person_detector.predict(frame, quantize=16, stream=True, classes=[0],  # class[0] = person
                                               verbose=False)  # return a list of Results objects

        for result in results:
            boxes = result.boxes  # Boxes object for bounding box outputs
            if boxes:
                for box in boxes:
                    if box.conf > 0.85:
                        self.found = True
                        break

    def set_eval(self):
        self.eval = True

    def unset_eval(self):
        self.eval = False

    def replace_humans(self):
        # if self.human is not None:
        #     self.human.remove()
        rnd = random.randint(1, 2)
        self.supervisor.step(self.timestep)
        if rnd < 10:
            human_model = 'human_0' + str(rnd) + '_standing'
            self.children.importMFNodeFromString(2, 'DEF ' + human_model + ' ' + human_model + ' {}')
            self.human = self.supervisor.getFromDef(human_model)
        else:
            human_model = 'human_' + str(rnd) + '_standing'
            self.children.importMFNodeFromString(2, 'DEF ' + human_model + ' ' + human_model + ' {}')
            self.human = self.supervisor.getFromDef(human_model)
        self.supervisor.step(self.timestep)
        room = self.get_random_room()
        x = random.uniform(room[0] - 2.5, room[0] + 2.5)
        y = random.uniform(room[1] - 2.5, room[1] + 2.5)
        self.human_position = self.human.getField('translation')
        self.human_position.setSFVec3f([x, y, 0.0004])
        rotation = self.human.getField('rotation')
        rotation.setSFRotation([0, 0, 1, random.randint(0, 50) * 0.1309])

    def get_random_room(self):
        room = random.choice(self.rooms)
        self.rooms.remove(room)
        return room

    def initialize_world(self):
        self.supervisor.step(self.timestep)
        grps = ['lr1', 'lr2', 'lr3', 'plants1', 'plants2', 'plants3', 'plants4']
        sides = ['paintings_1', 'paintings_2']
        fronts = ['billboard', 'door']
        ########################################
        # Change side objects position #########
        ########################################
        for item in grps:
            room = self.get_random_room()
            item = self.supervisor.getFromDef(item)
            position = item.getField('translation')
            position.setSFVec3f([room[0], room[1], 0])
        for item in sides:
            rnd = random.choice([-6, 0, 6])
            item = self.supervisor.getFromDef(item)
            position = item.getField('translation')
            position.setSFVec3f([rnd, 0, 0])
        for item in fronts:
            rnd = random.choice([-6, 0, 6])
            item = self.supervisor.getFromDef(item)
            position = item.getField('translation')
            position.setSFVec3f([0, rnd, 0])
        self.supervisor.step(self.timestep)


    def _apply_forced_anomaly(self):
        """Apply the anomaly stored in Knowledge to the simulator state."""
        self.ano_mode = self.forced_anomaly_mode
        self.image_anomaly = self.ano_mode in {"partial_image", "full_image", "partial_both", "full_both"}
        self.lidar_anomaly = self.ano_mode in {"partial_lidar", "full_lidar", "partial_both", "full_both"}

        if self.forced_image_region is not None:
            self.ano_x = int(self.forced_image_region["x"])
            self.ano_y = int(self.forced_image_region["y"])
            self.ano_width = int(self.forced_image_region["width"])
            self.ano_height = int(self.forced_image_region["height"])
            self.ano_col = tuple(self.forced_image_region.get("color", self.ano_col))
        elif self.forced_image_spec is not None and "color" in self.forced_image_spec:
            self.ano_col = tuple(int(c) for c in self.forced_image_spec["color"])

        if self.forced_lidar_region is not None:
            self.startingpoint = int(self.forced_lidar_region["start"])
            self.leng = int(self.forced_lidar_region["width"])

    def set_detected_anomaly(self, sensor_name, region, anomaly_mode=None):
        """Store Analyze-phase anomaly output in Knowledge and force it in future resets.

        sensor_name: "image" or "lidar".
        region for image: {x, y, width, height, optional color}.
        region for lidar: {start, width}.
        anomaly_mode defaults to partial_<sensor_name>.
        """
        if anomaly_mode is None:
            anomaly_mode = f"partial_{sensor_name}"

        self.forced_anomaly_mode = anomaly_mode
        self.forced_image_mask = None
        self.forced_image_spec = None
        self.forced_lidar_obstruction = None
        if sensor_name == "image":
            self.forced_image_region = dict(region)
            self.forced_lidar_region = None
        elif sensor_name == "lidar":
            self.forced_lidar_region = dict(region)
            self.forced_image_region = None
        else:
            raise ValueError(f"Unknown sensor_name: {sensor_name}")
        self._apply_forced_anomaly()

    def set_experiment_anomaly(self, mode, image_mask=None, image_spec=None, lidar_spec=None):
        """Force persistent ground-truth faults for detector data collection."""
        valid = {"normal", "partial_image", "partial_lidar", "full_image", "full_lidar", "partial_both", "full_both"}
        if mode not in valid:
            raise ValueError(f"Unsupported experiment anomaly mode: {mode}")
        self.forced_anomaly_mode = None if mode == "normal" else mode
        self.forced_image_region = None
        self.forced_lidar_region = None
        self.forced_image_mask = None if image_mask is None else np.asarray(image_mask, dtype=bool).copy()
        self.forced_image_spec = None if image_spec is None else dict(image_spec)
        self.forced_lidar_obstruction = None if lidar_spec is None else dict(lidar_spec)
        self.ano_mode = mode
        if image_spec is not None and "color" in image_spec:
            self.ano_col = tuple(int(c) for c in image_spec["color"])
        if mode != "normal":
            self._apply_forced_anomaly()
        else:
            self.image_anomaly = False
            self.lidar_anomaly = False

    def clear_experiment_anomaly(self):
        self.forced_anomaly_mode = None
        self.forced_image_region = None
        self.forced_lidar_region = None
        self.forced_image_mask = None
        self.forced_image_spec = None
        self.forced_lidar_obstruction = None
        self.ano_mode = "normal"
        self.image_anomaly = False
        self.lidar_anomaly = False

    def clear_detected_anomaly(self):
        """Clear Knowledge's forced anomaly entry and return to environment sampling."""
        self.forced_anomaly_mode = None
        self.forced_image_region = None
        self.forced_lidar_region = None
        self.forced_image_mask = None
        self.forced_image_spec = None
        self.forced_lidar_obstruction = None
        self.ano_mode = "normal"
        self.image_anomaly = False
        self.lidar_anomaly = False

    def _curriculum_scale(self) -> float:
        """0 at start of training, -> 1 after ramp."""
        if self.ano_ramp_steps <= 0:
            return 1.0
        return min(1.0, self.total_env_steps / float(self.ano_ramp_steps))

    def _sample_anomaly_mode(self):
        """
        Decide anomaly mode for the new episode using a curriculum schedule.

        Final mixture after ramp:
          - normal:  1 - (p_partial + p_full)
          - partial: p_partial
        """
        if self.inference_mode:
            self.p_partial_max = 0.4
            self.p_full_max = 0.5
            self.ano_ramp_steps = 1

        s = self._curriculum_scale()

        p_partial = s * self.p_partial_max
        p_full = s * self.p_full_max

        # Normalize/clip in case someone sets crazy values
        p_partial = max(0.0, min(1.0, p_partial))
        p_full = max(0.0, min(1.0, p_full))

        r = random.random()

        # Default flags
        self.image_anomaly = False
        self.lidar_anomaly = False
        self.ano_mode = "normal"

        # Decide: full vs partial vs none
        if r < p_full:
            # full failure
            if random.random() < self.p_img_vs_lidar:
                self.image_anomaly = True
                self.ano_mode = "full_image"
            else:
                self.lidar_anomaly = True
                self.ano_mode = "full_lidar"
        elif r < (p_full + p_partial):
            # partial corruption (we'll implement these in get_obs)
            if random.random() < self.p_img_vs_lidar:
                self.ano_mode = "partial_image"
            else:
                self.ano_mode = "partial_lidar"
        else:
            # normal
            pass

    def set_gate_debug(self, gates: dict):
        self.gate_debug = gates
