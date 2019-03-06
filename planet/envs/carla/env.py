"""OpenAI gym environment for Carla. Run this file for a demo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime as dt
import atexit
import cv2
import os
import json
import random
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
try:
    import scipy.misc
except Exception:
    pass

import gym
from gym.spaces import Box, Discrete, Tuple

from scenarios import DEFAULT_SCENARIO

# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get("CARLA_SERVER",
                               os.path.expanduser("~/Documents/carla9/CarlaUE4.sh"))

assert os.path.exists(SERVER_BINARY)
if "CARLA_PY_PATH" in os.environ:
    sys.path.append(os.path.expanduser(os.environ["CARLA_PY_PATH"]))
else:
    sys.path.append(os.path.expanduser("~/carla9/PythonAPI/"))

import carla
from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref


# Number of retries if the server doesn't respond
RETRIES_ON_ERROR = 5

# Dummy Z coordinate to use when we only care about (x, y)
GROUND_Z = 22

# Default environment configuration
ENV_CONFIG = {
    "log_images": True,
    "enable_planner": True,
    "framestack": 2,  # note: only [1, 2] currently supported
    "convert_images_to_video": True,
    "early_terminate_on_collision": True,
    "verbose": True,
    "reward_function": "custom",
    "render_x_res": 800,
    "render_y_res": 600,
    "x_res": 800,
    "y_res": 800,
    "server_map": "/Game/Maps/Town02",
    "scenarios": [DEFAULT_SCENARIO],
    "use_depth_camera": False,
    "discrete_actions": True,
    "squash_action_logits": False,
}

DISCRETE_ACTIONS = {
    # coast
    0: [0.0, 0.0],
    # turn left
    1: [0.0, -0.5],
    # turn right
    2: [0.0, 0.5],
    # forward
    3: [1.0, 0.0],
    # brake
    4: [-0.5, 0.0],
    # forward left
    5: [1.0, -0.5],
    # forward right
    6: [1.0, 0.5],
    # brake left
    7: [-0.5, -0.5],
    # brake right
    8: [-0.5, 0.5],
}

live_carla_processes = set()


# Mapping from string repr to one-hot encoding index to feed to the model
# Some command we want give to agent
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "STOP": 1,
    "LANE_KEEP": 2,
    "TURN_RIGHT": 3,
    "TURN_LEFT": 4,
    "SURPASS": 5
}





class CarlaEnv(gym.Env):
    def __init__(self, config=ENV_CONFIG):
        self.config = config
        self.city = self.config["server_map"].split("/")[-1]
        if self.config["enable_planner"]:
            pass

        if config["discrete_actions"]:
            self.action_space = Discrete(len(DISCRETE_ACTIONS))
        else:
            self.action_space = Box(-1.0, 1.0, shape=(2, ), dtype=np.float32)
        if config["use_depth_camera"]:
            image_space = Box(
                -1.0,
                1.0,
                shape=(config["y_res"], config["x_res"],
                       1 * config["framestack"]),
                dtype=np.float32)
        else:
            image_space = Box(
                0,
                255,
                shape=(config["y_res"], config["x_res"],
                       3 * config["framestack"]),
                dtype=np.uint8)
        self.observation_space = Tuple(  # forward_speed, dist to goal
            [
                image_space,
                Discrete(len(COMMAND_ORDINAL)),  # next_command
                Box(-128.0, 128.0, shape=(2, ), dtype=np.float32)
            ])

        self._spec = lambda: None
        self._spec.id = "Carla_v0"

        self.server_port = 2002
        self.server_process = None
        # self.client = None
        self.num_steps = 0
        self.actor_list = []  # save actor list for destroying them after finish
        self.total_reward = 0
        self.episode_id = None
        self.measurements_file = None
        self.weather = None
        self.scenario = None
        self.start_pos = None
        self.end_pos = None
        self.start_coord = None
        self.end_coord = None
        self.last_obs = None
        self.vehicle = None
        self._history_info = []       # info history
        self._history_collision = []  # collision history
        self._history_invasion = []   # invasion history
        self._image_depth = []        # save a list of depth image
        self._image_rgb = []          # save a list of rgb image
        self.client = carla.Client("localhost", self.server_port)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        # all kinds of sensors we need
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            # ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            # ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            # ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            # ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
            # 'Camera Semantic Segmentation (CityScapes Palette)'],
            # ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)'],
            ['sensor.other.collision', None, 'collision detector'],
            ['sensor.other.lane_detector', None, 'lane invasion detector']]

    def reset(self):
        try:
            return self._reset()
        except Exception as e:
            print(e)

    def _reset(self):
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = dt.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None

        # setup world and get blueprint
        world = self.world
        bp_library = world.get_blueprint_library()
        camera_transform = carla.Transform(carla.Location(x=0.3, z=1.7))

        # add vehicle
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[1]
        bp_vehicle = world.get_blueprint_library().find('vehicle.lincoln.mkz2017')
        bp_vehicle.set_attribute('role_name', 'hero')
        self.vehicle = world.spawn_actor(bp_vehicle, spawn_point)
        self.actor_list.append(self.vehicle)

        # add rgb camera
        bp_rgb = bp_library.find(self._sensors[0][0])
        bp_rgb.set_attribute('image_size_x', str(ENV_CONFIG["x_res"]))
        bp_rgb.set_attribute('image_size_y', str(ENV_CONFIG["y_res"]))
        weak_self = weakref.ref(self)
        camera_rgb = world.spawn_actor(bp_rgb, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(camera_rgb)

        # add depth camera
        bp_depth = bp_library.find(self._sensors[1][0])
        bp_depth.set_attribute('image_size_x', str(ENV_CONFIG["x_res"]))
        bp_depth.set_attribute('image_size_y', str(ENV_CONFIG["y_res"]))
        weak_self = weakref.ref(self)
        camera_depth = world.spawn_actor(bp_depth, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(camera_depth)


        # add collision sensors
        bp = world.get_blueprint_library().find('sensor.other.collision')
        collision_sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        collision_sensor.listen(lambda event: CarlaEnv._parse_collision(weak_self, event))
        self.actor_list.append(collision_sensor)

        # add lane invasion sensors
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        invasion_sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        weak_self = weakref.ref(self)
        invasion_sensor.listen(lambda event: CarlaEnv._parse_invasion(weak_self, event))
        self.actor_list.append(invasion_sensor)

        # get other measurement
        t = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        c = self.vehicle.get_vehicle_control()

        info = {"speed": math.sqrt(v.x**2 + v.y**2 + v.z**2),  # m/s
                     "location_x": t.location.x,
                     "location_y": t.location.y,
                     "Throttle": c.throttle,
                     "Steer": c.steer,
                     "Brake": c.brake}

        self._history_info.append(info)
        if len(self._history_info) > 16:
            self._history_info.pop(0)

        def parse_image(image, convert):
            image.convert(convert)
            image.save_to_disk('_out/%08d' % image.frame_number)
                # if not self:
                #     return
                # image.convert(convert)
                # array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
                # array = np.reshape(array, (image.height, image.width, 4))
                # array = array[:, :, -2:-5:-1]
                # self._image_depth.append(array)
                # if len(self._image_depth) > 16:
                #     self._image_depth.pop(0)

        camera_rgb.listen(lambda image: parse_image(image, cc.Raw))
        camera_depth.listen(lambda image: parse_image(image, cc.LogarithmicDepth))
        time.sleep(1)
        return self._image_rgb[-1]

    @staticmethod
    def _parse_image_depth(weak_self, image, convert):
        image.convert(convert)
        self = weak_self()
        image.save_to_disk('_out/%08d' % image.frame_number)
        if not self:
            return
        image.convert(convert)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, -2:-5:-1]
        self._image_depth.append(array)
        if len(self._image_depth) > 16:
            self._image_depth.pop(0)

    @staticmethod
    def _parse_image_rgb(weak_self, image, convert):
        image.save_to_disk('_out/%08d' % image.frame_number)
        self = weak_self()
        if not self:
            return
        image.convert(convert)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, -2:-5:-1]
        self._image_rgb.append(array)
        if len(self._image_rgb) > 16:
            self._image_rgb.pop(0)

    @staticmethod
    def _parse_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._history_collision.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history_collision.pop(0)

    @staticmethod
    def _parse_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        self._history.append(text)
        if len(self._history_invasion) > 4000:
            self._history_invasion.pop(0)

    def step(self, action):
        try:
            image, reward, done, info = self._step(action)
            return image, reward, done, info
        except Exception:
            print("Error during step, terminating episode early",
                  traceback.format_exc())
            return self._image_rgb[-1], 0.0, True, {}

    def _step(self, action):
        if self.config["discrete_actions"]:
            action = DISCRETE_ACTIONS[int(action)]

        throttle = float(np.clip(action[0], 0, 1))
        brake = float(np.abs(np.clip(action[0], -1, 0)))
        steer = float(np.clip(action[1], -1, 1))
        carla.VehicleControl(throttle=throttle,
                             steer=brake,
                             brake=steer,
                             hand_brake=False,
                             reverse=False,
                             manual_gear_shift=False,
                             gear=0)
        self.vehicle.apply_control(carla.VehicleControl)
        reward = compute_reward(self, self._history_info[-2], self._history_info[-1])

        return self._image_rgb[-1], reward, done, info

def compute_reward_custom(env, prev, current):
    reward = 0.0

    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    if env.config["verbose"]:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    # Distance travelled toward the goal in m
    reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)

    # Speed reward, up 30.0 (km/h)
    reward += np.clip(current["forward_speed"], 0.0, 30.0) / 10

    # New collision damage
    new_damage = (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])
    if new_damage:
        reward -= 100.0

    # Sidewalk intersection
    reward -= current["intersection_offroad"]

    # Opposite lane intersection
    reward -= current["intersection_otherlane"]

    # Reached goal
    if current["next_command"] == "REACH_GOAL":
        reward += 100.0

    return reward

REWARD_FUNCTIONS = {
    # "corl2017": compute_reward_corl2017,
    "custom": compute_reward_custom,
    # "lane_keep": compute_reward_lane_keep,
}


def compute_reward(env, prev, current):
    return REWARD_FUNCTIONS[env.config["reward_function"]](env, prev, current)

def sigmoid(x):
    x = float(x)
    return np.exp(x) / (1 + np.exp(x))


def collided_done(py_measurements):
    m = py_measurements
    collided = (m["collision_vehicles"] > 0 or m["collision_pedestrians"] > 0
                or m["collision_other"] > 0)
    return bool(collided or m["total_reward"] < -100)


if __name__ == "__main__":
   # for _ in range(2):
        env = CarlaEnv()
        obs = env.reset()
       # print(obs)
        # print(env.actor_list)
        # print("reset", obs)
        # start = time.time()
        # done = False
        # i = 0
        # total_reward = 0.0
        # while not done:
        #     i += 1
        #     if ENV_CONFIG["discrete_actions"]:
        #         obs, reward, done, info = env.step(1)
        #     else:
        #         obs, reward, done, info = env.step([0, 1, 0])
        #     total_reward += reward
        #     print(i, "rew", reward, "total", total_reward, "done", done)
        # print("{} fps".format(100 / (time.time() - start)))
