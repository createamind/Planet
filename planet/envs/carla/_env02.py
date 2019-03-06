#!/usr/bin/env python

import glob
import os
import sys
import re
import weakref
try:
    sys.path.append('/data/carla94/PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg')
except IndexError:
    pass

import carla
import pygame
import random
import time
import subprocess
from carla import ColorConverter as cc
import math
import matplotlib.pyplot as plt
import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple
# Default environment configuration
ENV_CONFIG = {
    "framestack": 1,
    "reward_function": "custom",
    #"render_x_res": 128,
    #"render_y_res": 128,
    "x_res": 96,
    "y_res": 96,
    "discrete_actions": True,
}

class CarlaEnv(gym.Env):
    def __init__(self, config=ENV_CONFIG):
        self.config = config
        self.command = {
            "stop": 1,
            "lane_keep": 2,
            "turn_right": 3,
            "turn_left": 4,
        }
        self.DISCRETE_ACTIONS = {
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
 

        if config["discrete_actions"]:
            self.action_space = Discrete(len(self.command))
        else:
            self.action_space = Box(-1.0, 1.0, shape=(2, ), dtype=np.float32)

        image_space = Box(
            0,
            255,
            shape=(config["y_res"], config["x_res"],
                   3 * config["framestack"]),
            dtype=np.uint8)
        self.observation_space = image_space
        # environment config
        self._spec = lambda: None
        self._spec.id = "Carla_v0"
        # experiment config
        self.num_steps = 0
        self.total_reward = 0
        self.episode_id = None
        self.measurements_file = None
        self.weather = None
        # actors
        self.actor_list = []          # save actor list for destroying them after finish
        self.vehicle = None
        self.collision_sensor = None
        self.camera_rgb = None
        self.invasion_sensor = None
        # states and data
        self._history_info = []       # info history
        self._history_collision = []  # collision history
        self._history_invasion = []   # invasion history
        self._image_depth = []        # save a list of depth image
        self._image_rgb = []          # save a list of rgb image
        self.image = []
        # server
        self.server_port = 2000
        try:
            self.client = carla.Client("localhost", self.server_port)
            self.client.set_timeout(7.0)
        except Exception as e:
            print("fail to connect simulator", e)
            self.client = carla.Client("localhost", self.server_port)
            self.client.set_timeout(10.0)
            
        self.world = self.client.get_world()
        self.map = self.world.get_map()

    def restart(self):
        world = self.world
        bp_library = world.get_blueprint_library()

        # setup vehicle
        spawn_point = random.choice(world.get_map().get_spawn_points())
        bp_vehicle = bp_library.find('vehicle.lincoln.mkz2017')
        bp_vehicle.set_attribute('role_name', 'hero')
        self.vehicle = world.spawn_actor(bp_vehicle, spawn_point)
        self.actor_list.append(self.vehicle)

        # setup camera
        camera_transform = carla.Transform(carla.Location(x=0, z=2.4))
        bp_rgb = bp_library.find('sensor.camera.rgb')
        self.camera_rgb = world.spawn_actor(bp_rgb, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera_rgb)

        # add collision sensors
        bp = bp_library.find('sensor.other.collision')
        self.collision_sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.collision_sensor)

        # add invasion sensors
        bp = bp_library.find('sensor.other.lane_detector')
        self.invasion_sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.actor_list.append(self.invasion_sensor)


    def reset(self):
        self.restart()
        weak_self = weakref.ref(self)
        # set invasion sensor
        self.invasion_sensor.listen(lambda event: self._parse_invasion(weak_self, event))
        # set collision sensor
        self.collision_sensor.listen(lambda event: self._parse_collision(weak_self, event))
        # set rgb camera sensor
        self.camera_rgb.listen(lambda image: self._parse_image(weak_self, image, carla.ColorConverter.Raw))
   #      time.sleep(0.1)
        while len(self._image_rgb)<1:
            time.sleep(0.1)
        return self._image_rgb[-1]

    @staticmethod
    def _parse_image(weak_self, image, cc):
        self = weak_self()
        if not self:
            return
        image.convert(cc)
        # image.save_to_disk('_out/%08d' % image.frame_number)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, -2:-5:-1]
        self._image_rgb.append(array)
        # time.sleep(1)
        self.image.append(image)
        # print(1)

    @staticmethod
    def _parse_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._history_collision.append((event.frame_number, intensity))
       # if len(self._history_collision) > 1600:
        #    self._history_collision.pop(0)

    @staticmethod
    def _parse_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        # print(str(event.crossed_lane_markings)) [carla.libcarla.LaneMarking.Solid]
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        # S for Solid B for Broken
        self._history_invasion.append(text[0][1])
      #  if len(self._history_invasion) > 16:
       #     self._history_collision.pop(0)

    def step(self, action):

        def compute_reward(info, prev_info):
            reward = 0.0
            reward += np.clip(info["speed"], 0, 30)/6
            reward -= 100 * int(len(self._history_collision) > 0)
            new_invasion = list(set(info["lane_invasion"]) - set(prev_info["lane_invasion"]))
            if 'S' in new_invasion:     # go across solid lane
                reward -= 20
            elif 'B' in new_invasion:   # go across broken lane
                reward -= 5
            return reward

        done = False
        if self.config["discrete_actions"]:
            action = self.DISCRETE_ACTIONS[int(action)]

        throttle = float(np.clip(action[0], 0, 1))
        brake = float(np.abs(np.clip(action[0], -1, 0)))
        steer = float(np.clip(action[1], -1, 1))
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        # get image
        time.sleep(0.1)

        t = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        c = self.vehicle.get_control()
        acceleration = self.vehicle.get_acceleration()
      
        if len(self._history_invasion) > 0:
            invasion = self._history_invasion[-1]
        else:
            invasion = []
        info = {"speed": math.sqrt(v.x**2 + v.y**2 + v.z**2),  # m/s
                "acceleration": math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2),
                "location_x": t.location.x,
                "location_y": t.location.y,
                "Throttle": c.throttle,
                "Steer": c.steer,
                "Brake": c.brake,
                "command": self.planner(),
                "lane_invasion": invasion,
                "traffic_light": str(self.vehicle.get_traffic_light_state()),    # Red Yellow Green Off Unknown
                "is_at_traffic_light": self.vehicle.is_at_traffic_light(),
                "speed_limit": self.vehicle.get_speed_limit()}       # True False

        if len(self._history_info) == 0:
            self._history_info.append(info)
        reward = compute_reward(info, self._history_info[-1])
        self._history_info.append(info)
        if len(self._history_info) > 16:
            self._history_info.pop(0)
        # early stop
        if len(self._history_collision) > 0:
            # print("collisin length", len(self._history_collision))
            done = True
        elif reward < -100:
            done = True
        done = False
        return self._image_rgb[-1], reward, done, self._history_info[-1]

    def render(self):
        import pygame
        display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        surface = pygame.surfarray.make_surface(env._image_rgb[-1].swapaxes(0, 1))
        display.blit(surface, (0, 0))
        time.sleep(0.01)
        pygame.display.flip()

    def planner(self):
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        waypoint = random.choice(waypoint.next(12.0))
        yaw = waypoint.transform.rotation.yaw
        if yaw > -90 or yaw < 60:
            command = "turn_right"
        elif yaw > 60 and yaw < 120:
            command = "lane_keep"
        elif yaw > 120 or yaw < -90:
            command = "turn_left"
        return self.command[command]


if __name__ == '__main__':
    env = CarlaEnv()
    obs = env.reset()
    print(obs.shape)
    done = False
    while not done:
        # env.render()
        obs, reward, done, info = env.step(3)
        # print(len(env._image_rgb), obs.shape)
        print(obs.shape)

    for actor in env.actor_list:
        actor.destroy()
