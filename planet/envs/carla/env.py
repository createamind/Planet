#!/usr/bin/env python

import glob
import os
import sys
import re
import weakref
import carla
import pygame
import random
import time
import subprocess
from carla import ColorConverter as cc
import math
# import matplotlib.pyplot as plt
import numpy as np
import gym
from gym.spaces import Box, Discrete, Tuple
from scipy.stats import multivariate_normal

# Default environment configuration
""" default is rgb 
    stack for gray depth segmentation stack together
    encode for encode measurement in forth channel """


# 2000 soft 3000 hard
ENV_CONFIG = {
    "x_res": 96,
    "y_res": 96,
    "port": 2000,
    "image_mode": "encode",
    "localhost": "192.168.100.37",
    "early_stop": False,       # if we use planet this has to be False
    "attention_mode": "soft",  # hard for dot product soft for adding noise None for regular
    "attention_channel": 3,    # int, the number of channel for we use attention mask on it, 3,6 is preferred
    "action_dim": 5,           # 4 for one point attention, 5 for control view field
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

        # change action space
        self.action_space = Box(-1.0, 1.0, shape=(ENV_CONFIG["action_dim"], ), dtype=np.float32)

        if ENV_CONFIG["image_mode"] == "encode":
            framestack = 7
        elif ENV_CONFIG["image_mode"] == "stack":
            framestack = 3
        else:
            framestack = 3

        image_space = Box(
            0,
            255,
            shape=(config["y_res"], config["x_res"], framestack),
            dtype=np.float32)
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
        self.feature_map = None
        # actors
        self.actor_list = []          # save actor list for destroying them after finish
        self.vehicle = None
        self.collision_sensor = None
        self.camera_rgb1 = None
        self.camera_rgb2 = None
        self.invasion_sensor = None
        # states and data
        self._history_info = []       # info history
        self._history_collision = []  # collision history
        self._history_invasion = []   # invasion history
        self._image_rgb1 = []          # save a list of rgb image
        self._image_rgb2 = []          # save a list of rgb image
        self._history_waypoint = []
        self._obs_collect = []
        # initialize our world
        self.server_port = ENV_CONFIG['port']
        self.world = None
        connect_fail_times = 0
        while self.world is None:
            try:
                self.client = carla.Client(ENV_CONFIG["localhost"], self.server_port)
                self.client.set_timeout(120.0)
                self.world = self.client.get_world()
                self.map = self.world.get_map()
            except Exception as e:
                connect_fail_times += 1
                print("Error connecting: {}, attempt {}".format(e, connect_fail_times))
                time.sleep(2)
            if connect_fail_times > 10:
                break

    def restart(self):
        """restart world and add sensors"""
        world = self.world
        # actors
        self.actor_list = []          # save actor list for destroying them after finish
        self.vehicle = None
        self.collision_sensor = None
        self.invasion_sensor = None
        # states and data
        self._history_info = []       # info history
        self._history_collision = []  # collision history
        self._history_invasion = []   # invasion history
        self._image_rgb1 = []         # save a list of rgb image
        self._image_rgb2 = []
        self._history_waypoint = []

        # destroy actors in the world before we start new episode
        for a in self.world.get_actors().filter('vehicle.*'):
            try:
                a.destroy()
            except:
                pass
        for a in self.world.get_actors().filter('sensor.*'):
            try:
                a.destroy()
            except:
                pass

        try:
            bp_library = world.get_blueprint_library()

            # setup vehicle
            spawn_point = random.choice(world.get_map().get_spawn_points())
            bp_vehicle = bp_library.find('vehicle.lincoln.mkz2017')
            bp_vehicle.set_attribute('role_name', 'hero')
            self.vehicle = world.try_spawn_actor(bp_vehicle, spawn_point)
            self.actor_list.append(self.vehicle)

            # setup rgb camera1
            camera_transform = carla.Transform(carla.Location(x=1, y=0, z=2))
            camera_rgb1 = bp_library.find('sensor.camera.rgb')
            camera_rgb1.set_attribute('fov', '120')
            camera_rgb1.set_attribute('image_size_x', str(ENV_CONFIG["x_res"]))
            camera_rgb1.set_attribute('image_size_y', str(ENV_CONFIG["y_res"]))
            self.camera_rgb1 = world.try_spawn_actor(camera_rgb1, camera_transform, attach_to=self.vehicle)
            self.actor_list.append(self.camera_rgb1)

            # add collision sensors
            bp = bp_library.find('sensor.other.collision')
            self.collision_sensor = world.try_spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            self.actor_list.append(self.collision_sensor)

            # add invasion sensors
            bp = bp_library.find('sensor.other.lane_detector')
            self.invasion_sensor = world.try_spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
            self.actor_list.append(self.invasion_sensor)
        except Exception as e:
            print("spawn fail, sad news", e)

    def reset(self):
        self.restart()
        weak_self = weakref.ref(self)
        # set invasion sensor
        self.invasion_sensor.listen(lambda event: self._parse_invasion(weak_self, event))
        # set collision sensor
        self.collision_sensor.listen(lambda event: self._parse_collision(weak_self, event))
        # set rgb camera sensor
        self.camera_rgb1.listen(lambda image: self._parse_image1(weak_self, image, cc.Raw, 'rgb'))
        while len(self._image_rgb1) < 4:
            print("resetting rgb")
            time.sleep(0.001)
        if ENV_CONFIG["image_mode"] == "encode":   # stack gray depth segmentation
            obs = np.concatenate([self._image_rgb1[-1], self._image_rgb1[-2],
                                  np.zeros([ENV_CONFIG['x_res'], ENV_CONFIG['y_res'], 1])], axis=2)
        else:
            obs = self._image_rgb1[-1]

        t = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        c = self.vehicle.get_control()
        acceleration = self.vehicle.get_acceleration()
        if len(self._history_invasion) > 0:
            invasion = self._history_invasion[-1]
        else:
            invasion = []
        self.planner()
        distance = ((self._history_waypoint[-1].transform.location.x - self.vehicle.get_location().x)**2 + 
                   (self._history_waypoint[-1].transform.location.y - self.vehicle.get_location().y)**2)**0.5

        info = {"speed": math.sqrt(v.x**2 + v.y**2 + v.z**2),  # m/s
                "acceleration": math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2),
                "location_x": t.location.x,
                "location_y": t.location.y,
                "Throttle": c.throttle,
                "Steer": c.steer,
                "Brake": c.brake,
                "command": self.planner(),
                "distance": distance,
                "lane_invasion": invasion,
                "traffic_light": str(self.vehicle.get_traffic_light_state()),    # Red Yellow Green Off Unknown
                "is_at_traffic_light": self.vehicle.is_at_traffic_light(),       # True False
                "collision": len(self._history_collision)
        }

        self._history_info.append(info)
        self._obs_collect.append(obs[:, :, 0:3])
        if len(self._obs_collect) > 32:
            self._obs_collect.pop(0)
        mask = self._compute_mask()
        # define how many channel we want play with
        if ENV_CONFIG["attention_mode"] == "soft":
            obs[:, :, 0:ENV_CONFIG["attention_channel"]] = obs[:, :, 0:ENV_CONFIG["attention_channel"]] + mask
        else:
            obs[:, :, 0:ENV_CONFIG["attention_channel"]] = obs[:, :, 0:ENV_CONFIG["attention_channel"]] * mask
        return np.clip(obs, 0, 255)

    @staticmethod
    def _compute_distance_transform(d):
        """compute the variance for attention mask when we adding noise
        if we specify attention mode to soft we will use this function """
        d = 0.006 * d**2 - 0.63
        # d = -24 + 2*d
        return np.maximum(6*d, 0)

    def _compute_mask(self, action=np.zeros(ENV_CONFIG["action_dim"])):
        """"compute mask for attention"""
        mu_1 = int(ENV_CONFIG["x_res"] * action[2] * 0.5)
        mu_2 = int(ENV_CONFIG["y_res"] * action[3] * 0.5)
        d_list = []
        point_list = self._generate_point_list()
        # TODO test different covariance
        if ENV_CONFIG["action_dim"] == 4:
            var = multivariate_normal(mean=[0, 0], cov=[[195, 0], [0, 195]])
        elif ENV_CONFIG["action_dim"] == 5:  # changing the view field by changing covariance
            var = multivariate_normal(mean=[0, 0], cov=[[400*(action[4]+1), 0], [0, 400*(action[4]+1)]])

        max_p = var.pdf([0, 0])
        for p in point_list:
            d = np.sqrt((mu_1 - p[0]) ** 2 + (mu_2 - p[1]) ** 2)
            if ENV_CONFIG["attention_mode"] == "soft":
                p_mask = float(self._compute_distance_transform(d) * np.random.randn(1))
            elif ENV_CONFIG["attention_mode"] == "hard":
                p_mask = (1.2 * (1/max_p)) * var.pdf([d, 0])
            else:  # if we want use raw rgb
                p_mask = 1
            d_list.append(p_mask)
        mask = np.reshape(d_list, [ENV_CONFIG["x_res"], ENV_CONFIG["y_res"]])
        return mask[:, :, np.newaxis]

    @staticmethod
    def _parse_image1(weak_self, image, cc, use):
        """convert BGRA to RGB"""
        self = weak_self()
        if not self:
            return

        def convert(cc):
            image.convert(cc)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, -2:-5:-1]
            array = array.astype(np.float32)
            return array

        if use == 'rgb':
            array = convert(cc)
            self._image_rgb1.append(array)
            if len(self._image_rgb1) > 32:
                self._image_rgb1.pop(0)
    @staticmethod
    def _parse_image2(weak_self, image, cc, use):
        """convert BGRA to RGB"""
        self = weak_self()
        if not self:
            return

        def convert(cc):
            image.convert(cc)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, -2:-5:-1]
            array = array.astype(np.float32)
            return array

        if use == 'rgb':
            array = convert(cc)
            self._image_rgb2.append(array)
            if len(self._image_rgb2) > 32:
                self._image_rgb2.pop(0)

    @staticmethod
    def _parse_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self._history_collision.append((event.frame_number, intensity))
        if len(self._history_collision) > 32:
            self._history_collision.pop(0)

    @staticmethod
    def _parse_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        # print(str(event.crossed_lane_markings)) [carla.libcarla.LaneMarking.Solid]
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        # S for Solid B for Broken
        self._history_invasion.append(text[0][1])
        if len(self._history_invasion) > 32:
             self._history_invasion.pop(0)

    def _generate_point_list(self):
        """
        generate a list of attention point represent the index for every pixel
        :return: Cartesian coordinates for pixels
        """
        r = int(ENV_CONFIG["x_res"]/2)
        point_list = []
        for i in range(r, -r, -1):
            for j in range(-r, r, 1):
                point_list.append((j, i))
        return point_list

    def step(self, action):

        def compute_reward(info, prev_info):
            reward = 0.0
            reward += np.clip(info["speed"], 0, 15)/4
            reward += info['distance']
            if info["collision"] == 1:
                reward -= 30
            elif 2 <= info["collision"] < 5:
                reward -= info['speed'] * 2
            elif info["collision"] > 5:
                reward -= info['speed'] * 1

            print("current speed", info["speed"], "current reward", reward, "collision", info['collision'])
            new_invasion = list(set(info["lane_invasion"]) - set(prev_info["lane_invasion"]))
            if 'S' in new_invasion:     # go across solid lane
                 reward -= info["speed"]
            elif 'B' in new_invasion:   # go across broken lane
                 reward -= 0.4 * info["speed"]
            return reward

        throttle = float(np.clip(action[0], 0, 1))
        brake = float(np.abs(np.clip(action[0], -1, 0)))
        steer = float(np.clip(action[1], -1, 1))
        distance_before_act = ((self._history_waypoint[-1].transform.location.x - self.vehicle.get_location().x)**2 + 
                   (self._history_waypoint[-1].transform.location.y - self.vehicle.get_location().y)**2)**0.5
      
        # command = self.planner()
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        # get image
        # time.sleep(0.047)

        t = self.vehicle.get_transform()
        v = self.vehicle.get_velocity()
        c = self.vehicle.get_control()
        acceleration = self.vehicle.get_acceleration()
        if len(self._history_invasion) > 0:
            invasion = self._history_invasion[-1]
        else:
            invasion = []
             
        command = self.planner()
       
        distance_after_act = ((self._history_waypoint[-2].transform.location.x - self.vehicle.get_location().x)**2 + 
                              (self._history_waypoint[-2].transform.location.y - self.vehicle.get_location().y)**2)**0.5
        info = {"speed": math.sqrt(v.x**2 + v.y**2 + v.z**2),  # m/s
                "acceleration": math.sqrt(acceleration.x**2 + acceleration.y**2 + acceleration.z**2),
                "location_x": t.location.x,
                "location_y": t.location.y,
                "Throttle": c.throttle,
                "Steer": c.steer,
                "Brake": c.brake,
                "command": command,
                "distance": distance_before_act - distance_after_act,  # distance to waypoint
                "lane_invasion": invasion,
                "traffic_light": str(self.vehicle.get_traffic_light_state()),    # Red Yellow Green Off Unknown
                "is_at_traffic_light": self.vehicle.is_at_traffic_light(),       # True False
                "collision": len(self._history_collision)}

        self._history_info.append(info)
        reward = compute_reward(self._history_info[-1], self._history_info[-2])
        # print(self._history_info[-1]["speed"], self._history_info[-1]["collision"])

        # early stop
        done = False
        if ENV_CONFIG["early_stop"]:
            if len(self._history_collision) > 0:
                # print("collisin length", len(self._history_collision))
                done = True
            elif reward < -100:
                done = True

        if ENV_CONFIG["image_mode"] == "encode":   # stack gray depth segmentation
            obs = np.concatenate([self._image_rgb1[-1], self._image_rgb1[-2],
                                  self.encode_measurement(info)], axis=2)
        else:
            obs = self._image_rgb1[-1]



        mask = self._compute_mask(action)
        if ENV_CONFIG["attention_mode"] == "soft":
            obs[:, :, 0:ENV_CONFIG["attention_channel"]] = obs[:, :, 0:ENV_CONFIG["attention_channel"]] + mask
        else:
            obs[:, :, 0:ENV_CONFIG["attention_channel"]] = obs[:, :, 0:ENV_CONFIG["attention_channel"]] * mask

        self._obs_collect.append(np.clip(obs[:, :, 0:3], 0, 255))  # clip in case we want render
        if len(self._obs_collect) > 32:
            self._obs_collect.pop(0)

        return np.clip(obs, 0, 255), reward, done, self._history_info[-1]

    def render(self):

        display = pygame.display.set_mode(
            (ENV_CONFIG["x_res"], ENV_CONFIG["y_res"]),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        # surface = pygame.surfarray.make_surface(env._image_rgb1[-1].swapaxes(0, 1))
        surface = pygame.surfarray.make_surface(env._obs_collect[-1].swapaxes(0, 1))
        display.blit(surface, (0, 0))
        time.sleep(0.01)
        pygame.display.flip()

    def planner(self):
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        waypoint = random.choice(waypoint.next(12.0))
        self._history_waypoint.append(waypoint)
        yaw = waypoint.transform.rotation.yaw
        if yaw > -90 or yaw < 60:
            command = "turn_right"
        elif yaw > 60 and yaw < 120:
            command = "lane_keep"
        elif yaw > 120 or yaw < -90:
            command = "turn_left"
        return self.command[command]

    @staticmethod
    def encode_measurement(py_measurements):
        """encode measurements into another channel"""
        feature_map = np.zeros([4, 4])
        feature_map[0, :] = (py_measurements["command"]) * 60.0
        feature_map[1, :] = (py_measurements["speed"]) * 4.0
        feature_map[2, :] = (py_measurements["command"]) * 60.0
        feature_map[3, :] = (py_measurements["Steer"]+1) * 120.0
        stack = int(ENV_CONFIG["x_res"]/4)
        feature_map = np.tile(feature_map, (stack, stack))
        feature_map = feature_map.astype(np.float32)
        return feature_map[:, :, np.newaxis]


if __name__ == '__main__':
    env = CarlaEnv()
    obs = env.reset()
    print(obs.shape)
    done = False
    i = 0
    start = time.time()
    R = 0
    while i < 50:
        env.render()
        obs, reward, done, info = env.step([1, 0, 0, 0, 1])
        R += reward
        print(R)
        i += 1
    # print("{:.2f} fps".format(float(len(env._image_rgb1) / (time.time() - start))))
    print("{:.2f} fps".format(float(i / (time.time() - start))))
    print(R)
