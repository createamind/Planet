#!/usr/bin/env python

import glob
import os
import sys
import re
import weakref
try:
    sys.path.append('/home/gu/Documents/carla94/PythonAPI/carla-0.9.4-py3.5-linux-x86_64.egg')
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

""" default is rgb 
    stack for gray depth segmentation stack together
    encode for encode measurement in forth channel """

ENV_CONFIG = {
    "x_res": 96,
    "y_res": 96,
    "discrete_actions": False,
    "image_mode": "stack",   # stack3 encode4
    "early_stop": False,      # if we use planet this has to be False
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

        if ENV_CONFIG["image_mode"] == "encode":
            framestack = 4
        elif ENV_CONFIG["image_mode"] == "stack":
            framestack = 3
        else:
            framestack = 3

        image_space = Box(
            0,
            255,
            shape=(config["y_res"], config["x_res"], framestack),
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
        self.feature_map = None
        # actors
        self.actor_list = []          # save actor list for destroying them after finish
        self.vehicle = None
        self.collision_sensor = None
        self.camera_rgb = None
        self.invasion_sensor = None
        self.camera_segmentation = None
        self.camera_depth = None
        # states and data
        self._history_info = []       # info history
        self._history_collision = []  # collision history
        self._history_invasion = []   # invasion history
        self._image_depth = []        # save a list of depth image
        self._image_rgb = []          # save a list of rgb image
        self._image_segmentation = []
        self._image_gray = []
        # initialize our world
        self.server_port = 2000
        self.world = None
        connect_fail_times = 0
        while self.world is None:
            try:
                self.client = carla.Client("localhost", self.server_port)
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
        fail_time = 0
        while None in self.actor_list or len(self.actor_list) < 5:
            try:
                bp_library = world.get_blueprint_library()

                # setup vehicle
                spawn_point = random.choice(world.get_map().get_spawn_points())
                bp_vehicle = bp_library.find('vehicle.lincoln.mkz2017')
                bp_vehicle.set_attribute('role_name', 'hero')
                self.vehicle = world.try_spawn_actor(bp_vehicle, spawn_point)
                self.actor_list.append(self.vehicle)

                # setup rgb camera
                camera_transform = carla.Transform(carla.Location(x=1, y=0, z=2))
                camera_rgb = bp_library.find('sensor.camera.rgb')
                camera_rgb.set_attribute('fov', '120')
                camera_rgb.set_attribute('image_size_x', str(ENV_CONFIG["x_res"]))
                camera_rgb.set_attribute('image_size_y', str(ENV_CONFIG["y_res"]))
                self.camera_rgb = world.try_spawn_actor(camera_rgb, camera_transform, attach_to=self.vehicle)
                self.actor_list.append(self.camera_rgb)

                # setup depth camera
                camera_depth = bp_library.find('sensor.camera.depth')
                camera_depth.set_attribute('fov', '120')
                camera_depth.set_attribute('image_size_x', str(ENV_CONFIG["x_res"]))
                camera_depth.set_attribute('image_size_y', str(ENV_CONFIG["y_res"]))
                self.camera_depth = world.try_spawn_actor(camera_depth, camera_transform, attach_to=self.vehicle)
                self.actor_list.append(self.camera_depth)

                # setup segmentation camera
                camera_segmentation = bp_library.find('sensor.camera.semantic_segmentation')
                camera_segmentation.set_attribute('fov', '120')
                camera_segmentation.set_attribute('image_size_x', str(ENV_CONFIG["x_res"]))
                camera_segmentation.set_attribute('image_size_y', str(ENV_CONFIG["y_res"]))
                self.camera_segmentation = world.try_spawn_actor(camera_segmentation, camera_transform, attach_to=self.vehicle)
                self.actor_list.append(self.camera_segmentation)

                # add collision sensors
                bp = bp_library.find('sensor.other.collision')
                self.collision_sensor = world.try_spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
                self.actor_list.append(self.collision_sensor)

                # add invasion sensors
                bp = bp_library.find('sensor.other.lane_detector')
                self.invasion_sensor = world.try_spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
                self.actor_list.append(self.invasion_sensor)
            except:
                fail_time += 1
                time.sleep(0.1)

            if fail_time > 20:
                print("spawn actor fail, so sad!!!")
                break

    def reset(self):
        self.restart()
        weak_self = weakref.ref(self)
        # set invasion sensor
        self.invasion_sensor.listen(lambda event: self._parse_invasion(weak_self, event))
        # set collision sensor
        self.collision_sensor.listen(lambda event: self._parse_collision(weak_self, event))
        # set rgb camera sensor
        self.camera_rgb.listen(lambda image: self._parse_image(weak_self, image,
                                                               carla.ColorConverter.Raw, 'rgb'))
        # set depth camera sensor
        self.camera_depth.listen(lambda image: self._parse_image(weak_self, image,
                                                                 carla.ColorConverter.LogarithmicDepth, 'depth'))
        # set segmentation camera sensor
        self.camera_segmentation.listen(lambda image: self._parse_image(weak_self, image,
                                                                        carla.ColorConverter.Raw, 'seg'))

        while len(self._image_rgb) < 4:
            print("resetting")
            time.sleep(0.01)
        if ENV_CONFIG["image_mode"] == "encode":   # stack gray depth segmentation
            obs = np.concatenate([self._image_gray[-1][:, :, np.newaxis],
                                  self._image_depth[-1][:, :, np.newaxis],
                                  self._image_segmentation[-1][:, :, np.newaxis]*21,
                                  np.zeros([ENV_CONFIG['x_res'], ENV_CONFIG['y_res'], 1])], axis=2)
        elif ENV_CONFIG["image_mode"] == "stack":
            obs = np.concatenate([self._image_gray[-1][:, :, np.newaxis],
                                  self._image_depth[-1][:, :, np.newaxis],
                                  self._image_segmentation[-1][:, :, np.newaxis] * 21], axis=2)
        else:
            obs = self._image_rgb[-1]

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
                "is_at_traffic_light": self.vehicle.is_at_traffic_light(),       # True False
                "collision": len(self._history_collision)
        }

        self._history_info.append(info)

        return obs

    @staticmethod
    def _parse_image(weak_self, image, cc, use):
        """convert BGRA to RGB"""
        self = weak_self()
        if not self:
            return

        def convert(cc):
            image.convert(cc)
            # image.save_to_disk('_out/%08d' % image.frame_number)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, -2:-5:-1]
            return array

        if use == 'rgb':
            array = convert(cc)
            self._image_rgb.append(array)
            self._image_gray.append(0.45*array[:, :, 0] +
                                    0.45*array[:, :, 1] +
                                    0.1*array[:, :, 2])
            if len(self._image_gray) > 32:
                self._image_gray.pop(0)
            if len(self._image_rgb) > 32:
                self._image_rgb.pop(0)
        if use == 'depth':
            array = convert(cc)
            # it is the same in each channel of depth image
            self._image_depth.append(array[:, :, 0])
            if len(self._image_depth) > 32:
                self._image_depth.pop(0)
        if use == 'seg':
            array = convert(cc)
            # segmentation information encode in red channel
            self._image_segmentation.append(array[:, :, 0])
            if len(self._image_segmentation) > 32:
                self._image_segmentation.pop(0)


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

    def step(self, action):

        def compute_reward(info, prev_info):
            reward = 0.0
            reward += np.clip(info["speed"], 0, 30)/6

            if info["collision"] == 1:
                reward -= 100

            elif 2 <= info["collision"] < 5:
                reward -= info['speed'] * 15
            elif info["collision"] > 5:
                reward -= info['speed'] * 10

            # print(info["collision"])
            new_invasion = list(set(info["lane_invasion"]) - set(prev_info["lane_invasion"]))
            if 'S' in new_invasion:     # go across solid lane
                reward -= 3
            elif 'B' in new_invasion:   # go across broken lane
                reward -= 2
            return reward

        if self.config["discrete_actions"]:
            action = self.DISCRETE_ACTIONS[int(action)]

        throttle = float(np.clip(action[0], 0, 1))
        brake = float(np.abs(np.clip(action[0], -1, 0)))
        steer = float(np.clip(action[1], -1, 1))
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
        # get image
        # time.sleep(0.1)

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
                "is_at_traffic_light": self.vehicle.is_at_traffic_light(),       # True False
                "collision": len(self._history_collision)
        }

        self._history_info.append(info)
        reward = compute_reward(self._history_info[-1], self._history_info[-2])
        # print(self._history_info[-2]["speed"],  self._history_info[-1]["speed"])

        # early stop
        done = False
        if ENV_CONFIG["early_stop"]:
            if len(self._history_collision) > 0:
                # print("collisin length", len(self._history_collision))
                done = True
            elif reward < -100:
                done = True

        if ENV_CONFIG["image_mode"] == "encode":   # stack gray depth segmentation
            obs = np.concatenate([self._image_gray[-1][:, :, np.newaxis],
                                  self._image_depth[-1][:, :, np.newaxis],
                                  self._image_segmentation[-1][:, :, np.newaxis]*21,
                                  self.encode_measurement(info)], axis=2)
        elif ENV_CONFIG["image_mode"] == "stack":
            obs = np.concatenate([self._image_gray[-1][:, :, np.newaxis],
                                  self._image_depth[-1][:, :, np.newaxis],
                                  self._image_segmentation[-1][:, :, np.newaxis] * 21], axis=2)
        else:
            obs = self._image_rgb[-1]

        return obs, reward, done, self._history_info[-1]

    def render(self):
        import pygame
        display = pygame.display.set_mode(
            (ENV_CONFIG["x_res"], ENV_CONFIG["y_res"]),
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

    @staticmethod
    def encode_measurement(py_measurements):
        feature_map = np.zeros([4, 4])
        feature_map[0, :] = (py_measurements["command"]) * 60
        feature_map[1, :] = (py_measurements["speed"]) * 4
        feature_map[2, :] = (py_measurements["command"]) * 60
        feature_map[3, :] = (py_measurements["acceleration"]) * 20
        stack = int(ENV_CONFIG["x_res"]/4)
        feature_map = np.tile(feature_map, (stack, stack))
        return feature_map[:, :, np.newaxis]


if __name__ == '__main__':
    env = CarlaEnv()

    obs = env.reset()
    print(obs.shape)
    done = False
    i = 0
    while not done:
        # env.render()
        obs, reward, done, info = env.step([1, 0])
        # print(len(env._image_rgb), obs.shape)
        print(i, reward)
        i += 1
    # for actor in env.actor_list:
    #     print(actor.id)
    #     actor.destroy()
    #     print("test", actor.is_alive)
