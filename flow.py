import copy
import json
import time
import cityflow
import numpy as np
from datetime import datetime
import os
import random
import re
import math


class Simulator:
    def __init__(self):
        self.vec_size = 12
        self.intersection_size = 4
        self.route = []
        self.OTD = []
        self.cityflow_route = []
        self.road = {'1,3': np.zeros((self.vec_size, 1)), '3,4': np.zeros((self.vec_size, 1)),
                     '4,2': np.zeros((self.vec_size, 1)), '2,1': np.zeros((self.vec_size, 1))}
        self.speed = {'1,3': np.zeros((self.vec_size, 1)), '3,4': np.zeros((self.vec_size, 1)),
                      '4,2': np.zeros((self.vec_size, 1)), '2,1': np.zeros((self.vec_size, 1))}

    def generate_config(self, route, flow):
        flows = []
        for i in range(len(route)):
            flow_base = {
                "vehicle": {
                    "length": 5.0,
                    "width": 2.0,
                    "maxPosAcc": 1.0,
                    "maxNegAcc": 4.5,
                    "usualPosAcc": 2.0,
                    "usualNegAcc": 4.5,
                    "minGap": 2.5,
                    "maxSpeed": 15,
                    "headwayTime": 1.5
                },
                "route": [
                ],
                "interval": 10.0,
                "startTime": 0,
                "endTime": 600
            }
            if route[i] == [1, 3, 4]:
                # flow_base['route'] = ['road_1_0_1', 'road_1_1_1', 'road_1_2_0', 'road_2_2_0']
                flow_base['route'] = ['road_1_1_1', 'road_1_2_0', 'road_2_2_0']
            if route[i] == [3, 4, 2]:
                # flow_base['route'] = ['road_0_2_0', 'road_1_2_0', 'road_2_2_3', 'road_2_1_3']
                flow_base['route'] = ['road_1_2_0', 'road_2_2_3', 'road_2_1_3']
            if route[i] == [4, 2, 1]:
                # flow_base['route'] = ['road_2_3_3', 'road_2_2_3', 'road_2_1_2', 'road_1_1_2']
                flow_base['route'] = ['road_2_2_3', 'road_2_1_2', 'road_1_1_2']
            if route[i] == [2, 1, 3]:
                # flow_base['route'] = ['road_3_1_2', 'road_2_1_2', 'road_1_1_1', 'road_1_2_1']
                flow_base['route'] = ['road_2_1_2', 'road_1_1_1', 'road_1_2_1']
            for j in range(len(flow[i])):
                new_flow = copy.deepcopy(flow_base)
                new_flow["startTime"] = j * 600
                new_flow["endTime"] = (j + 1) * 600
                new_flow["interval"] = max(600 / float(flow[i][j] + 1), 1)
                flows.append(new_flow)
        path = '/mnt/e/Cityflow/examples/flow/generate_flow_' + str(int(time.time())) + '.json'
        with open(path, "w") as f:
            json.dump(flows, f, indent=2)
        config_base = {"interval": 1.0, "seed": 0, "dir": "",
                       "roadnetFile": "/mnt/e/CityFlow/examples/roadnet_4x7.json",
                       "flowFile": path, "rlTrafficLight": False, "laneChange": True, "saveReplay": True,
                       "roadnetLogFile": "/mnt/e/CityFlow/frontend/replay/exp_roadnet.json",
                       "replayLogFile": "/mnt/e/CityFlow/frontend/replay/exp_replay.txt"}
        with open('/mnt/e/CityFlow/examples/config_exp.json', 'w') as f:
            json.dump(config_base, f, indent=2)

    def volume2speed(self, volume):
        if volume < 16:
            return 15
        elif volume < 196:
            return 60 / math.sqrt(volume)
        else:
            return 4

    def convert(self, OD_list, convert_type):
        for OD in OD_list.keys():
            roads = []
            routes = []
            route = ""
            for c in OD:
                if c == ",":
                    routes.append(int(route))
                    route = ""
                    continue
                route = route + c
            routes.append(int(route))
            for k in range(len(routes) - 1):
                road = str(routes[k]) + ',' + str(routes[k + 1])
                roads.append(road)
            self.route.append(roads)
            self.cityflow_route.append(routes)
            self.OTD.append(OD_list[OD])
        if convert_type == 'cityflow':
            self.generate_config(self.cityflow_route, self.OTD)
            eng = cityflow.Engine('/mnt/e/CityFlow/examples/config_exp.json', thread_num=4)
            lanes = ['road_1_1_1', 'road_1_2_0', 'road_2_2_3', 'road_2_1_2']
            mapping = {'road_1_1_1': '1,3', 'road_1_2_0': '3,4', 'road_2_2_3': '4,2', 'road_2_1_2': '2,1'}
            for i in range(12):
                lane_vehicles = {'1,3': {}, '3,4': {}, '4,2': {}, '2,1': {}}
                for _ in range(600):
                    eng.next_step()
                    vehicles = eng.get_lane_vehicles()
                    speeds = eng.get_vehicle_speed()
                    for lane in lanes:
                        for j in range(3):
                            vehicle = vehicles[lane + '_' + str(j)]
                            for v in vehicle:
                                if 'shadow' in v:
                                    continue
                                if v in lane_vehicles[mapping[lane]]:
                                    lane_vehicles[mapping[lane]][v].append(speeds[v])
                                else:
                                    lane_vehicles[mapping[lane]][v] = []
                for road in lane_vehicles:
                    self.road[road][i] = len(lane_vehicles[road].keys())
                    avg = 0.0
                    for v in lane_vehicles[road].keys():
                        if not len(lane_vehicles[road][v]) == 0:
                            avg = avg + np.average(lane_vehicles[road][v])
                    if len(lane_vehicles[road].keys()) == 0:
                        self.speed[road][i] = avg
                    else:
                        self.speed[road][i] = avg / len(lane_vehicles[road].keys())
        if convert_type == 'rule_0.5':
            for i in range(self.vec_size):
                for j in range(len(self.OTD)):
                    road = self.route[j][0]
                    self.road[road][i] = self.road[road][i] + self.OTD[j][i]
                    if i <= self.vec_size - 2:
                        road = self.route[j][0]
                        self.road[road][i + 1] = self.road[road][i + 1] + self.OTD[j][i] * 0.5
                        road = self.route[j][1]
                        self.road[road][i + 1] = self.road[road][i + 1] + self.OTD[j][i] * 0.5
                    if i <= self.vec_size - 3:
                        road = self.route[j][1]
                        self.road[road][i + 2] = self.road[road][i + 2] + self.OTD[j][i] * 0.5
            for i in range(self.vec_size):
                for key in self.road.keys():
                    self.speed[key][i] = self.volume2speed(self.road[key][i])
        if convert_type == 'rule_proportion':
            for i in range(self.vec_size - 1):
                for j in range(len(self.OTD)):
                    road = self.route[j][0]
                    self.road[road][i] = self.road[road][i] + self.OTD[j][i]
                    if i <= self.vec_size - 2:
                        v1 = self.road[self.route[j][0]][i] + 1
                        v2 = self.road[self.route[j][1]][i] + 1
                        road = self.route[j][0]
                        self.road[road][i + 1] = self.road[road][i + 1] + self.OTD[j][i] * v1 / (v1 + v2)
                        road = self.route[j][1]
                        self.road[road][i + 1] = self.road[road][i + 1] + self.OTD[j][i] * v2 / (v1 + v2)
                    if i <= self.vec_size - 3:
                        v1 = self.road[self.route[j][0]][i] + 1
                        v2 = self.road[self.route[j][1]][i] + 1
                        road = self.route[j][1]
                        self.road[road][i + 2] = self.road[road][i + 2] + self.OTD[j][i] * v2 / (v1 + v2)

        return self.road, self.speed


def normal(sigma, x):
    return math.exp((-1 * ((x - 6) ** 2) / (2 * sigma * sigma))) / (sigma * math.sqrt(2 * math.pi))


def generate_flow(convert_type, flow_type):
    n_time_interval = 12
    list_road_names = ["1,3", "3,4", "4,2", "2,1"]
    list_OD_names = ["1,3,4", "3,4,2", "4,2,1"]

    OD_lists = []
    volumes = []
    speeds = []
    test_input = []
    test_volumes = []
    test_speeds = []
    test_const = []
    input = []
    const = []
    paremeter = {"1,3,4": [np.random.randint(12, 1200), float(np.random.rand(1, 1)) * 5],
                 "3,4,2": [np.random.randint(12, 1200), float(np.random.rand(1, 1)) * 5],
                 "4,2,1": [np.random.randint(12, 1200), float(np.random.rand(1, 1)) * 5]}
    for i in range(1):
        simulator = Simulator()
        OD_list = {"1,3,4": np.random.randint(1, 100, size=(12, 1)),
                   "3,4,2": np.random.randint(1, 100, size=(12, 1)),
                   "4,2,1": np.random.randint(1, 100, size=(12, 1))}
        if flow_type == 'random':
            OD_list = {"1,3,4": np.random.randint(1, 100, size=(12, 1)),
                       "3,4,2": np.random.randint(1, 100, size=(12, 1)),
                       "4,2,1": np.random.randint(1, 100, size=(12, 1))}
        if flow_type == 'poisson':
            OD_list = {"1,3,4": np.random.poisson(3, size=(12, 1)) * 16.67,
                       "3,4,2": np.random.poisson(3, size=(12, 1)) * 16.67,
                       "4,2,1": np.random.poisson(3, size=(12, 1)) * 16.67}
        if flow_type == 'increasing':
            OD_list = {"1,3,4": np.sort(np.random.randint(1, 100, size=(12, 1))),
                       "3,4,2": np.sort(np.random.randint(1, 100, size=(12, 1))),
                       "4,2,1": np.sort(np.random.randint(1, 100, size=(12, 1)))}
        if flow_type == 'decreasing':
            OD_list = {"1,3,4": -np.sort(-np.random.randint(1, 100, size=(12, 1))),
                       "3,4,2": -np.sort(-np.random.randint(1, 100, size=(12, 1))),
                       "4,2,1": -np.sort(-np.random.randint(1, 100, size=(12, 1)))}
        if flow_type == 'uniform':
            OD_list = {"1,3,4": np.ones((12, 1)) * 50,
                       "3,4,2": np.ones((12, 1)) * 50,
                       "4,2,1": np.ones((12, 1)) * 50}
        if flow_type == 'normal':
            for key in OD_list.keys():
                sigma = 4
                for j in range(12):
                    OD_list[key][j] = normal(sigma, j + 1) * paremeter[key][0]
        OD_lists.append(OD_lists)
        volume_list, speed_list = simulator.convert(OD_list, convert_type)
        volume = np.concatenate([
            volume_list["1,3"].reshape((1, 1, n_time_interval)),
            volume_list["3,4"].reshape((1, 1, n_time_interval)),
            volume_list["4,2"].reshape((1, 1, n_time_interval)),
            volume_list["2,1"].reshape((1, 1, n_time_interval))],
            axis=1)
        speed = np.concatenate([
            speed_list["1,3"].reshape((1, 1, n_time_interval)),
            speed_list["3,4"].reshape((1, 1, n_time_interval)),
            speed_list["4,2"].reshape((1, 1, n_time_interval)),
            speed_list["2,1"].reshape((1, 1, n_time_interval))],
            axis=1)
        if i >= 2000:
            test_const.append(0.0)
            test_input.append(np.concatenate([
                OD_list["1,3,4"].reshape((1, 1, n_time_interval)),
                OD_list["3,4,2"].reshape((1, 1, n_time_interval)),
                OD_list["4,2,1"].reshape((1, 1, n_time_interval)),
            ], axis=1))
            test_volumes.append(volume)
            test_speeds.append(speed)
            continue
        const.append(0.0)
        input.append(np.concatenate([
            OD_list["1,3,4"].reshape((1, 1, n_time_interval)),
            OD_list["3,4,2"].reshape((1, 1, n_time_interval)),
            OD_list["4,2,1"].reshape((1, 1, n_time_interval)),
        ], axis=1))
        volumes.append(volume)
        speeds.append(speed)

    np.save('flow/hz_otd_train_' + convert_type + flow_type + '.npy', input)
    np.save('flow/hz_otd_test_' + convert_type + flow_type + '.npy', test_input)
    np.save('flow/hz_volume_train_' + convert_type + flow_type + '.npy', volumes)
    np.save('flow/hz_volume_test_' + convert_type + flow_type + '.npy', test_volumes)
    np.save('flow/hz_speed_train_' + convert_type + flow_type + '.npy', speeds)
    np.save('flow/hz_speed_test_' + convert_type + flow_type + '.npy', test_speeds)


if __name__ == '__main__':
    convert = ['cityflow']
    flow = ['random']
    for c in convert:
        for f in flow:
            generate_flow(c, f)
