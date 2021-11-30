import copy
import csv
import json
import math
import random
import time

import cityflow
import numpy as np
import pandas as pd
import tensorflow as tf

loss_gls = {}


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
    def v2s(self, volume):
        speed = copy.deepcopy(volume)
        for i in range(len(speed)):
            for j in range(4):
                for k in range(12):
                    speed[i][0][j][k] = self.volume2speed(speed[i][0][j][k])
        return speed
    def v2s_test(self, volume):
        speed = copy.deepcopy(volume)
        for j in range(4):
            for k in range(12):
                speed[j][k] = self.volume2speed(speed[j][k])
        return speed
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
                flow_base['route'] = ['road_1_0_1', 'road_1_1_1', 'road_1_2_0', 'road_2_2_0']
            if route[i] == [3, 4, 2]:
                flow_base['route'] = ['road_0_2_0', 'road_1_2_0', 'road_2_2_3', 'road_2_1_3']
            if route[i] == [4, 2, 1]:
                flow_base['route'] = ['road_2_3_3', 'road_2_2_3', 'road_2_1_2', 'road_1_1_2']
            if route[i] == [2, 1, 3]:
                flow_base['route'] = ['road_3_1_2', 'road_2_1_2', 'road_1_1_1', 'road_1_2_1']
            for j in range(len(flow[i])):
                new_flow = copy.deepcopy(flow_base)
                new_flow["startTime"] = j * 600
                new_flow["endTime"] = (j + 1) * 600
                new_flow["interval"] = max(600 / float(flow[i][j]), 1)
                flows.append(new_flow)
        path = '/mnt/e/exp/data/generate_flow_' + str(int(time.time())) + '.json'
        with open(path, "w") as f:
            json.dump(flows, f, indent=2)
        config_base = {"interval": 1.0, "seed": 0, "dir": "",
                       "roadnetFile": "/mnt/e/CityFlow/examples/roadnet_2x2.json",
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
                        for j in range(4):
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


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def EM(Volume, road2OD):
    Volumes = []
    for i in range(len(Volume)):
        Volumes.append(np.mean(Volume[i]))
    OD_mean = np.zeros(3)
    Probs = []
    for i in range(len(road2OD)):
        j = len(road2OD[i])
        Prob = []
        for k in range(j):
            Prob.append(1 / j)
        # Prob[0] = 1
        Probs.append(Prob)
    for i in range(1000):
        # print("Epoch: ", i)
        OD_prediction = []
        for j in range(3):
            OD_prediction.append([])
        for j in range(len(road2OD)):
            for k in range(len(road2OD[j])):
                OD_prediction[road2OD[j][k]].append(Probs[j][k] * Volumes[j])
        for j in range(len(OD_mean)):
            OD_mean[j] = np.mean(OD_prediction[j])
        for j in range(len(road2OD)):
            sum = 0
            for k in range(len(road2OD[j])):
                sum += OD_mean[road2OD[j][k]]
            for k in range(len(road2OD[j])):
                Probs[j][k] = OD_mean[road2OD[j][k]] / sum
    return OD_mean

class Model:

    def __init__(self):
        self.batch_size = 128
        self.units_size = 128
        self.learning_rate_b = 0.001
        self.learning_rate_c = 0.001
        self.epoch = 10000

        self.n_time_interval = 12
        self.n_od = 3
        self.n_road = 4
        self.early_stop = 50

    def dense_layer(self, input, units, activation="relu"):
        if activation == "relu":
            output = tf.nn.relu(tf.layers.dense(input, units))
        elif activation == "sigmoid":
            output = tf.nn.sigmoid(tf.layers.dense(input, units))
        elif activation == "softmax":
            output = tf.nn.softmax(tf.layers.dense(input, units))
        else:
            output = None
        return output

    def build_network(self, OD, speed):
        with tf.variable_scope("odinit"):
            list_OD = []
            for od_ind in range(self.n_od):
                list_OD.append(tf.gather(OD, od_ind, axis=1))
        with tf.variable_scope("speed2od"):
            tensor_otd = tf.reshape(list_OD, (-1, self.n_od * self.n_time_interval))
            speed = tf.reshape(speed, (-1, self.n_road * self.n_time_interval))
            hid = tf.layers.dropout(tf.nn.relu(tf.layers.dense(speed, self.units_size)), rate=0.3)
            out = tf.nn.sigmoid(tf.layers.dense(hid, self.n_od * self.n_time_interval)) * 100
        with tf.variable_scope("loss_v2s"):
            loss = tf.losses.mean_squared_error(out, tensor_otd)
        train_vars = tf.trainable_variables()
        v2s_vars = [var for var in train_vars if var.name.startswith('speed2od')]
        v2s_optimizer = tf.train.AdamOptimizer(self.learning_rate_c).minimize(loss, var_list=v2s_vars)
        return loss, v2s_optimizer, out

    def train(self, convert_type, flow_type):
        tf.reset_default_graph()
        input = np.load('flow/otd_train_' + convert_type + flow_type + '.npy')
        volumes = np.load('flow/volume_train_' + convert_type + flow_type + '.npy')
        speeds = np.load('flow/speed_train_' + convert_type + flow_type + '.npy')
        test_input = np.load('flow/otd_test_' + convert_type + flow_type + '.npy')
        test_volumes = np.load('flow/volume_test_' + convert_type + flow_type + '.npy')
        test_speeds = np.load('flow/speed_test_' + convert_type + flow_type + '.npy')
        if convert_type == 'cityflow':
            simulator = Simulator()
            speeds = simulator.v2s(volumes)
            test_speeds = simulator.v2s(test_volumes)
        input = np.reshape(input, (-1, self.n_od, self.n_time_interval))
        volumes = np.reshape(volumes, (-1, self.n_road, self.n_time_interval))
        speeds = np.reshape(speeds, (-1, self.n_road, self.n_time_interval))
        test_input = np.reshape(test_input, (-1, self.n_od, self.n_time_interval))
        test_volumes = np.reshape(test_volumes, (-1, self.n_road, self.n_time_interval))
        test_speeds = np.reshape(test_speeds, (-1, self.n_road, self.n_time_interval))
        otd_input = tf.placeholder(tf.float32, [None, self.n_od, self.n_time_interval], name="otd_input")
        speed_label = tf.placeholder(tf.float32, [None, self.n_road, self.n_time_interval], name="speed_label")
        loss, optimizer, out = self.build_network(otd_input, speed_label)

        # print(np.shape(out_otd))
        loss_od = 0.0
        loss_volume = 0.0
        loss_speed = 0.0
        loss_gls[convert_type][flow_type] = [convert_type, flow_type]
        for o in range(100):
            test_otd = test_input[o]
            test_speed = test_speeds[o]
            test_volume = test_volumes[o]
            OD_list = {"1,3,4": np.random.randint(1, 100, size=(12, 1)),
                       "3,4,2": np.random.randint(1, 100, size=(12, 1)),
                       "4,2,1": np.random.randint(1, 100, size=(12, 1))}
            road2OD = [[0,2],[1,2],[0,1],[0,2]]
            otd = []
            otd_EM = EM(test_volume, road2OD)
            for i in range(len(list(OD_list.keys()))):
                otd.append(np.ones((self.n_time_interval, 1)) * otd_EM[i])
            otd = np.reshape(otd, (self.n_od, self.n_time_interval))
            for i in range(len(list(OD_list.keys()))):
                OD_list[list(OD_list.keys())[i]] = otd[i]

            simulator = Simulator()
            volume_list, speed_list = simulator.convert(OD_list, convert_type)
            volume = [
                volume_list["1,3"].reshape(self.n_time_interval),
                volume_list["3,4"].reshape(self.n_time_interval),
                volume_list["4,2"].reshape(self.n_time_interval),
                volume_list["2,1"].reshape(self.n_time_interval)]
            speed = [
                speed_list["1,3"].reshape(self.n_time_interval),
                speed_list["3,4"].reshape(self.n_time_interval),
                speed_list["4,2"].reshape(self.n_time_interval),
                speed_list["2,1"].reshape(self.n_time_interval)]
            if convert_type == 'cityflow':
                simulator = Simulator()
                speed = simulator.v2s_test(volume)
            loss_od += rmse(otd, test_otd)
            loss_volume += rmse(volume, test_volume)
            loss_speed += rmse(speed, test_speed)
        loss_od = loss_od / 100
        loss_volume = loss_volume / 100
        loss_speed = loss_speed / 100
        loss_gls[convert_type][flow_type].append(loss_od)
        loss_gls[convert_type][flow_type].append(loss_volume)
        loss_gls[convert_type][flow_type].append(loss_speed)


if __name__ == "__main__":
    convert = ['rule_0.5']
    flow = ['random', 'increasing', 'decreasing', 'uniform', 'normal', 'poisson']
    for c in convert:
        loss_gls[c] = {}
        for f in flow:
            loss_gls[c][f] = []
            model = Model()
            model.train(c, f)
    with open('EM_cityflow.csv', 'w') as f:
        writer = csv.writer(f)
        for c in convert:
            for fl in flow:
                writer.writerow(loss_gls[c][fl])
