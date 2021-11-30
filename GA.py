import cityflow
import json
import numpy as np
import copy
import time
import random
import math
import csv
rate = 0.5


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


def get_loss(candidate, convert_type, test_speed):
    simulator = Simulator()
    OD_list = {'1, 3, 4': candidate[0], '3, 4, 2': candidate[1], '4, 2, 1': candidate[2]}
    _, speed_dict = simulator.convert(OD_list, convert_type)
    speed = []
    for key in speed_dict.keys():
        speed.append(speed_dict[key][:, 0])
    return rmse(speed, test_speed[0])


def check(seq):
    for i in range(len(seq)):
        seq[i] = int(seq[i])
        if seq[i] < 1:
            seq[i] = 1
        if seq[i] > 100:
            seq[i] = 100
    return seq


def mutate(candidate):
    mutation = copy.deepcopy(candidate)
    for i in range(len(mutation)):
        for j in range(12):
            if random.random() < rate:
                mutation[i][j] = mutation[i][j] * (0.75 + random.random() / 2) + random.randint(0, 20) - 10
        mutation[i] = check(mutation[i])
    return mutation


def output(candidate, convert_type, test_input, test_volume, test_speed):
    simulator = Simulator()
    OD_list = {'1, 3, 4': candidate[0], '3, 4, 2': candidate[1], '4, 2, 1': candidate[2]}
    volume_dict, speed_dict = simulator.convert(OD_list, convert_type)
    speed = []
    volume = []
    for key in speed_dict.keys():
        speed.append(speed_dict[key][:, 0])
    for key in volume_dict.keys():
        volume.append(volume_dict[key][:, 0])
    return rmse(candidate, test_input[0]), rmse(volume, test_volume[0]), rmse(speed, test_speed[0])


def train(convert_type, flow_type, number):
    test_input = np.load('flow/otd_test_' + convert_type + flow_type + '.npy')[number]
    test_volume = np.load('flow/volume_test_' + convert_type + flow_type + '.npy')[number]
    test_speed = np.load('flow/speed_test_' + convert_type + flow_type + '.npy')[number]
    candidates = []
    for i in range(1000):
        candidate = []
        for k in range(3):
            candidate.append(np.random.randint(1, 100, size=12))
        candidates.append(candidate)
    ans = 10000
    for i in range(1000):
        print(i)
        # get reward
        total = 0.0
        max_result = 10000
        max_candidate = []
        result = []
        for candidate in candidates:
            loss = get_loss(candidate, convert_type, test_speed)
            result.append(loss)
            total = total + loss * loss
            if loss < max_result:
                max_result = loss
                max_candidate = candidate
        # next generation
        print(i, max_result, len(candidates))
        # print(max_candidate)
        if max_result < ans:
            ans = max_result
        else:
            ans = ans
        children = []
        n = len(candidates) // 2
        for j in range(n):
            x = random.randint(0, len(candidates) - 1)
            y = random.randint(0, len(candidates) - 1)
            while y == x:
                y = random.randint(0, len(candidates) - 1)
            if result[x] < result[y]:
                children.append(copy.deepcopy(candidates[x]))
                children.append(mutate(candidates[x]))
            if result[y] <= result[x]:
                children.append(copy.deepcopy(candidates[y]))
                children.append(mutate(candidates[y]))
            # x = candidates[x]
            # y = candidates[y]
            candidates.pop(x)
            if y > x:
                candidates.pop(y - 1)
        # end
        del candidates
        candidates = children
    return output(max_candidate, convert_type, test_input, test_volume, test_speed)


if __name__ == "__main__":
    convert = ['rule_0.5']
    flow = ['random', 'increasing', 'decreasing', 'uniform', 'normal', 'poisson']
    # flow = ['poisson']
    loss = {}
    for c in convert:
        print(c)
        loss[c] = {}
        for f in flow:
            print(f)
            loss[c][f] = [c, f]
            loss_od_all = 0.0
            loss_volume_all = 0.0
            loss_speed_all = 0.0
            for i in range(10):
                loss_od, loss_volume, loss_speed = train(c, f, i)
                loss_od_all += loss_od
                loss_volume_all += loss_volume
                loss_speed_all += loss_speed
            loss_od_all = loss_od_all / 10
            loss_volume_all = loss_volume_all / 10
            loss_speed_all = loss_speed_all / 10
            loss[c][f].append(loss_od_all)
            loss[c][f].append(loss_volume_all)
            loss[c][f].append(loss_speed_all)

    with open('loss_GA.csv', 'w') as f:
        writer = csv.writer(f)
        for c in convert:
            for fl in flow:
                writer.writerow(loss[c][fl])

