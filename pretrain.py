import copy
import json
import math
import time
import warnings

# import cityflow
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from datetime import datetime
import os
import random
import re


class Model:

    def __init__(self):
        self.batch_size = 128
        self.units_size = 128
        self.learning_rate_b = 0.0005
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

    def build_network(self, noise_input, volume, speed, const, list_road_names, list_OD_names, network_type):
        OD = noise_input
        with tf.variable_scope("pretrain"):
            OD = OD / 5
            list_road = []
            with tf.variable_scope("odinit"):
                list_OD = []
                for od_ind in range(self.n_od):
                    list_OD.append(tf.gather(OD, od_ind, axis=1))
            if network_type == 'fc':
                for road_ind in range(self.n_road):
                    this_road_ODs = []
                    for od_ind in range(self.n_od):
                        this_road_ODs.append(self.dense_layer(self.dense_layer(list_OD[od_ind], self.units_size), 12))
                    list_road.append(tf.reshape(tf.math.add_n(this_road_ODs), (-1, 1, 12)))
            if network_type == 'bi_graph':
                list_od_road_segments = []
                for od_ind in range(self.n_od):
                    od_name = list_OD_names[od_ind]
                    od_road_segments = []
                    all_occur_pos = [m.start() for m in re.finditer(',', od_name)]
                    all_st_en = [-1] + all_occur_pos + [len(od_name)]
                    for ind in range(len(all_st_en) - 2):
                        od_road_segments.append(od_name[all_st_en[ind] + 1:all_st_en[ind + 2]])
                    list_od_road_segments.append(od_road_segments)
                for road_ind in range(self.n_road):
                    with tf.variable_scope("road" + str(road_ind)):
                        road_name = list_road_names[road_ind]
                        this_road_at_t = []
                        for t in range(self.n_time_interval):
                            this_road_ODs_contribute_at_t = []
                            for od_ind in range(self.n_od):
                                if road_name in list_od_road_segments[od_ind]:
                                    attention_weight = []  # attention weight
                                    for od_ind_inner in range(self.n_od):
                                        attention_weight.append(self.dense_layer(
                                            self.dense_layer(tf.reshape(list_OD[od_ind_inner], shape=(-1, 1, 12)),
                                                             int(self.units_size / 4), activation="sigmoid"),
                                            int(self.units_size / 4), activation="relu"))
                                        # filter_ex = tf.zeros([3, 1, 16])
                                        # cnn_in = tf.reshape(list_OD[od_ind_inner], shape=(-1, 1, 12))
                                        # cnn_hid1 = tf.layers.Conv1D(16, 3, activation="sigmoid", data_format="channels_first")(cnn_in)
                                        # cnn_hid2 = tf.layers.Conv1D(16, 3, activation="sigmoid", data_format="channels_first")(cnn_hid1)
                                        # cnn_hid3 = tf.layers.Conv1D(8, 3, activation="sigmoid", data_format="channels_first")(cnn_hid2)
                                        # attention_weight.append(tf.reshape(cnn_hid3, shape=(-1, 1, int(cnn_hid3.shape[1]) * int(cnn_hid3.shape[2]))))
                                    attention_weight_concat = tf.concat(attention_weight, axis=2)
                                    attention_weight_final = self.dense_layer(attention_weight_concat, 12,
                                                                              activation='sigmoid')
                                    # attention_weight_final = tf.nn.softmax(attention_weight_raw, axis=2)
                                    this_road_ODs_contribute_at_t.append(tf.reduce_sum(
                                        tf.math.multiply(attention_weight_final,
                                                         tf.reshape(list_OD[od_ind], shape=(-1, 1, 12))), axis=2,
                                        keepdims=True))
                            this_road_at_t.append(tf.reshape(tf.math.add_n(this_road_ODs_contribute_at_t), (-1, 1, 1)))
                        list_road.append(tf.reshape(tf.concat(this_road_at_t, axis=2), (-1, 1, 12)))
            tensor_road = tf.concat(list_road, axis=1)
            tensor_road = tensor_road * 5
        # calculate loss
        with tf.variable_scope("volume2speed"):
            tensor_road_volume = tf.reshape(tensor_road, (-1, self.n_road * self.n_time_interval, 1))
            speed = tf.reshape(speed, (-1, self.n_road * self.n_time_interval, 1))
            hid = tf.layers.dropout(tf.nn.sigmoid(tf.layers.dense(tensor_road_volume, self.units_size)), rate=0.3)
            out = tf.nn.sigmoid(tf.layers.dense(hid, 1)) * 15
            # out = []
            # for road_ind in range(self.n_road):
            #     road_volume = tf.gather(tensor_road, road_ind, axis=1)
            #     hid = tf.layers.dropout(tf.nn.sigmoid(tf.layers.dense(road_volume, self.units_size)), rate=0.3)
            #     out.append(tf.reshape(tf.nn.sigmoid(tf.layers.dense(hid, self.n_time_interval)) * 15,
            #                           (-1, 1, self.n_time_interval)))
            # out = tf.concat(out, axis=1)
        with tf.variable_scope("loss_o2v"):
            loss1 = tf.losses.mean_squared_error(tensor_road, volume)
        with tf.variable_scope("loss_v2s"):
            loss2 = tf.losses.mean_squared_error(out, speed)
        train_vars = tf.trainable_variables()
        gen_vars = [var for var in train_vars if var.name.startswith('pretrain')]
        gen_optimizer = tf.train.AdamOptimizer(self.learning_rate_b).minimize(loss1, var_list=gen_vars)
        v2s_vars = [var for var in train_vars if var.name.startswith('volume2speed')]
        v2s_optimizer = tf.train.AdamOptimizer(self.learning_rate_c).minimize(loss2, var_list=v2s_vars)
        return OD, loss1, loss2, gen_optimizer, v2s_optimizer, tensor_road, out

    def train(self, network_type, convert_type, flow_type, n_otd):
        self.n_od = n_otd
        warnings.filterwarnings('ignore')
        tf.reset_default_graph()
        list_road_names = ["1,3", "3,4", "4,2", "2,1"]
        # list_road_names = ["1,5", "5,3", "3,6", "6,4", "4,7", "7,2", "2,8", "8,1"]
        list_OD_names = ["1,3,4", "3,4,2", "4,2,1", "1,5,3,6,4",
                         "3,6,4,7,2",
                         "4,7,2,8,1",
                         "2,1,3",
                         "1,3",
                         "3,4",
                         "4,2",
                         "2,1",
                         "1,3,4,2",
                         "3,4,2,1",
                         "4,2,1,3",
                         "2,1,3,4",
                         "1,3,4,2,1",
                         "3,4,2,1,3",
                         "4,2,1,3,4",
                         "2,1,3,4,2"]
        list_OD_names = list_OD_names[0:n_otd]
        OD_lists = []
        volumes = []
        speeds = []
        test_input = []
        test_volumes = []
        test_speeds = []
        test_const = []
        input = []
        const = []

        for i in range(2100):
            # print(i)
            simulator = Simulator()
            ODs = ["1,3,4", "3,4,2", "4,2,1"]
            # ODs = ["4,1", "2,1", "3,1",
            #        "1,5,3,6,4",
            #        "3,6,4,7,2",
            #        "4,7,2,8,1",
            #        "2,1,3",
            #        "1,3",
            #        "3,4",
            #        "4,2",
            #        "2,1",
            #        "1,3,4,2",
            #        "3,4,2,1",
            #        "4,2,1,3",
            #        "2,1,3,4",
            #        "1,3,4,2,1",
            #        "3,4,2,1,3",
            #        "4,2,1,3,4",
            #        "2,1,3,4,2"]
            OD_list = {}
            for j in range(n_otd):
                OD_list[ODs[j]] = np.random.randint(1, 100, size=(12, 1))
            volume_list, speed_list = simulator.convert(OD_list, convert_type)
            volume = np.concatenate([
                volume_list["1,3"].reshape((1, 1, self.n_time_interval)),
                volume_list["3,4"].reshape((1, 1, self.n_time_interval)),
                volume_list["4,2"].reshape((1, 1, self.n_time_interval)),
                volume_list["2,1"].reshape((1, 1, self.n_time_interval))
                # volume_list["1,5"].reshape((1, 1, self.n_time_interval)),
                # volume_list["3,6"].reshape((1, 1, self.n_time_interval)),
                # volume_list["4,7"].reshape((1, 1, self.n_time_interval)),
                # volume_list["2,8"].reshape((1, 1, self.n_time_interval)),
                # volume_list["5,3"].reshape((1, 1, self.n_time_interval)),
                # volume_list["6,4"].reshape((1, 1, self.n_time_interval)),
                # volume_list["7,2"].reshape((1, 1, self.n_time_interval)),
                # volume_list["8,1"].reshape((1, 1, self.n_time_interval))
            ],
                axis=1)
            speed = np.concatenate([
                speed_list["1,3"].reshape((1, 1, self.n_time_interval)),
                speed_list["3,4"].reshape((1, 1, self.n_time_interval)),
                speed_list["4,2"].reshape((1, 1, self.n_time_interval)),
                speed_list["2,1"].reshape((1, 1, self.n_time_interval))
                # speed_list["1,5"].reshape((1, 1, self.n_time_interval)),
                # speed_list["3,6"].reshape((1, 1, self.n_time_interval)),
                # speed_list["4,7"].reshape((1, 1, self.n_time_interval)),
                # speed_list["2,8"].reshape((1, 1, self.n_time_interval)),
                # speed_list["5,3"].reshape((1, 1, self.n_time_interval)),
                # speed_list["6,4"].reshape((1, 1, self.n_time_interval)),
                # speed_list["7,2"].reshape((1, 1, self.n_time_interval)),
                # speed_list["8,1"].reshape((1, 1, self.n_time_interval))
            ],
                axis=1)
            if i >= 2000:
                test_const.append(0.0)
                od_raw = []
                for k in range(self.n_od):
                    od_raw.append(OD_list[ODs[k]].reshape((1, 1, self.n_time_interval)))
                test_input.append(np.concatenate(od_raw, axis=1))
                test_volumes.append(volume)
                test_speeds.append(speed)
                # continue
            const.append(0.0)
            od_raw = []
            for k in range(self.n_od):
                od_raw.append(OD_list[ODs[k]].reshape((1, 1, self.n_time_interval)))
            input.append(np.concatenate([od_raw], axis=1))
            volumes.append(volume)
            speeds.append(speed)

        # np.save('otd_train' + str(len(input)) + '.npy', input)
        # np.save('flow/otd_test_cityflowrandom.npy', test_input)
        # np.save('cityflow_train' + str(len(volumes)) + '.npy', volumes)
        # np.save('flow/volume_test_cityflowrandom.npy', test_volumes)
        # np.save('flow/speed_test_cityflowrandom.npy', test_speeds)

        # input = np.load('flow/otd_train_' + convert_type + flow_type + '.npy')
        # test_input = np.load('flow/otd_test_' + convert_type + flow_type + '.npy')
        # volumes = np.load('flow/volume_train_' + convert_type + flow_type + '.npy')
        # test_volumes = np.load('flow/volume_test_' + convert_type + flow_type + '.npy')
        # speeds = np.load('flow/speed_train_' + convert_type + flow_type + '.npy')
        # test_speeds = np.load('flow/speed_test_' + convert_type + flow_type + '.npy')
        if convert_type == 'cityflow':
            simulator = Simulator()
            speeds = simulator.v2s(volumes)
            test_speeds = simulator.v2s(test_volumes)
        input = np.reshape(input, (-1, self.n_od, self.n_time_interval))
        const = np.reshape(const, (-1, 1))
        volumes = np.reshape(volumes, (-1, self.n_road, self.n_time_interval))
        speeds = np.reshape(speeds, (-1, self.n_road, self.n_time_interval))
        test_input = np.reshape(test_input, (-1, self.n_od, self.n_time_interval))
        test_const = np.reshape(test_const, (-1, 1))
        test_volumes = np.reshape(test_volumes, (-1, self.n_road, self.n_time_interval))
        test_speeds = np.reshape(test_speeds, (-1, self.n_road, self.n_time_interval))
        noise_input = tf.placeholder(tf.float32, [None, self.n_od, self.n_time_interval], name="noise_input")
        volume_label = tf.placeholder(tf.float32, [None, self.n_road, self.n_time_interval], name="volume_label")
        speed_label = tf.placeholder(tf.float32, [None, self.n_road, self.n_time_interval], name="speed_label")
        const_ph = tf.placeholder(tf.float32, [None, 1])
        output, loss1, loss2, optimizer1, optimizer2, volume_pred, speed_pred = self.build_network(noise_input,
                                                                                                   volume_label,
                                                                                                   speed_label,
                                                                                                   const_ph,
                                                                                                   list_road_names,
                                                                                                   list_OD_names,
                                                                                                   network_type)
        step = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            min_loss = 10000
            min_loss_step = 0
            for epo in range(self.epoch):
                # noise = np.random.uniform(-1, 1, size=(2000, self.n_od, self.n_time_interval))
                step += 1
                sess.run(optimizer1,
                         feed_dict={noise_input: input, volume_label: volumes, speed_label: speeds, const_ph: const})
                train_loss_volume = sess.run(loss1,
                                             feed_dict={noise_input: input, volume_label: volumes, speed_label: speeds,
                                                        const_ph: const})
                test_loss_volume = sess.run(loss1, feed_dict={noise_input: test_input, volume_label: test_volumes,
                                                              speed_label: test_speeds,
                                                              const_ph: test_const})
                # with open('pretrain_log/' + network_type + '_' + convert_type + '_' + flow_type + '_pretain_b.txt',
                #           'a') as f:
                print('step', step, ' train_loss', train_loss_volume, '    test_loss', test_loss_volume)
                if test_loss_volume < min_loss - 0.01:
                    min_loss = test_loss_volume
                    min_loss_step = step
                if step - min_loss_step > self.early_stop and step > 500:
                    break

            step = 0
            min_loss = 10000
            min_loss_step = 0
            for epo in range(self.epoch):
                step += 1
                sess.run(optimizer2,
                         feed_dict={noise_input: input, volume_label: volumes, speed_label: speeds, const_ph: const})
                train_loss_speed = sess.run(loss2,
                                            feed_dict={noise_input: input, volume_label: volumes, speed_label: speeds,
                                                       const_ph: const})
                test_loss_speed = sess.run(loss2, feed_dict={noise_input: test_input, volume_label: test_volumes,
                                                             speed_label: test_speeds,
                                                             const_ph: test_const})
                # with open('pretrain_log/' + network_type + '_' + convert_type + '_' + flow_type + '_pretain_c.txt',
                #           'a') as f:
                print('step', step, ' train_loss', train_loss_speed, '    test_loss', test_loss_speed)
                if test_loss_speed < min_loss - 0.01:
                    min_loss = test_loss_speed
                    min_loss_step = step
                if step - min_loss_step > self.early_stop and step > 500:
                    break

            saver = tf.train.Saver()
            saver.save(sess, "./model/" + network_type + '_' + convert_type + '_' + flow_type + str(
                self.n_od) + str(self.n_road) + "_volume2speed_time/model")
            # print("od: ", np.reshape(test_input[0], (-1, self.n_od, self.n_time_interval)))
            # print("od pred:", sess.run(output, feed_dict={
            #     noise_input: np.reshape(test_input[0], (-1, self.n_od, self.n_time_interval)),
            #     volume_label: np.reshape(test_volumes[0], (-1, self.n_road, self.n_time_interval)),
            #     speed_label: np.reshape(test_speeds[0], (-1, self.n_road, self.n_time_interval)),
            #     const_ph: test_const[0:1]}))
            # print("volume: ", np.reshape(test_volumes[0], (-1, self.n_road, self.n_time_interval)))
            # print("volume pred: ", sess.run(volume_pred, feed_dict={
            #     noise_input: np.reshape(test_input[0], (-1, self.n_od, self.n_time_interval)),
            #     volume_label: np.reshape(test_volumes[0], (-1, self.n_road, self.n_time_interval)),
            #     speed_label: np.reshape(test_speeds[0], (-1, self.n_road, self.n_time_interval)),
            #     const_ph: test_const[0:1]}))
            # print("speed: ", np.reshape(test_speeds[0], (-1, self.n_road, self.n_time_interval)))
            # print("speed pred: ", sess.run(speed_pred, feed_dict={
            #     noise_input: np.reshape(test_input[0], (-1, self.n_od, self.n_time_interval)),
            #     volume_label: np.reshape(test_volumes[0], (-1, self.n_road, self.n_time_interval)),
            #     speed_label: np.reshape(test_speeds[0], (-1, self.n_road, self.n_time_interval)),
            #     const_ph: test_const[0:1]}))


class Simulator:
    def __init__(self):
        self.vec_size = 12
        self.intersection_size = 4
        self.route = []
        self.OTD = []
        self.cityflow_route = []
        self.road = {
            '1,3': np.zeros((self.vec_size, 1)), '3,4': np.zeros((self.vec_size, 1)),
            '4,2': np.zeros((self.vec_size, 1)), '2,1': np.zeros((self.vec_size, 1))
            # '1,5': np.zeros((self.vec_size, 1)), '3,6': np.zeros((self.vec_size, 1)),
            #          '4,7': np.zeros((self.vec_size, 1)), '2,8': np.zeros((self.vec_size, 1)),
            #          '5,3': np.zeros((self.vec_size, 1)), '6,4': np.zeros((self.vec_size, 1)),
            #          '7,2': np.zeros((self.vec_size, 1)), '8,1': np.zeros((self.vec_size, 1))
        }
        self.speed = {'1,3': np.zeros((self.vec_size, 1)), '3,4': np.zeros((self.vec_size, 1)),
                      '4,2': np.zeros((self.vec_size, 1)), '2,1': np.zeros((self.vec_size, 1))}

    def v2s(self, volume):
        speed = copy.deepcopy(volume)
        for i in range(len(speed)):
            for j in range(4):
                for k in range(12):
                    speed[i][0][j][k] = self.volume2speed(speed[i][0][j][k])
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
                        if len(self.route[j]) >= 2:
                            road = self.route[j][1]
                            self.road[road][i + 1] = self.road[road][i + 1] + self.OTD[j][i] * 0.5
                    if i <= self.vec_size - 3:
                        if len(self.route[j]) >= 2:
                            road = self.route[j][1]
                            self.road[road][i + 2] = self.road[road][i + 2] + self.OTD[j][i] * 0.5
                        if len(self.route[j]) >= 3:
                            road = self.route[j][2]
                            self.road[road][i + 2] = self.road[road][i + 2] + self.OTD[j][i] * 0.5
                    if i <= self.vec_size - 4:
                        if len(self.route[j]) >= 3:
                            road = self.route[j][2]
                            self.road[road][i + 2] = self.road[road][i + 2] + self.OTD[j][i] * 0.5
                        if len(self.route[j]) >= 4:
                            road = self.route[j][3]
                            self.road[road][i + 2] = self.road[road][i + 2] + self.OTD[j][i] * 0.5
                    if i <= self.vec_size - 5:
                        if len(self.route[j]) >= 4:
                            road = self.route[j][3]
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


if __name__ == '__main__':
    model = Model()
    model.train('bi_graph', 'rule_0.5', 'random')
