import numpy as np
import pandas as pd
import tensorflow as tf


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

    def build_network(self, volume, speed):
        tensor_road = volume
        with tf.variable_scope("volume2speed"):
            tensor_road_volume = tf.reshape(tensor_road, (-1, self.n_road * self.n_time_interval, 1))
            speed = tf.reshape(speed, (-1, self.n_road * self.n_time_interval, 1))
            hid = tf.layers.dropout(tf.nn.relu(tf.layers.dense(tensor_road_volume, self.units_size)), rate=0.3)
            out = tf.nn.sigmoid(tf.layers.dense(hid, 1)) * 15
        with tf.variable_scope("loss_v2s"):
            loss = tf.losses.mean_squared_error(out, speed)
        train_vars = tf.trainable_variables()
        v2s_vars = [var for var in train_vars if var.name.startswith('volume2speed')]
        v2s_optimizer = tf.train.AdamOptimizer(self.learning_rate_c).minimize(loss, var_list=v2s_vars)
        return loss, v2s_optimizer, tensor_road, out

    def train(self):
        df = pd.read_csv('/mnt/e/exp/curve.csv', header=None)
