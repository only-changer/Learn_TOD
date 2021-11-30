import time

from pretrain import Model as pretrain_model
from gen import Model as gen_model
import numpy as np
import json
import csv
import warnings
from multiprocessing import Process


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def train(c, f, n):
    print(c, f)
    pretrain = pretrain_model()
    pretrain.train('fc', c, f, n)
    loss_od = 0.0
    loss_volume = 0.0
    loss_speed = 0.0

    for i in range(1):
        print(i)
        gen = gen_model()
        real_od, pred_od, real_volume, pred_volume, real_speed, pred_speed, real_loss_tod, real_loss_volume, real_loss_speed = gen.train(
            'fc', c, f, i, n)

        loss_od += real_loss_tod
        loss_volume += real_loss_volume
        loss_speed += real_loss_speed
        print(rmse(real_od, pred_od), '_', rmse(real_volume, pred_volume), '_', rmse(real_speed, pred_speed), "_", real_loss_tod, real_loss_volume, real_loss_speed)
        with open('ab2' + c + f + str(n) + '.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(
                [rmse(real_od, pred_od), rmse(real_volume, pred_volume), rmse(real_speed, pred_speed), real_loss_tod, real_loss_volume, real_loss_speed])
    loss_od = loss_od / 10
    loss_volume = loss_volume / 10
    loss_speed = loss_speed / 10
    with open('ab2' + c + f + str(n) + '.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([loss_od, loss_volume, loss_speed])


if __name__ == '__main__':
    time_start = time.time()
    procs = []
    warnings.filterwarnings('ignore')
    convert = ['rule_0.5']
    flow = ['random']
    # flow = ['normal', 'poisson']
    for n in [3]:
        for c in convert:
            for f in flow:
                proc = Process(target=train, args=(c, f, n,))
                procs.append(proc)
                proc.start()
    for proc in procs:
        proc.join()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
