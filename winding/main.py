import argparse
import os
import shutil
import sys
import traceback
from time import time

import numpy as np
import pandas as pd
import torch
from tensorboard_logger import configure, log_value

sys.path.append(os.path.dirname(sys.path[0]))

from taho.model import (
    MIMO,
    GRUCell,
    HOARNNCell,
    HOGRUCell,
    IncrHOARNNCell,
    IncrHOGRUCell,
)
from taho.train import EpochTrainer
from taho.util import SimpleLogger, predict, prediction_error, show_data, t2np

GPU = torch.cuda.is_available()


"""
potentially varying input parameters
"""
parser = argparse.ArgumentParser(
    description="Models for Continuous Stirred Tank dataset"
)

# model definition
methods = """
set up model
- model:
    GRU (compensated GRU to avoid linear increase of state; has standard GRU as special case for Euler scheme and equidistant data)
    GRUinc (incremental GRU, for baseline only)
- time_aware:
    no: ignore uneven spacing: for GRU use original GRU implementation; ignore 'scheme' variable
    input: use normalized next interval size as extra input feature
    variable: time-aware implementation
"""


parser.add_argument(
    "--time_aware",
    type=str,
    default="variable",
    choices=["no", "input", "variable"],
    help=methods,
)
parser.add_argument(
    "--model", type=str, default="GRU", choices=["GRU", "GRUinc", "ARNN", "ARNNinc"]
)
parser.add_argument(
    "--interpol", type=str, default="constant", choices=["constant", "linear"]
)

parser.add_argument(
    "--gamma", type=float, default=1.0, help="diffusion parameter ARNN model"
)
parser.add_argument(
    "--step_size",
    type=float,
    default=1.0,
    help="fixed step size parameter in the ARNN model",
)


# data
parser.add_argument(
    "--missing",
    type=float,
    default=0.0,
    help="fraction of missing samples (0.0 or 0.5)",
)

# model architecture
parser.add_argument("--k_state", type=int, default=20, help="dimension of hidden state")

# in case method == 'variable'
RKchoices = ["Euler", "Midpoint", "Kutta3", "RK4"]
parser.add_argument(
    "--scheme",
    type=str,
    default="Euler",
    choices=RKchoices,
    help="Runge-Kutta training scheme",
)

# training
parser.add_argument(
    "--batch_size", type=int, default=16, help="batch size"
)  # Original default value: 512
parser.add_argument(
    "--epochs", type=int, default=1000, help="Number of epochs"
)  # Original default value: 4000
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--bptt", type=int, default=20, help="bptt")
parser.add_argument("--dropout", type=float, default=0.0, help="drop prob")
parser.add_argument("--l2", type=float, default=0.0, help="L2 regularization")


# admin
parser.add_argument(
    "--save", type=str, default="results", help="experiment logging folder"
)
parser.add_argument(
    "--eval_epochs", type=int, default=20, help="validation every so many epochs"
)
parser.add_argument("--seed", type=int, default=0, help="random seed")

# during development
parser.add_argument(
    "--reset",
    action="store_true",
    help="reset even if same experiment already finished",
)


paras = parser.parse_args()

hard_reset = paras.reset
# if paras.save already exists and contains log.txt:
# reset if not finished, or if hard_reset
log_file = os.path.join(paras.save, "log.txt")
if os.path.isfile(log_file):
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        completed = "Finished" in content
        if completed and not hard_reset:
            print("Exit; already completed and no hard reset asked.")
            sys.exit()  # do not overwrite folder with current experiment
        else:  # reset folder
            shutil.rmtree(paras.save, ignore_errors=True)


# setup logging
logging = SimpleLogger(log_file)  # log to file
configure(paras.save)  # tensorboard logging
logging("Args: {}".format(paras))


"""
fixed input parameters
"""
frac_dev = 15 / 100
frac_test = 15 / 100

GPU = torch.cuda.is_available()
logging("Using GPU?", GPU)

# set random seed for reproducibility
torch.manual_seed(paras.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(paras.seed)
np.random.seed(paras.seed)


"""
Load data
"""
## Original version of loading data
# data = np.loadtxt('winding\data\winding_missing_prob_0.00.dat')
# t = np.expand_dims(data[:, 0], axis=1)  # (Nsamples, 1)
# X = data[:, 1:6]  # (Nsamples, 5)
# Y = data[:, 6:8]  # (Nsamples, 2)
# k_in = X.shape[1]
# k_out = Y.shape[1]

# dt = np.expand_dims(data[:, 8], axis=1)  # (Nsamples, 1) # dt: sample rate
# logging('loaded data, \nX', X.shape, '\nY', Y.shape, '\nt', t.shape, '\ndt', dt.shape,
#         '\ntime intervals dt between %.3f and %.3f wide (%.3f on average).'%(np.min(dt), np.max(dt), np.mean(dt)))

# ## HomeRLer's Version of loading data in a single trajectory
# data = pd.read_csv("winding\data\odom-19-02-2024-run6.csv", index_col=0).to_numpy()
# t = np.expand_dims(data[:, 0], axis=1)  # (Nsamples, 1)

# X_1 = data[:, 1:11]  # (Nsamples, 10)
# X_2 = data[:, 26:34]  # (Nsamples, 8)
# X = np.hstack((X_1, X_2))  # (Nsamples, 18) x,y,z,qx,qy,qz,qw,bu,bv,bw,pwm1-8
# X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))
# Y = data[:, 11:14]  # (Nsamples, 3)
# k_in = X.shape[1]
# k_out = Y.shape[1]
# sample_rate = 0.1
# dt = sample_rate * np.ones(
#     (X.shape[0], 1)
# )  # (Nsamples, 1) # In out version, assume sample rate is 0.1
# logging(
#     "loaded data, \nX",
#     X.shape,
#     "\nY",
#     Y.shape,
#     "\nt",
#     t.shape,
#     "\ndt",
#     dt.shape,
#     "\ntime intervals dt between %.3f and %.3f wide (%.3f on average)."
#     % (np.min(dt), np.max(dt), np.mean(dt)),
# )
# N = X.shape[0]  # number of samples in total


# ## Load data from the discrete dataset
dataset_dir = "dataset"
files_list = os.listdir(dataset_dir)
file_counts = 0
X: list[np.ndarray] = []  # (file_nums, sample_nums, feature_nums)
Y: list[np.ndarray] = []  # (file_nums, sample_nums, feature_nums)
t: list[np.ndarray] = []  # (file_nums, sample_nums, 1)
sample_num_list: list = []  # (file_nums, 1)
for file in files_list:
    file_dir = dataset_dir + "/" + file
    data = pd.read_csv(file_dir).to_numpy()
    file_counts += 1
    X1 = data[:, 1:14] # x,y,z,qx,qy,qz,qw,bu,bv,bw,bp,bq,br
    X2 = data[:, 26:34] # pwm1-8
    X_np = np.hstack((X1, X2))
    X_np = (X_np - np.min(X_np, axis=0)) / (np.max(X_np, axis=0) - np.min(X_np, axis=0))
    X.append(X_np)  # (sample_nums, 10)
    Y_np = data[:, 11:14] # bp,bq,br
    Y_np = (Y_np - np.min(Y_np, axis=0)) / (np.max(Y_np, axis=0) - np.min(Y_np, axis=0))
    Y.append(Y_np)  # (sample_nums, 3)
    t.append(data[:, 0])  # (sample_nums, 1)
    sample_num_list.append(data.shape[0])  # (file_nums, 1)

N = sum([subdataset_X.shape[0] for subdataset_X in X])
sample_rate = 0.1
sample_num_list: np.ndarray = np.array(sample_num_list)
dt = sample_rate * np.ones((N, 1))  # (total_sample_nums, 1)
k_in = X[0].shape[1]
k_out = Y[0].shape[1]
logging(
    "Data has loaded, \nloaded subdatasets numbers,",
    file_counts,
    "\nX feature numbers:",
    k_in,
    "\nY feature numbers:",
    k_out,
    "\ntotal sample numbers:",
    N,
    "\nsample rate:",
    sample_rate,
    "\nsample nums of different csvs:",
    sample_num_list,
)


Ndev_num_list = frac_dev * sample_num_list
Ndev_num_list = Ndev_num_list.astype(int).tolist()
Ntest_num_list = frac_test * sample_num_list
Ntest_num_list = Ntest_num_list.astype(int).tolist()
Ntrain_num_list = sample_num_list - Ndev_num_list - Ntest_num_list

logging(
    "Totally, there are {} samples for training, then {} samples for development and {} samples for testing".format(
        np.sum(Ntrain_num_list), np.sum(Ndev_num_list), np.sum(Ntest_num_list)
    )
)


# Ndev = int(frac_dev * N)
# Ntest = int(frac_test * N)
# Ntrain = N - Ntest - Ndev

# logging(
#     "first {} for training, then {} for development and {} for testing".format(
#         Ntrain, Ndev, Ntest
#     )
# )

"""
evaluation function
RRSE error
"""


Xtrain_list: list = []
Ytrain_list: list = []
ttrain_list: list = []
dttrain_list: list = []

Xdev_list: list = []
Ydev_list: list = []
tdev_list: list = []
dtdev_list: list = []

Xtest_list: list = []
Ytest_list: list = []
ttest_list: list = []
dttest_list: list = []

for (i, Ndev), Ntrain in zip(enumerate(Ndev_num_list), Ntrain_num_list):
    Xtrain_list.append(X[i][:Ntrain, :])
    Ytrain_list.append(Y[i][1 : Ntrain + 1, :])
    ttrain_list.append(t[i][1 : Ntrain + 1])
    dttrain_list.append(dt[:Ntrain])

    Xdev_list.append(X[i][Ntrain : Ntrain + Ndev, :])
    Ydev_list.append(Y[i][Ntrain + 1 : Ntrain + Ndev + 1, :])
    tdev_list.append(t[i][Ntrain + 1 : Ntrain + Ndev + 1])
    dtdev_list.append(dt[Ntrain : Ntrain + Ndev])

    Xtest_list.append(X[i][Ntrain + Ndev : -1, :])
    Ytest_list.append(Y[i][Ntrain + Ndev + 1 :, :])
    ttest_list.append(t[i][Ntrain + Ndev + 1 :])
    dttest_list.append(dt[Ntrain + Ndev : -1])


"""
- model:
    GRU (compensated GRU to avoid linear increase of state; has standard GRU as special case for Euler scheme and equidistant data)
    GRUinc (incremental GRU, for baseline only)
- time_aware:
    no: ignore uneven spacing: for GRU use original GRU implementation
    input: use normalized next interval size as extra input feature
    variable: time-aware implementation
"""

# time_aware options

# if paras.time_aware == "input":
#     # expand X matrices with additional input feature, i.e., normalized duration dt to next sample
#     dt_mean, dt_std = np.mean(dttrain), np.std(dttrain)
#     dttrain_n = (dttrain - dt_mean) / dt_std
#     dtdev_n = (dtdev - dt_mean) / dt_std
#     dttest_n = (dttest - dt_mean) / dt_std

#     Xtrain = np.concatenate([Xtrain, dttrain_n], axis=1)
#     Xdev = np.concatenate([Xdev, dtdev_n], axis=1)
#     Xtest = np.concatenate([Xtest, dttest_n], axis=1)

#     k_in += 1

# if paras.time_aware == "no" or paras.time_aware == "input":
#     # in case 'input': variable intervals already in input X;
#     # now set actual time intervals to 1 (else same effect as time_aware == 'variable')
#     dttrain = np.ones(dttrain.shape)
#     dtdev = np.ones(dtdev.shape)
#     dttest = np.ones(dttest.shape)

# set model:
if paras.model == "GRU":
    cell_factory = GRUCell if paras.time_aware == "no" else HOGRUCell
elif paras.model == "GRUinc":
    cell_factory = IncrHOGRUCell
elif paras.model == "ARNN":
    cell_factory = HOARNNCell
elif paras.model == "ARNNinc":
    cell_factory = IncrHOARNNCell
else:
    raise NotImplementedError("unknown model type " + paras.model)

dt_mean = np.mean(dttrain_list[0])
model = MIMO(
    k_in,
    k_out,
    paras.k_state,
    dropout=paras.dropout,
    cell_factory=cell_factory,
    meandt=dt_mean,
    train_scheme=paras.scheme,
    eval_scheme=paras.scheme,
    gamma=paras.gamma,
    step_size=paras.step_size,
    interpol=paras.interpol,
)


if GPU:
    model = model.cuda()

params = sum(
    [np.prod(p.size()) for p in model.parameters()]
)  # the total number of parameters
logging(
    "\nModel %s (time_aware: %s, scheme %s) with %d trainable parameters"
    % (paras.model, paras.time_aware, paras.scheme, params)
)
for n, p in model.named_parameters():
    p_params = np.prod(p.size())
    print("\t%s\t%d (cuda: %s)" % (n, p_params, str(p.is_cuda)))

logging("Architecture: ", model)
log_value("model/params", params, 0)

optimizer = torch.optim.Adam(model.parameters(), lr=paras.lr, weight_decay=paras.l2)


# prepare tensors for evaluation
# 都转成行向量
Xtrain_tn_list: list[torch.Tensor] = []
Ytrain_tn_list: list[torch.Tensor] = []
ttrain_tn_list: list[torch.Tensor] = []
dttrain_tn_list: list[torch.Tensor] = []

Xdev_tn_list: list[torch.Tensor] = []
Ydev_tn_list: list[torch.Tensor] = []
tdev_tn_list: list[torch.Tensor] = []
dtdev_tn_list: list[torch.Tensor] = []

Xtest_tn_list: list[torch.Tensor] = []
Ytest_tn_list: list[torch.Tensor] = []
ttest_tn_list: list[torch.Tensor] = []
dttest_tn_list: list[torch.Tensor] = []

for i in range(len(Xtrain_list)):
    Xtrain_tn_list.append(torch.tensor(Xtrain_list[i], dtype=torch.float).unsqueeze(0))
    Ytrain_tn_list.append(torch.tensor(Ytrain_list[i], dtype=torch.float).unsqueeze(0))
    ttrain_tn_list.append(torch.tensor(ttrain_list[i], dtype=torch.float).unsqueeze(0))
    dttrain_tn_list.append(
        torch.tensor(dttrain_list[i], dtype=torch.float).unsqueeze(0)
    )

    Xdev_tn_list.append(torch.tensor(Xdev_list[i], dtype=torch.float).unsqueeze(0))
    Ydev_tn_list.append(torch.tensor(Ydev_list[i], dtype=torch.float).unsqueeze(0))
    tdev_tn_list.append(torch.tensor(tdev_list[i], dtype=torch.float).unsqueeze(0))
    dtdev_tn_list.append(torch.tensor(dtdev_list[i], dtype=torch.float).unsqueeze(0))

    Xtest_tn_list.append(torch.tensor(Xtest_list[i], dtype=torch.float).unsqueeze(0))
    Ytest_tn_list.append(torch.tensor(Ytest_list[i], dtype=torch.float).unsqueeze(0))
    ttest_tn_list.append(torch.tensor(ttest_list[i], dtype=torch.float).unsqueeze(0))
    dttest_tn_list.append(torch.tensor(dttest_list[i], dtype=torch.float).unsqueeze(0))

if GPU:
    for i in range(len(Xtrain_tn_list)):
        Xtrain_tn_list[i] = Xtrain_tn_list[i].cuda()
        Ytrain_tn_list[i] = Ytrain_tn_list[i].cuda()
        ttrain_tn_list[i] = ttrain_tn_list[i].cuda()
        dttrain_tn_list[i] = dttrain_tn_list[i].cuda()

        Xdev_tn_list[i] = Xdev_tn_list[i].cuda()
        Ydev_tn_list[i] = Ydev_tn_list[i].cuda()
        tdev_tn_list[i] = tdev_tn_list[i].cuda()
        dtdev_tn_list[i] = dtdev_tn_list[i].cuda()

        Xtest_tn_list[i] = Xtest_tn_list[i].cuda()
        Ytest_tn_list[i] = Ytest_tn_list[i].cuda()
        ttest_tn_list[i] = ttest_tn_list[i].cuda()
        dttest_tn_list[i] = dttest_tn_list[i].cuda()


trainer = EpochTrainer(
    model,
    optimizer,
    paras.epochs,
    Xtrain_list,
    Ytrain_list,
    dttrain_list,
    batch_size=paras.batch_size,
    gpu=GPU,
    bptt=paras.bptt,
)  # dttrain ignored for all but 'variable' methods

t00 = time()

best_dev_error = 1.0e5
best_dev_epoch = 0
error_test = -1

max_epochs_no_decrease = 1000

try:  # catch error and redirect to logger
    for epoch in range(1, paras.epochs + 1):
        # train 1 epoch
        ave_mse_train = trainer(epoch)

        if epoch % paras.eval_epochs == 0:
            with torch.no_grad():
                model.eval()
                # (1) forecast on train data steps

                Y_train_pred_list, h_train_pred_list, error_train_list = predict(
                    Xtrain_tn_list,
                    Ytrain_tn_list,
                    dttrain_tn_list,
                    model,
                    is_mse_list=False,
                )
                ave_error_train = np.mean(np.array(error_train_list))

                # (2) forecast on dev data
                Y_dev_pred_list, h_dev_pred_list, error_dev_list, mse_dev_list = (
                    predict(
                        Xdev_tn_list,
                        Ydev_tn_list,
                        dtdev_tn_list,
                        model,
                    )
                )

                ave_mse_dev = np.mean(np.array(mse_dev_list))
                ave_error_dev = np.mean(np.array(error_dev_list))

                # report evaluation results
                log_value("train/average_mse", ave_mse_train, epoch)
                log_value("train/average_error", ave_error_train, epoch)
                log_value("dev/average_loss", ave_mse_dev, epoch)
                log_value("dev/average_error", ave_error_dev, epoch)

                logging(
                    "epoch %04d | ave_loss %.3f (train), %.3f (dev) | ave_error %.3f (train), %.3f (dev) | tt %.2fmin"
                    % (
                        epoch,
                        ave_mse_train,
                        ave_mse_dev,
                        ave_error_train,
                        ave_error_dev,
                        (time() - t00) / 60.0,
                    )
                )
                for i in range(len(Xtrain_list)):
                    show_data(
                        ttrain_list[i],
                        Ytrain_list[i],
                        t2np(Y_train_pred_list[i]),
                        paras.save + "/train",
                        "current_train_results of %dth trajectory" % i,
                        msg="train results (train error %.3f) at iter %d"
                        % (error_train_list[i], epoch),
                    )
                    show_data(
                        tdev_list[i],
                        Ydev_list[i],
                        t2np(Y_dev_pred_list[i]),
                        paras.save + "/dev",
                        "current_dev_results of %dth trajectory" % i,
                        msg="dev results (dev error %.3f) at iter %d"
                        % (error_dev_list[i], epoch),
                    )

                # update best dev model
                if ave_error_dev < best_dev_error:
                    best_dev_error = ave_error_dev
                    best_dev_epoch = epoch
                    log_value("dev/best_error", best_dev_error, epoch)

                    Y_test_pred_list, h_test_pred_list, error_test_list = predict(
                        Xtest_tn_list,
                        Ytest_tn_list,
                        dttest_tn_list,
                        model,
                        is_mse_list=False,
                    )


                    for i in range(len(Xdev_list)):
                        show_data(
                            tdev_list[i],
                            Ydev_list[i],
                            t2np(Y_dev_pred_list[i]),
                            paras.save + "/best_dev/dev",
                            "best_dev_dev_results of %dth trajectory" % i,
                            msg="dev results (dev error %.3f) at iter %d"
                            % (error_dev_list[i], epoch),
                        )
                        show_data(
                            ttest_list[i],
                            Ytest_list[i],
                            t2np(Y_test_pred_list[i]),
                            paras.save + "/best_dev/test",
                            "best_dev_test_results of %dth trajectory" % i,
                            msg="test results (test error %.3f) at iter %d (=best dev)"
                            % (error_test_list[i], epoch),
                        )

                        log_value("test/corresp_error", error_test_list, epoch)
                        logging("new best dev average error %.3f" % best_dev_error)

                        # make figure of best model on train, dev and test set for debugging

                        # save model
                        # torch.save(model.state_dict(), os.path.join(paras.save, 'best_dev_model_state_dict.pt'))
                        torch.save(model, os.path.join(paras.save, "best_dev_model.pt"))

                        # save dev and test predictions of best dev model
                        # pickle.dump(
                        #     {
                        #         "t_dev": tdev,
                        #         "y_target_dev": Ydev,
                        #         "y_pred_dev": t2np(Ydev_pred),
                        #         "t_test": ttest,
                        #         "y_target_test": Ytest,
                        #         "y_pred_test": t2np(Ytest_pred),
                        #     },
                        #     open(os.path.join(paras.save, "data4figs.pkl"), "wb"),
                        # )

                elif epoch - best_dev_epoch > max_epochs_no_decrease:
                    logging(
                        "Development error did not decrease over %d epochs -- quitting."
                        % max_epochs_no_decrease
                    )
                    break

    log_value("finished/best_dev_error", best_dev_error, 0)
    log_value("finished/corresp_test_error", error_test_list, 0)

    logging(
        "Finished: best dev error",
        best_dev_error,
        "at epoch",
        best_dev_epoch,
        "with corresp. test error",
        error_test_list,
    )


except:
    var = traceback.format_exc()
    logging(var)
