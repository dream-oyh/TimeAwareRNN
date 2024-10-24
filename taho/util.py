import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

plt.switch_backend("agg")


class SimpleLogger(object):
    def __init__(self, f, header="#logger output"):
        dir = os.path.dirname(f)
        # print('test dir', dir, 'from', f)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(f, "w") as fID:
            fID.write("%s\n" % header)
        self.f = f

    def __call__(self, *args):
        # standard output
        print(*args)
        # log to file
        try:
            with open(self.f, "a") as fID:
                fID.write(" ".join(str(a) for a in args) + "\n")
        except:
            print("Warning: could not log to", self.f)


def show_data(t, target, pred, folder, tag, msg=""):
    plt.figure(1)
    maxv = np.max(target)
    minv = np.min(target)
    view = maxv - minv

    # linear
    n = target.shape[1]
    for i in range(n):
        ax_i = plt.subplot(n, 1, i + 1)
        plt.plot(t, target[:, i], "g--")
        plt.plot(t, pred[:, i], "r.")
        # ax_i.set_ylim(minv - view/10, maxv + view/10)
        if i == 0:
            plt.title(msg)

    # fig, axs = plt.subplots(6, 1)
    # for i, ax in enumerate(axs):
    #    ax.plot(target[:, i], 'g--', pred[:, i], 'r-')

    plt.savefig("%s/%s.png" % (folder, tag))
    plt.close("all")


def t2np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.squeeze().detach().cpu().numpy()


def prediction_error(truth: np.ndarray, prediction: np.ndarray):
    assert (
        truth.shape == prediction.shape
    ), "Incompatible truth and prediction for calculating prediction error"
    # each shape (sequence, n_outputs)
    # Root Relative Squared Error
    se = np.sum((truth - prediction) ** 2, axis=0)  # summed squared error per channel
    rse = se / np.sum((truth - np.mean(truth, axis=0)) ** 2)  # relative squared error
    rrse = np.mean(np.sqrt(rse))  # square root, followed by mean over channels
    return 100 * rrse  # in percentage


def predict(
    x_list: list[torch.Tensor],
    y_list: list[torch.Tensor],
    dt_list: list[torch.Tensor],
    model,
    is_mse_list: Optional[bool] = True,
) -> list:
    assert (
        len(x_list) == len(y_list) == len(dt_list)
    ), "The dimensions of input data are different"
    h_pred_list: list[torch.Tensor] = []
    Y_pred_list: list[torch.Tensor] = []
    error_list = []
    mse_list = []

    for i in range(len(x_list)):
        x_tn = x_list[i]
        dt_tn = dt_list[i]
        y_tn = y_list[i]
        Y_pred, h_pred = model(x_tn, dt=dt_tn)
        error_step = prediction_error(y_list[i], t2np(Y_pred))
        if is_mse_list:
            mse_step = model.criterion(Y_pred, y_tn)
            mse_list.append(mse_step)

        error_list.append(error_step)
        Y_pred_list.append(Y_pred)
        h_pred_list.append(h_pred)

    ans = [Y_pred_list, h_pred_list, error_list]

    if is_mse_list:
        ans.append(mse_list)

    return ans


def show_data_for_list(
    t_list, y_list, y_pred_list, paras, type: str, error_list, epoch
):
    for i in range(len(t_list)):
        show_data(
            t_list[i],
            y_list[i],
            t2np(y_pred_list[i]),
            paras.save + "/" + type,
            "current_" + type + "_results of %dth trajectory" % i,
            msg=type
            + "results ("
            + type
            + "error %.3f) at iter %d" % (error_list[i], epoch),
        )
