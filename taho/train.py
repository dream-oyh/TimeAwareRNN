import numpy as np
import torch
import torch.nn as nn

"""
EpochTrainer for training recurrent models on single sequence of inputs and outputs,
by chunking into bbtt-long segments.
"""


class EpochTrainer(object):
    def __init__(
        self,
        model: nn.Module,
        optimizer,
        epochs,
        X: list[np.ndarray],
        Y: list[np.ndarray],
        dt: list[np.ndarray],
        batch_size=1,
        gpu=False,
        bptt=50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.X_list, self.Y_list, self.dt_list = X, Y, dt
        self.batch_size = batch_size
        self.gpu = gpu
        self.Xtrain, self.Ytrain = None, None
        self.train_inds = []
        self.bptt = (
            bptt  # for now: constant segment length, hence constant train indices
        )
        self.set_train_tensors()
        self.all_states = None

        # print(
        #     "Initialized epoch trainer: original shape for X",
        #     len(self.X),
        #     "and for Y",
        #     self.Y.shape,
        # )
        print(
            "segmented size (segments, bptt, in) for X",
            self.Xtrain.size(),
            "and (segments, bptt, out) for Y",
            self.Ytrain.size(),
        )
        print("for batch size", self.batch_size)

    def set_train_tensors(self):
        w = self.bptt

        # 切割，每个数据集当作一条完整轨迹，现在要将每个完整的轨迹切成 bptt 长度的一小段轨迹（每一小段视作一个 instance）
        # 所以 self.X_train_ins_list 就存放了从每个数据集拆出的 instance。

        self.X_train_ins_list: list[np.ndarray] = []
        self.Y_train_ins_list: list[np.ndarray] = []
        self.dt_train_ins_list: list[np.ndarray] = []
        self.train_inds_list: list[list] = []

        for i in range(len(self.X)):
            X_train_trajectory = self.X[i]
            Y_train_trajectory = self.Y[i]
            dt_train_trajectory = self.dt[i]

            N_X = X_train_trajectory.shape[0]
            N_Y = Y_train_trajectory.shape[0]
            N_dt = dt_train_trajectory.shape[0]

            assert N_X == N_Y == N_dt, "the dimensions of X, Y, dt are different"

            X_ins = torch.tensor(
                np.asarray(
                    [
                        X_train_trajectory[j : min(N_X, j + w), :]
                        for j in range(max(1, N_X - w + 1))
                    ]
                ),
                dtype=torch.float,
            )
            Y_ins = torch.tensor(
                np.asarray(
                    [
                        Y_train_trajectory[j : min(N_Y, j + w), :]
                        for j in range(max(1, N_Y - w + 1))
                    ]
                ),
                dtype=torch.float,
            )
            dt_ins = torch.tensor(
                np.asarray(
                    [
                        dt_train_trajectory[j : min(N_dt, j + w), :]
                        for j in range(max(1, N_dt - w + 1))
                    ]
                ),
                dtype=torch.float,
            )

            if self.gpu:
                X_ins = X_ins.cuda()
                Y_ins = Y_ins.cuda()
                dt_ins = dt_ins.cuda()

            self.X_train_ins_list.append(X_ins)
            self.Y_train_ins_list.append(Y_ins)
            self.dt_train_ins_list.append(dt_ins)
            self.train_inds_list.append(
                list(range(X_ins.shape[0]))
            )  # 每个数据集的子轨迹索引 (file_nums, range(instance_nums N-w+1))

        self.X_train_tensor_list: list[torch.Tensor] = []
        self.Y_train_tensor_list: list[torch.Tensor] = []
        self.dt_train_tensor_list: list[torch.Tensor] = []

        for X_part, Y_part, dt_part in zip(self.X_list, self.Y_list, self.dt_list):
            X_part_tensor = torch.tensor(X_part, dtype=torch.float).unsqueeze(0)
            Y_part_tensor = torch.tensor(Y_part, dtype=torch.float).unsqueeze(0)
            dt_part_tensor = torch.tensor(dt_part, dtype=torch.float).unsqueeze(0)

            if self.gpu:
                X_part_tensor = X_part_tensor.cuda()
                Y_part_tensor = Y_part_tensor.cuda()
                dt_part_tensor = dt_part_tensor.cuda()

            self.X_train_tensor_list.append(
                X_part_tensor
            )  # (file_nums, 1, sample_nums, k_in)  all lengths
            self.Y_train_tensor_list.append(
                Y_part_tensor
            )  # (file_nums, 1, sample_nums, k_out)  all lengths
            self.dt_train_tensor_list.append(dt_part_tensor)

    def set_states(self):
        with torch.no_grad():  # no backprob beyond initial state for each chunk.
            self.all_states_list: list[torch.Tensor] = []

            for i in range(len(self.X_train_tensor_list)):
                X_train_tensor = self.X_train_tensor_list[i]
                dt_train_tensor = self.dt_train_tensor_list[i]

                all_states = (
                    self.model(X_train_tensor, state0=None, dt=dt_train_tensor)[1]
                    .squeeze(0)
                    .data
                )
                # (sample_nums, k_states)
                self.all_states_list.append(all_states)

    def __call__(self, epoch):
        for train_inds in self.train_inds_list:
            np.random.shuffle(train_inds)

        # iterations within current epoch
        epoch_loss = 0.0
        cum_bs = 0

        # train initial state only once per epoch
        self.model.train()
        self.model.zero_grad()

        X_ins_0 = self.X_train_tensor_list[0][:, : self.bptt, :]
        Y_ins_0 = self.Y_train_tensor_list[0][:, : self.bptt, :]
        dt_ins_0 = self.dt_train_tensor_list[0][:, : self.bptt, :]

        Y_pred, _ = self.model(X_ins_0, dt=dt_ins_0)
        loss = self.model.criterion(Y_pred, Y_ins_0)
        loss.backward()
        self.optimizer.step()

        # set all states only once per epoch (trains much faster than at each iteration)
        # (no gradient through initial state for each chunk)
        self.set_states()

        total_sample_nums = sum(
            [len(train_inds) for train_inds in self.train_inds_list]
        )  # 各个数据集合起来的总子轨迹量

        for i in range(int(np.ceil(total_sample_nums / self.batch_size))):
            # get indices for next batch
            for i, train_inds in enumerate(self.train_inds_list):
                # 对每个数据集分别进行计算
                iter_inds = train_inds[
                    i * self.batch_size : min(
                        (i + 1) * self.batch_size, len(train_inds)
                    )
                ]
                bs = len(iter_inds)

                state0 = self.all_states[i][iter_inds, :]
                cum_bs += bs

                X = self.X_train_ins_list[i][iter_inds, :, :]
                Y_target = self.Y_train_ins_list[i][iter_inds, :, :]
                dt = self.dt_train_ins_list[i][iter_inds, :]

                self.model.train()
                self.model.zero_grad()
                Y_pred, _ = self.model(X, state0=state0, dt=dt)
                loss = self.model.criterion(Y_pred, Y_target)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * bs

                # 原仓库里没有下面这一条更新状态的命令，但是我觉得得加上

                self.set_states()

        epoch_loss /= cum_bs

        return epoch_loss
