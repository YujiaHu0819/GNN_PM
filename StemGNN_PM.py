import os
from pathlib import Path
import json
from datetime import datetime
import time


import numpy as np
import torch
import pandas as pd
import pickle


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torch_data
from torch.utils.data import DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset


import torch.fft

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from datetime import datetime

import argparse
import pandas as pd


os.chdir("/mnt/sdb/yujia/pmf")

data_dir = Path.cwd()

# datasets = ['BPI2019', 'InternationalDeclarations', 'DomesticDeclarations']
datasets = ['RTFMP', 'italian', 'Sepsis', 'InternationalDeclarations', 'DomesticDeclarations', 'BPI2019', 'BPI2018', 'BPI2017', 'bpi12']

agg_types = ['equisize','equitemp']
no_intervals_alls = [500, 1000, 3000]
# horizon = 100
# no_intervals = 400

window_size = 12
horizon = 3


for dataset in datasets:
    for agg_type in agg_types:
        for no_intervals_all in no_intervals_alls:
            print('======================================== \n ============================================================= \n ============================================================= \n\n\n\n')
            
            datapath = f'{data_dir}/dataset/{dataset}/{dataset}_input_{no_intervals_all}_{agg_type}.csv'
            input = pd.read_csv(datapath)
            train_dataset = input[:int(0.73 * len(input))]
            # print(train)
            valid_dataset = input[int(0.7 * len(input)):int((0.7 + 0.05) * len(input))]
            test_dataset = input[int(0.25 * len(input)):]

            # result_fold = './result'
            result_train_fold = './result/train'
            result_test_fold = './result/test'


            # RTFMP_path_100 = data_dir/"RTFMP_input_100.csv"

            # RTFMP_path_1000 = data_dir/'RTFMP_input_1000.csv'

            # italian_path_100 =data_dir/"italian_input_100_equisize.csv" 

            # italian_path_1000 =data_dir/"italian_input_1000_equisize.csv" 

            # RTFMP_path_100_equitemp = data_dir/"RTFMP_input_100_equitemp.csv"

            # RTFMP_path_1000_equitemp = data_dir/"RTFMP_input_1000_equitemp.csv"

            # italian_path_100_equitemp = data_dir/'italian_input_100_equitemp.csv'

            # italian_path_1000_equitemp = data_dir/'italian_input_1000_equitemp.csv'



            ## DataLoader

            def normalized(data, normalize_method, norm_statistic=None):
                if normalize_method == 'min_max':
                    if not norm_statistic:
                        norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
                    scale = norm_statistic['max'] - norm_statistic['min'] + 1e-5
                    data = (data - norm_statistic['min']) / scale
                    data = np.clip(data, 0.0, 1.0)
                elif normalize_method == 'z_score':
                    if not norm_statistic:
                        norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
                    mean = norm_statistic['mean']
                    std = norm_statistic['std']
                    std = [1 if i == 0 else i for i in std]
                    data = (data - mean) / std
                    norm_statistic['std'] = std
                return data, norm_statistic


            def de_normalized(data, normalize_method, norm_statistic):
                if normalize_method == 'min_max':
                    if not norm_statistic:
                        norm_statistic = dict(max=np.max(data, axis=0), min=np.min(data, axis=0))
                    scale = norm_statistic['max'] - norm_statistic['min'] + 1e-8
                    data = data * scale + norm_statistic['min']
                elif normalize_method == 'z_score':
                    if not norm_statistic:
                        norm_statistic = dict(mean=np.mean(data, axis=0), std=np.std(data, axis=0))
                    mean = norm_statistic['mean']
                    std = norm_statistic['std']
                    std = [1 if i == 0 else i for i in std]
                    data = data * std + mean
                return data


            class ForecastDataset(torch_data.Dataset):
                def __init__(self, df, window_size, horizon, normalize_method=None, norm_statistic=None, interval=1):
                    self.window_size = window_size
                    self.interval = interval
                    self.horizon = horizon
                    self.normalize_method = normalize_method
                    self.norm_statistic = norm_statistic
                    df = pd.DataFrame(df)
                    df = df.fillna(method='ffill', limit=len(df)).fillna(method='bfill', limit=len(df)).values
                    self.data = df
                    self.df_length = len(df)
                    self.x_end_idx = self.get_x_end_idx()
                    if normalize_method:
                        self.data, _ = normalized(self.data, normalize_method, norm_statistic)

                def __getitem__(self, index):
                    # print(len(self.x_end_idx))
                    # print('index', index)
                    hi = self.x_end_idx[index]
                    lo = hi - self.window_size
                    train_data = self.data[lo: hi]
                    target_data = self.data[hi:hi + self.horizon]
                    x = torch.from_numpy(train_data).type(torch.float)
                    y = torch.from_numpy(target_data).type(torch.float)
                    return x, y

                def __len__(self):
                    return len(self.x_end_idx)

                def get_x_end_idx(self):
                    # each element `hi` in `x_index_set` is an upper bound for get training data
                    # training data range: [lo, hi), lo = hi - window_size
                    x_index_set = range(self.window_size, self.df_length - self.horizon + 1)
                    x_end_idx = [x_index_set[j * self.interval] for j in range((len(x_index_set)) // self.interval)]
                    return x_end_idx
                


            # input_dataset = ForecastDataset(input, window_size= 10, horizon= horizon)
            # input_dataloader = DataLoader(input_dataset, batch_size=4, shuffle = True)

            def masked_MAPE(v, v_, axis=None):
                '''
                Mean absolute percentage error.
                :param v: np.ndarray or int, ground truth.
                :param v_: np.ndarray or int, prediction.
                :param axis: axis to do calculation.
                :return: int, MAPE averages on all elements of input.
                '''
                mask = (v == 0)
                percentage = np.abs(v_ - v) / np.abs(v)
                if np.any(mask):
                    masked_array = np.ma.masked_array(percentage, mask=mask)  # mask the dividing-zero as invalid
                    result = masked_array.mean(axis=axis)
                    if isinstance(result, np.ma.MaskedArray):
                        return result.filled(np.nan)
                    else:
                        return result
                return np.mean(percentage, axis).astype(np.float64)


            def MAPE(v, v_, axis=None):
                '''
                Mean absolute percentage error.
                :param v: np.ndarray or int, ground truth.
                :param v_: np.ndarray or int, prediction.
                :param axis: axis to do calculation.
                :return: int, MAPE averages on all elements of input.
                '''
                mape = (np.abs(v_ - v) / (np.abs(v)+1e-5)).astype(np.float64)
                mape = np.where(mape > 5, 5, mape)
                return np.mean(mape, axis)


            def RMSE(v, v_, axis=None):
                '''
                Mean squared error.
                :param v: np.ndarray or int, ground truth.
                :param v_: np.ndarray or int, prediction.
                :param axis: axis to do calculation.
                :return: int, RMSE averages on all elements of input.
                '''
                return np.sqrt(np.mean((v_ - v) ** 2, axis)).astype(np.float64)


            def MAE(v, v_, axis=None):
                '''
                Mean absolute error.
                :param v: np.ndarray or int, ground truth.
                :param v_: np.ndarray or int, prediction.
                :param axis: axis to do calculation.
                :return: int, MAE averages on all elements of input.
                '''
                return np.mean(np.abs(v_ - v), axis).astype(np.float64)


            def evaluate(y, y_hat, by_step=False, by_node=False):
                '''
                :param y: array in shape of [count, time_step, node].
                :param y_hat: in same shape with y.
                :param by_step: evaluate by time_step dim.
                :param by_node: evaluate by node dim.
                :return: array of mape, mae and rmse.
                '''
                if not by_step and not by_node:
                    return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)
                if by_step and by_node:
                    return MAPE(y, y_hat, axis=0), MAE(y, y_hat, axis=0), RMSE(y, y_hat, axis=0)
                if by_step:
                    return MAPE(y, y_hat, axis=(0, 2)), MAE(y, y_hat, axis=(0, 2)), RMSE(y, y_hat, axis=(0, 2))
                if by_node:
                    # print('label shape', y.shape)
                    return MAPE(y, y_hat, axis=(0, 1)), MAE(y, y_hat, axis=(0, 1)), RMSE(y, y_hat, axis=(0, 1))
                



            ## GNN part

            class GLU(nn.Module):
                def __init__(self, input_channel, output_channel):
                    super(GLU, self).__init__()
                    self.linear_left = nn.Linear(input_channel, output_channel)
                    self.linear_right = nn.Linear(input_channel, output_channel)

                def forward(self, x):
                    return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


            class StockBlockLayer(nn.Module):
                def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
                    super(StockBlockLayer, self).__init__()
                    self.time_step = time_step
                    self.unit = unit
                    self.stack_cnt = stack_cnt
                    self.multi = multi_layer
                    self.weight = nn.Parameter(
                        torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi,
                                    self.multi * self.time_step))  # [K+1, 1, in_c, out_c]
                    nn.init.xavier_normal_(self.weight)
                    self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
                    self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
                    if self.stack_cnt == 0:
                        self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
                    self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
                    self.relu = nn.ReLU()
                    self.GLUs = nn.ModuleList()
                    self.output_channel = 4 * self.multi
                    for i in range(3):
                        if i == 0:
                            self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                            self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                        elif i == 1:
                            self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                            self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                        else:
                            self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                            self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))

                def spe_seq_cell(self, input):
                    batch_size, k, input_channel, node_cnt, time_step = input.size()
                    input = input.view(batch_size, -1, node_cnt, time_step)
                    # print(input.shape)

                    # ffted = torch.rfft(input, 1, onesided=False)
                    ffted = torch.view_as_real(torch.fft.fft(input, dim=1))
                    # ffted = torch.fft.rfft(input, 1)

                    # ffted = torch.fft.fft(input, dim=-1)    
                    # ffted = torch.stack((ffted.real, ffted.imag), -1)   

                    # print('ffted', ffted.shape)

                    real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
                    img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
                    # print('real1', real.shape)

                    for i in range(3):
                        real = self.GLUs[i * 2](real)
                        img = self.GLUs[2 * i + 1](img)
                    real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
                    img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()

                    # print('real', real.shape)

                    time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
                    # print('inner', time_step_as_inner.shape)


                    # iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
                    iffted = torch.fft.irfft(torch.view_as_complex(time_step_as_inner), n=time_step_as_inner.shape[1], dim=1)

                    # iffted = torch.fft.irfft2(torch.complex(time_step_as_inner[:,:,0], time_step_as_inner[:,:,1]), dim = -1)
                    # iffted = torch.fft.ifft2(time_step_as_inner, dim=-1)                   # 输入为复数形式


                    # iffted = torch.fft.ifft2(torch.complex(time_step_as_inner[..., 0],      # 输入为数组形式
                    #                                         time_step_as_inner[..., 1]), dim=-1)    

                    # print('iffted', iffted.real.shape)
                    return iffted
                    # return iffted.real

                def forward(self, x, mul_L):
                    # print(self.time_step)
                    # print(self.unit)
                    # print(self.stack_cnt)
                    # print(self.multi)
                    

                    mul_L = mul_L.unsqueeze(1)
                    x = x.unsqueeze(1)
                    gfted = torch.matmul(mul_L, x)
                    gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)

                    # print(gconv_input.shape, self.weight.shape)

                    igfted = torch.matmul(gconv_input, self.weight)
                    igfted = torch.sum(igfted, dim=1)
                    forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
                    forecast = self.forecast_result(forecast_source)
                    if self.stack_cnt == 0:
                        backcast_short = self.backcast_short_cut(x).squeeze(1)
                        backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
                    else:
                        backcast_source = None
                    return forecast, backcast_source


            class Model(nn.Module):
                def __init__(self, units, stack_cnt, time_step, multi_layer, horizon, dropout_rate=0.5, leaky_rate=0.2,
                            device =torch.device("cuda" if torch.cuda.is_available() else "cpu") 
                            #  device='cpu'
                            ):
                    super(Model, self).__init__()
                    self.unit = units
                    self.stack_cnt = stack_cnt
                    self.unit = units
                    self.alpha = leaky_rate
                    self.time_step = time_step
                    self.horizon = horizon
                    self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
                    nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
                    self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
                    nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
                    self.GRU = nn.GRU(self.time_step, self.unit)
                    self.multi_layer = multi_layer
                    self.stock_block = nn.ModuleList()
                    self.stock_block.extend(
                        [StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)])
                    self.fc = nn.Sequential(
                        nn.Linear(int(self.time_step), int(self.time_step)),
                        nn.LeakyReLU(),
                        nn.Linear(int(self.time_step), self.horizon),
                    )
                    self.leakyrelu = nn.LeakyReLU(self.alpha)
                    self.dropout = nn.Dropout(p=dropout_rate)
                    self.to(device)

                def get_laplacian(self, graph, normalize):
                    """
                    return the laplacian of the graph.
                    :param graph: the graph structure without self loop, [N, N].
                    :param normalize: whether to used the normalized laplacian.
                    :return: graph laplacian.
                    """
                    if normalize:
                        D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
                        L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
                    else:
                        D = torch.diag(torch.sum(graph, dim=-1))
                        L = D - graph
                    return L

                def cheb_polynomial(self, laplacian):
                    """
                    Compute the Chebyshev Polynomial, according to the graph laplacian.
                    :param laplacian: the graph laplacian, [N, N].
                    :return: the multi order Chebyshev laplacian, [K, N, N].
                    """
                    N = laplacian.size(0)  # [N, N]
                    laplacian = laplacian.unsqueeze(0)
                    first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
                    second_laplacian = laplacian
                    third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
                    forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
                    multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
                    return multi_order_laplacian

                def latent_correlation_layer(self, x):
                    input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
                    input = input.permute(1, 0, 2).contiguous()
                    attention = self.self_graph_attention(input)
                    attention = torch.mean(attention, dim=0)
                    degree = torch.sum(attention, dim=1)
                    # laplacian is sym or not
                    attention = 0.5 * (attention + attention.T)
                    degree_l = torch.diag(degree)
                    #[node, node]

                    diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
                    laplacian = torch.matmul(diagonal_degree_hat,
                                            torch.matmul(degree_l - attention, diagonal_degree_hat))
                    
                    mul_L = self.cheb_polynomial(laplacian)

                    # with open(f'./mul_L/{dataset}_{no_intervals_all}_{agg_type}.pkl', 'wb') as f:
                    #         pickle.dump(mul_L, f)
                    
                    return mul_L, attention

                def self_graph_attention(self, input):
                    input = input.permute(0, 2, 1).contiguous()
                    bat, N, fea = input.size()
                    key = torch.matmul(input, self.weight_key)
                    query = torch.matmul(input, self.weight_query)
                    data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
                    data = data.squeeze(2)
                    data = data.view(bat, N, -1)
                    data = self.leakyrelu(data)
                    attention = F.softmax(data, dim=2)
                    attention = self.dropout(attention)
                    return attention

                def graph_fft(self, input, eigenvectors):
                    return torch.matmul(eigenvectors, input)

                def forward(self, x):
                    # print('input size', x.shape)
                    mul_L, attention = self.latent_correlation_layer(x)
                    
                    # np.save(f'./matrix/latent_matrix/{dataset}_{no_intervals_all}_{agg_type}.npy', mul_L.numpy())
                    
                    # print('mul_L after latent', mul_L.shape)
                    # print('attention after latent', attention.shape)
                    with open(f'./mul_L/{dataset}_{no_intervals_all}_{agg_type}.pkl', 'wb') as f:
                            pickle.dump(mul_L, f)


                    X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
                    #(batch, sequence, features) ==> (batch, 1, features, sequences)
                    result = []
                    # print('X_Shape', X.shape, 'mul_L shape', mul_L.shape)
                    
                    for stack_i in range(self.stack_cnt):
                        forecast, X = self.stock_block[stack_i](X, mul_L)
                        result.append(forecast)
                    forecast = result[0] + result[1]
                    forecast = self.fc(forecast)
                    if forecast.size()[-1] == 1:
                        return forecast.unsqueeze(1).squeeze(-1), attention
                    else:
                        return forecast.permute(0, 2, 1).contiguous(), attention
                    



            def cv_train(train_data, 
                        # valid_data, 
                        # args, 
                        window_size,
                        multi_layer,
                        horizon,
                        epoch,
                        lr,
                        batch_size,
                        exponential_decay_step,
                        validate_freq,
                        early_stop,
                        # early_stop_step,
                        decay_rate,
                        norm_method,
                        optimizer,
                        result_file,
                        device =torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        dataset = dataset):


                tscv = TimeSeriesSplit(n_splits = 10)
                # rmse = []

                history = {'train_loss': [], 'valid_loss': []}

                # for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_data)))):

                for train_index, test_index in tscv.split(train_data):
                    # print('Fold {}'.format(fold + 1))
                    train_data1 = np.array(train_data)
                    train_set, valid_set = train_data1[train_index], train_data1[test_index]
                    train_set, valid_set = pd.DataFrame(train_set), pd.DataFrame(valid_set)
                    print("train_set_shape: ", train_set.shape, 
                        "\nvalid_set_shape: ",valid_set.shape)


                    if norm_method == 'z_score':
                        train_mean = np.mean(train_set, axis=0)
                        train_std = np.std(train_set, axis=0)
                        normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
                    elif norm_method == 'min_max':
                        train_min = np.min(train_set, axis=0)
                        train_max = np.max(train_set, axis=0)
                        normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
                    else:
                        normalize_statistic = None
                    if normalize_statistic is not None:
                        with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
                            json.dump(normalize_statistic, f)

                    # train_sampler = SubsetRandomSampler(train_idx)
                    # val_sampler = SubsetRandomSampler(val_idx)

                    train_set1 = ForecastDataset(train_set, window_size=window_size, horizon=horizon,
                                            normalize_method=norm_method, norm_statistic=normalize_statistic)
                    valid_set1 = ForecastDataset(valid_set, window_size=window_size, horizon=horizon,
                                            normalize_method=norm_method, norm_statistic=normalize_statistic)

                    # print(train_set)

                    train_loader = torch_data.DataLoader(train_set1, batch_size=batch_size, shuffle=True)
                                                    # , drop_last=False     , num_workers=0)
                    valid_loader = torch_data.DataLoader(valid_set1, batch_size=batch_size, shuffle=False)
                                                    #  , num_workers=0)
                    
                    # train_loader = torch_data.DataLoader(train_set, batch_size=batch_size, sampler= train_sampler, drop_last=False, num_workers = 0)
                    # valid_loader = torch_data.DataLoader(train_set, batch_size=batch_size, sampler= val_sampler) 

                    
                    node_cnt = train_data.shape[1]
                    print(f'{dataset}_node_cnt: ', node_cnt)
                    model = Model(node_cnt, 2, window_size, multi_layer, horizon=horizon)
                    model.to(device)


                    if optimizer == 'RMSProp':
                        my_optim = torch.optim.RMSprop(params=model.parameters(), lr=lr, eps=1e-08)
                    else:
                        my_optim = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
                    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=decay_rate)

                    
                
                    forecast_loss = nn.MSELoss(reduction='mean').to(device)

                    total_params = 0
                    for name, parameter in model.named_parameters():
                        if not parameter.requires_grad: continue
                        param = parameter.numel()
                        total_params += param
                    print(f"Total Trainable Params: {total_params}")


                    best_validate_mae = np.inf
                    validate_score_non_decrease_count = 0
                    performance_metrics = {}


                    for epoch in range(epoch):

                        epoch_start_time = time.time()
                        model.train()
                        loss_total = 0
                        cnt = 0

                        # next(iter(train_loader))
                        for i, (inputs, target) in enumerate(train_loader):
                        # for inputs, target in train_loader:
                            print("input shape \n ", input.shape)
                            print('target shape \n ', target.shape)
                            inputs = inputs.to(device)
                            target = target.to(device)
                            model.zero_grad()
                            forecast, _ = model(inputs)

                            # print(forecast.shape)

                            loss = forecast_loss(forecast, target)
                            cnt += 1
                            loss.backward()
                            my_optim.step()
                            loss_total += float(loss)
                        print('| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                                time.time() - epoch_start_time), loss_total / cnt))
                        
                        save_model(model, result_file, epoch)


                        model.eval()
                        val_loss = 0
                        cnt_val = 0

                        for i, (inputs, target) in enumerate(valid_loader):
                            inputs = inputs.to(device)
                            target = target.to(device)

                            forecast, _ = model(inputs) 
                            
                            loss = forecast_loss(forecast, target)
                            cnt_val += 1
                            val_loss += float(loss)

                        print('| end of epoch {:3d} | time: {:5.2f}s | valid_total_loss {:5.4f}'.format(epoch, (
                                time.time() - epoch_start_time), val_loss / cnt))


                        history['train_loss'].append(loss_total)
                        history['valid_loss'].append(val_loss)

                    # if (epoch+1) % exponential_decay_step == 0:
                    #     my_lr_scheduler.step()
                    # if (epoch + 1) % validate_freq == 0:
                    #     is_best_for_now = False
                    #     print('------ validate on data: VALIDATE ------')


                        # performance_metrics = \
                        #     validate(model, valid_loader, device, norm_method, normalize_statistic,
                        #              node_cnt, window_size, horizon,
                        #              result_file=result_file)
                        # if best_validate_mae > performance_metrics['mae']:
                        #     best_validate_mae = performance_metrics['mae']
                        #     is_best_for_now = True
                        #     validate_score_non_decrease_count = 0
                        # else:
                        #     validate_score_non_decrease_count += 1
                        
                        
                        # save model
                        # if is_best_for_now:
                        #     save_model(model, result_file)



                    # early stop
                    # if early_stop and validate_score_non_decrease_count >= early_stop_step:
                    #     break
                return history



            def save_model(model, model_dir, epoch=None):
                if model_dir is None:
                    return
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                epoch = str(epoch) if epoch else ''
                file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
                with open(file_name, 'wb') as f:
                    torch.save(model, f)


            def load_model(model_dir, epoch=None):
                if not model_dir:
                    return
                epoch = str(epoch) if epoch else ''
                file_name = os.path.join(model_dir, epoch + '_stemgnn.pt')
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                if not os.path.exists(file_name):
                    return
                with open(file_name, 'rb') as f:
                    model = torch.load(f)
                return model



            def inference(model, dataloader, device, node_cnt, window_size, horizon):
                forecast_set = []
                target_set = []
                model.eval()
                with torch.no_grad():
                    for i, (inputs, target) in enumerate(dataloader):
                        inputs = inputs.to(device)
                        target = target.to(device)
                        # print('input', inputs)
                        # print('target', target)

                        step = 0
                        forecast_steps = np.float64(np.zeros([inputs.size()[0], horizon, node_cnt]))
                        while step < horizon:
                            forecast_result, a = model(inputs)
                            len_model_output = forecast_result.size()[1]
                            if len_model_output == 0:
                                raise Exception('Get blank inference result')
                            inputs[:, :window_size - len_model_output, :] = inputs[:, len_model_output:window_size,
                                                                            :].clone()
                            
                            # print(forecast_result.shape)


                            inputs[:, window_size - len_model_output:, :] = forecast_result.clone()
                            forecast_steps[:, step:min(horizon - step, len_model_output) + step, :] = \
                                forecast_result[:, :min(horizon - step, len_model_output), :].detach().cpu().numpy()
                            step += min(horizon - step, len_model_output)

                            # print('forecast_set', forecast_steps)
                            # print('target_set', target)

                        forecast_set.append(forecast_steps)
                        target_set.append(target.detach().cpu().numpy())

                        # print(len(forecast_set), np.concatenate(forecast_set, axis=0).shape)
                        # print(len(target_set), np.concatenate(target_set, axis=0).shape)

                return np.concatenate(forecast_set, axis=0), np.concatenate(target_set, axis=0)
                # return forecast_set, target_set


            def validate(model, dataloader, device, normalize_method, statistic,
                        node_cnt, window_size, horizon,
                        result_file=None):
                start = datetime.now()
                forecast_norm, target_norm = inference(model, dataloader, device,
                                                    node_cnt, window_size, horizon)
                if normalize_method and statistic:
                    forecast = de_normalized(forecast_norm, normalize_method, statistic)
                    target = de_normalized(target_norm, normalize_method, statistic)
                else:
                    forecast, target = forecast_norm, target_norm
                score = evaluate(target, forecast)
                score_by_node = evaluate(target, forecast, by_node=True)
                end = datetime.now()

                # print(score)
                score_norm = evaluate(target_norm, forecast_norm)

                print(f'NORM: MAPE {score_norm[0]:7.9%}; MAE {score_norm[1]:7.9f}; RMSE {score_norm[2]:7.9f}.')
                print(f'RAW : MAPE {score[0]:7.9%}; MAE {score[1]:7.9f}; RMSE {score[2]:7.9f}.')
                print(f'Score by node: MAPE {score_by_node[0].mean()}; MAE {score_by_node[1].mean()}; RMSE {score_by_node[2].mean()}')

                if result_file:
                    if not os.path.exists(result_file):
                        os.makedirs(result_file)
                    step_to_print = 0
                    forcasting_2d = forecast[:, step_to_print, :]
                    forcasting_2d_target = target[:, step_to_print, :]

                    np.savetxt(f'{result_file}/target.csv', forcasting_2d_target, delimiter=",")
                    np.savetxt(f'{result_file}/predict.csv', forcasting_2d, delimiter=",")
                    np.savetxt(f'{result_file}/predict_abs_error.csv',
                            np.abs(forcasting_2d - forcasting_2d_target), delimiter=",")
                    np.savetxt(f'{result_file}/predict_ape.csv',
                            np.abs((forcasting_2d - forcasting_2d_target) / forcasting_2d_target), delimiter=",")

                # print()
                return dict(mae=score[1], mae_node=score_by_node[1], mape=score[0], mape_node=score_by_node[0],
                            rmse=score[2], rmse_node=score_by_node[2])



            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # model.to(device)



            def train(train_data, 
                        valid_data, 
                        # args, 
                        window_size,
                        multi_layer,
                        horizon,
                        epoch,
                        lr,
                        batch_size,
                        exponential_decay_step,
                        validate_freq,
                        early_stop,
                        # early_stop_step,
                        decay_rate,
                        norm_method,
                        optimizer,
                        result_file,
                        device =torch.device("cuda" if torch.cuda.is_available() else "cpu") ):


                # for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_data)))):
                #     print('Fold {}'.format(fold + 1))

                if not os.path.exists(result_file):
                    os.mkdir(result_file)

                # node_cnt = train_data.shape[1]
                # print("train_data \n", train_data)
                node_cnt = train_data.shape[1]
                print(f'{dataset}_node_cnt: ', node_cnt)

                model = Model(node_cnt, 2, window_size, multi_layer, horizon=horizon)
                model.to(device)
                if len(train_data) == 0:
                    raise Exception('Cannot organize enough training data')
                # if len(valid_data) == 0:
                #     raise Exception('Cannot organize enough validation data')

                if norm_method == 'z_score':
                    train_mean = np.mean(train_data, axis=0)
                    train_std = np.std(train_data, axis=0)
                    normalize_statistic = {"mean": train_mean.tolist(), "std": train_std.tolist()}
                elif norm_method == 'min_max':
                    train_min = np.min(train_data, axis=0)
                    train_max = np.max(train_data, axis=0)
                    normalize_statistic = {"min": train_min.tolist(), "max": train_max.tolist()}
                else:
                    normalize_statistic = None

                if normalize_statistic is not None:
                    with open(os.path.join(result_file, 'norm_stat.json'), 'w') as f:
                        json.dump(normalize_statistic, f)

                if optimizer == 'RMSProp':
                    my_optim = torch.optim.RMSprop(params=model.parameters(), lr=lr, eps=1e-08)
                else:
                    my_optim = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))
                my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=my_optim, gamma=decay_rate)

                train_set = ForecastDataset(train_data, window_size=window_size, horizon=horizon,
                                            normalize_method=norm_method, norm_statistic=normalize_statistic)
                valid_set = ForecastDataset(valid_data, window_size=window_size, horizon=horizon,
                                            normalize_method=norm_method, norm_statistic=normalize_statistic)
                train_loader = torch_data.DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=True,
                                                    num_workers=0)
                valid_loader = torch_data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

                forecast_loss = nn.MSELoss(reduction='mean').to(device)

                total_params = 0
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad: continue
                    param = parameter.numel()
                    total_params += param
                print(f"Total Trainable Params: {total_params}")


                best_validate_mae = np.inf
                validate_score_non_decrease_count = 0
                performance_metrics = {}


                for epoch in range(epoch):
                    epoch_start_time = time.time()
                    model.train()
                    loss_total = 0
                    cnt = 0
                    for i, (inputs, target) in enumerate(train_loader):
                        inputs = inputs.to(device)
                        target = target.to(device)
                        model.zero_grad()
                        forecast, _ = model(inputs)

                        # print(forecast.shape)
                        loss = forecast_loss(forecast, target)
                        cnt += 1
                        loss.backward()
                        my_optim.step()
                        loss_total += float(loss)
                    print(f'{dataset}_{agg_type}', '| end of epoch {:3d} | time: {:5.2f}s | train_total_loss {:5.4f}'.format(epoch, (
                            time.time() - epoch_start_time), loss_total / cnt))
                    # save_model(model, result_file, epoch)

                    if (epoch+1) % exponential_decay_step == 0:
                        my_lr_scheduler.step()
                    if (epoch + 1) % validate_freq == 0:
                        is_best_for_now = False



                        print(f'------ validate on {dataset}_{agg_type} data: VALIDATE ------')
                        performance_metrics = \
                            validate(model, valid_loader, device, norm_method, normalize_statistic,
                                    node_cnt, window_size, horizon,
                                    result_file=result_file)
                        if best_validate_mae > performance_metrics['mae']:
                            best_validate_mae = performance_metrics['mae']
                            is_best_for_now = True
                            validate_score_non_decrease_count = 0
                        else:
                            validate_score_non_decrease_count += 1
                        
                        
                        # save model
                        if is_best_for_now:
                            save_model(model, result_file)



                    # early stop
                    # if early_stop and validate_score_non_decrease_count >= early_stop_step:
                    #     break
                return performance_metrics, normalize_statistic



            def test(test_data, 
                    # args, 
                    window_size,
                    horizon,
                    norm_method,
                    batch_size,
                    result_train_file, 
                    result_test_file,
                    device =torch.device("cuda" if torch.cuda.is_available() else "cpu") 
                    # device = 'cpu'
                    ):
                
                if not os.path.exists(result_test_file):
                    os.mkdir(result_test_file)

                with open(os.path.join(result_train_file, 'norm_stat.json'),'r') as f:
                    normalize_statistic = json.load(f)


                model = load_model(result_train_file)
                node_cnt = test_data.shape[1]
                test_set = ForecastDataset(test_data, window_size=window_size, horizon=horizon,
                                        normalize_method=norm_method, norm_statistic=normalize_statistic)
                test_loader = torch_data.DataLoader(test_set, batch_size=batch_size, drop_last=False,
                                                    shuffle=False, num_workers=0)
                performance_metrics = validate(model, test_loader, device, norm_method, normalize_statistic,
                                node_cnt, window_size, horizon, result_file=result_test_file)
                print('performance_metrics', performance_metrics)
                mae, mape, rmse = performance_metrics['mae'], performance_metrics['mape'], performance_metrics['rmse']
                mae_node, mape_node, rmse_node = performance_metrics['mae_node'].mean(), performance_metrics['mape_node'].mean(), performance_metrics['rmse_node'].mean()
                # print('Performance on ', f'{dataset}_{agg_type} ','test set: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mape, mae, rmse))
                # print('Performance on ', f'{dataset}_{agg_type} by node','test set: MAPE: {:5.2f} | MAE: {:5.2f} | RMSE: {:5.4f}'.format(mape_node, mae_node, rmse_node))

                print('Performance on ', f'{dataset}_{agg_type} ','test set: RMSE: {:5.4f} | MAPE: {:5.2f} | MAE: {:5.2f} '.format(rmse, mape, mae))
                print('Performance on ', f'{dataset}_{agg_type} by node','test set: RMSE: {:5.4f}| MAPE: {:5.2f} | MAE: {:5.2f} '.format(rmse_node, mape_node, mae_node))
                print(rmse, mape, mae)
                print(rmse_node, mape_node, mae_node)



            # parser = argparse.ArgumentParser()
            # parser.add_argument('--train', type=bool, default=True)
            # parser.add_argument('--evaluate', type=bool, default=True)
            # parser.add_argument('--dataset', type=str, default='ECG_data')
            # parser.add_argument('--window_size', type=int, default=12)
            # parser.add_argument('--horizon', type=int, default=3)
            # parser.add_argument('--train_length', type=float, default=7)
            # parser.add_argument('--valid_length', type=float, default=2)
            # parser.add_argument('--test_length', type=float, default=1)
            # parser.add_argument('--epoch', type=int, default=50)
            # parser.add_argument('--lr', type=float, default=1e-4)
            # parser.add_argument('--multi_layer', type=int, default=5)
            # parser.add_argument('--device', type=str, default='cpu')
            # parser.add_argument('--validate_freq', type=int, default=1)
            # parser.add_argument('--batch_size', type=int, default=32)
            # parser.add_argument('--norm_method', type=str, default='z_score')
            # parser.add_argument('--optimizer', type=str, default='RMSProp')
            # parser.add_argument('--early_stop', type=bool, default=False)
            # parser.add_argument('--exponential_decay_step', type=int, default=5)
            # parser.add_argument('--decay_rate', type=float, default=0.5)
            # parser.add_argument('--dropout_rate', type=float, default=0.5)
            # parser.add_argument('--leakyrelu_rate', type=int, default=0.2)





            # RTFMP_train_data_equitemp = RTFMP_1000_equitemp[:int(0.7 * len(RTFMP_1000_equitemp))]
            # RTFMP_valid_data_equitemp = RTFMP_1000_equitemp[int(0.7 * len(RTFMP_1000_equitemp)):int((0.7 + 0.1) * len(RTFMP_1000_equitemp))]
            # RTFMP_test_data_equitemp = RTFMP_1000_equitemp[int(0.3 * len(RTFMP_1000_equitemp)):]



            # train(train_data = train_dataset, 
            #             valid_data = valid_dataset, 
            #             # args, 
            #             window_size = window_size,
            #             multi_layer = 5,
            #             horizon = horizon,
            #             epoch = 50,
            #             lr = 0.00001,
            #             batch_size = 32,
            #             exponential_decay_step = 5,
            #             validate_freq = 1,
            #             early_stop = False,
            #             # early_stop_step,
            #             decay_rate = 0.5,
            #             norm_method = 'z_score',
            #             optimizer = 'RMSProp',
            #             result_file = f'{result_train_fold}/{dataset}_{no_intervals_all}_{agg_type}'
            #             ) 



            test(test_data = test_dataset, 
                    # args, 
                    window_size = window_size,
                    horizon = horizon,
                    norm_method = 'z_score',
                    batch_size = 32,
                    result_train_file = f'{result_train_fold}/{dataset}_{no_intervals_all}_{agg_type}',
                    result_test_file = f'{result_test_fold}/{dataset}_{no_intervals_all}_{agg_type}'
                    )
            # f'{dataset}_input_{no_intervals_all}_{agg_type}

print("all finish!!!")
