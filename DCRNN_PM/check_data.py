# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import argparse
import numpy as np
import os
import pandas as pd


def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=False, add_day_in_week=False, scaler=None
):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param add_time_in_day:
    :param add_day_in_week:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, num_nodes = df.shape[0], df.shape[1]
    print('df shape: \n', num_samples, num_nodes)

    # data = np.expand_dims(df.values, axis=-1)
    # print('data: \n', data.shape)
    # print('\n', data)

    # data_list = [data]

    # print('data list \n'), 
    # print('\n', data_list)

    # if add_time_in_day:
    #     time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    #     time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
    #     data_list.append(time_in_day)
    #     print('time in day \n', time_in_day.shape)
    #     print('\n', time_in_day)
    # print('data list 2 in append \n', data_list)
    # if add_day_in_week:
    #     day_in_week = np.zeros(shape=(num_samples, num_nodes, 7))
    #     day_in_week[np.arange(num_samples), :, df.index.dayofweek] = 1
    #     data_list.append(day_in_week)

    # data = np.concatenate(data_list, axis=-1)

    # print('data after concat: \n', data.shape, data) #(52116, 325, 2)


    data = df


    # epoch_len = num_samples + min(x_offsets) - max(y_offsets)
    x, y = [], []
    # t is the index of the last observation.
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive

    print('min t: \n', min_t, '\n max t: \n', max_t)

    for t in range(min_t, max_t):
        x_t = data[t + x_offsets, ...]
        y_t = data[t + y_offsets, ...]
        x.append(x_t)
        y.append(y_t)

    print(x_offsets)
    print(y_offsets)
    print(len(x), len(y))
    # print('x value \n', x[:5], '\n')
    # print('y value \n', y[:5])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)

    print(x.shape, y.shape)

    return x, y


def generate_train_val_test(file_path, output_dir, window, horizon):
    # df = pd.read_hdf(file_path)
    df = np.load(file_path) 
    # 0 is the latest observed sample.

    x_offsets = np.sort(
        # np.concatenate(([-week_size + 1, -day_size + 1], np.arange(-11, 1, 1)))
        np.concatenate((np.arange(1-window, 1, 1),))
    )

    # Predict the next one hour
    y_offsets = np.sort(np.arange(1, horizon+1, 1))

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets
    )

    print("x shape: ", x.shape, ", y shape: ", y.shape)
    # Write the data into npz file.
    # num_test = 6831, using the last 6831 examples as testing.
    # for the rest: 7/8 is used for training, and 1/8 is used for validation.
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.23)
    num_train = round(num_samples * 0.75)
    num_val = num_samples - num_test - num_train

    # train
    x_train, y_train = x[:num_train], y[:num_train]
    # val
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    # test
    x_test, y_test = x[-num_test:], y[-num_test:]

    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)

        # np.savez_compressed(
        #     os.path.join(output_dir, "%s.npz" % cat),
        #     x=_x,
        #     y=_y,
        #     x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        #     y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        # )


datasets = ['RTFMP', 'italian', 'Sepsis', 'InternationalDeclarations', 'DomesticDeclarations', 'BPI2019', 'BPI2018', 'BPI2017', 'bpi12']
num = 3000
agg_type =['equisize', 'equitemp']

window_size = 45
horizon_num = 15

for dataset in datasets:
     for agg in agg_type:

        if not os.path.exists(f'./data_pm/{dataset}_{num}_{agg}_{window_size}/'):
                os.makedirs(f'./data_pm/{dataset}_{num}_{agg}_{window_size}/')



        data_path = f'/mnt/sdb/yujia/pmf/matrix/dfg_time_matrix_{dataset}_{num}_{agg}_org.npy'
                
        # data_path = '/mnt/sdb/yujia/DCRNN_PyTorch/pems-bay.h5'
        output_dir = f'/mnt/sdb/yujia/DCRNN_PyTorch/data_pm/{dataset}_{num}_{agg}_{window_size}/'

        generate_train_val_test(data_path, output_dir, window=window_size, horizon=horizon_num)


print("finish!!!")
# def main(args):
#     print("Generating training data")
#     generate_train_val_test(args)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--output_dir", type=str, default="data/", help="Output directory."
#     )
#     parser.add_argument(
#         "--file_path",
#         type=str,
#         default="data/metr-la.h5",
#         help="Raw traffic readings.",
#     )
#     args = parser.parse_args()
#     main(args)
