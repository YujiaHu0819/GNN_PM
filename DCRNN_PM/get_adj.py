import pickle
import numpy as np


datasets = ['RTFMP', 'italian', 'Sepsis', 'InternationalDeclarations', 'DomesticDeclarations', 'BPI2019', 'BPI2018', 'BPI2017', 'bpi12']
agg_type = ['equisize', 'equitemp']
no_intervals_all = [500, 1000, 3000]

def min_max_scaling(matrix):
                flattened_data = matrix.flatten()
                min_value = np.min(flattened_data)
                max_value = np.max(flattened_data)
                scaled_data = (flattened_data - min_value) / (max_value - min_value)
                scaled_matrix = scaled_data.reshape(matrix.shape)
                return scaled_matrix


def get_adj(file_path, output_path):
                data = np.load(file_path)
                adj = np.mean(data, axis=0)

                scaled_adj = min_max_scaling(adj)
                # print(adj)
                print('\n\n scaled \n\n', scaled_adj.shape)

                # with open(output_path, 'wb') as f:
                #     pickle.dump(scaled_adj, f)

                print('function done!!!')
                return scaled_adj


for dataset in datasets:
    for agg in agg_type:
        for num in no_intervals_all:

            file_path = f'/mnt/sdb/yujia/pmf/matrix/dfg_time_matrix_{dataset}_{num}_{agg}_org.npy'
            output_path = f'/mnt/sdb/yujia/DCRNN_PyTorch/matrix/{dataset}_{num}_{agg}.pkl'

            get_adj(file_path, output_path)

print('finished!!')