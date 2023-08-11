from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import yaml

from lib.utils import load_graph_data
# from model.pytorch.dcrnn_supervisor import DCRNNSupervisor
from model.pytorch.latent_supervisor import DCRNNSupervisor


import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


datasets = ['RTFMP', 'italian', 'Sepsis', 'InternationalDeclarations', 'DomesticDeclarations', 'BPI2019', 'BPI2018', 'BPI2017', 'bpi12']
agg_type = ['equisize', 'equitemp']
no_intervals_all = [500, 1000, 3000]

for dataset in datasets:
    for agg in agg_type:
        for num in no_intervals_all:

            config_filename = f'/mnt/sdb/yujia/DCRNN_PyTorch/yaml_100/{dataset}_{agg}_{num}.yaml'

            def main(args):
                with open(config_filename) as f:
                    supervisor_config = yaml.load(f, Loader=yaml.FullLoader)

                    # graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
                    # # sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
                    # adj_mx = load_graph_data(graph_pkl_filename)

                    # print('supervisor_config \n', supervisor_config)
                    # print('graph_pkl_filename \n', graph_pkl_filename)
                    # print('adj_mx shape : \n', adj_mx.shape)
                    # print(adj_mx)

                    supervisor = DCRNNSupervisor(**supervisor_config)

                    supervisor.train()


            if __name__ == '__main__':
                parser = argparse.ArgumentParser()
                parser.add_argument('--config_filename', default='./data/model/dcrnn_bay.yaml', type=str,
                                    help='Configuration filename for restoring the model.')
                parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
                args = parser.parse_args()
                main(args)
