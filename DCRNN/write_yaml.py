import yaml

# # Example YAML data
# yaml_data = {
#     'name': 'John Doe',
#     'age': 30,
#     'email': 'johndoe@example.com'
# }


def generate_yaml_file(data, filename):
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)



datasets = ['RTFMP', 'italian', 'Sepsis', 'InternationalDeclarations', 'DomesticDeclarations', 'BPI2019', 'BPI2018', 'BPI2017', 'bpi12']
agg_type = ['equisize', 'equitemp']
no_intervals_all = [3000]
node = [13, 18, 18, 36, 19, 44, 43, 28, 26]
horizon = 15
sequence_len = 45

step_list = [20, 30, 40, 50]
# [5, 10, 15, 20]

for agg in agg_type:
    for num in no_intervals_all:
        for i in range(len(datasets)):
            dataset = datasets[i]
            node_num = node[i]

            yaml_data = {
                'base_dir': 'data/model/model',
                'log_level': 'INFO',
                'data': {
                    'batch_size': 8,
                    'dataset_dir': f'/mnt/sdb/yujia/DCRNN_PyTorch/data_pm/{dataset}_{num}_{agg}_{sequence_len}',
                    'test_batch_size': 8,
                    'val_batch_size': 8,
                    'graph_pkl_filename': f'/mnt/sdb/yujia/DCRNN_PyTorch/matrix/italian_{num}_{agg}.pkl',
                    'dataset': f'{dataset}',
                    'agg_type': f'{agg}',
                    'intervals': num
                    },
                

                'model': {
                    'cl_decay_steps': 2000,
                    'filter_type': 'dual_random_walk',
                    #
                    'horizon': horizon,
                    #
                    'input_dim': node_num,
                    'l1_decay': 0,
                    'max_diffusion_step': 2,
                    #
                    'num_nodes': node_num,
                    'num_rnn_layers': 2,
                    'output_dim': 1,
                    'rnn_units': 64,
                    #
                    'seq_len': sequence_len,
                    'use_curriculum_learning': True
                        },
                

                'train': {
                    'base_lr': 0.01,
                    'dropout': 0,
                    'epoch': 0,
                    'epochs': 100,
                    'epsilon': 1.0e-3,
                    'global_step': 0,
                    'lr_decay_ratio': 0.1,
                    'max_grad_norm': 5,
                    'max_to_keep': 100,
                    'min_learning_rate': 1.0e-06,
                    'optimizer': 'adam',
                    'patience': 50,
                    'steps': step_list,
                    # steps: [20, 30, 40, 50]
                    #
                    'test_every_n_epochs': 5,
                    'log_dir': f'/mnt/sdb/yujia/DCRNN_PyTorch/log_latent/{dataset}_{num}_{agg}/'
                }
                
            }

            output_path = f'/mnt/sdb/yujia/DCRNN_PyTorch/yaml_100/{dataset}_{agg}_{num}.yaml'

            generate_yaml_file(yaml_data, output_path)





