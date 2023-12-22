import os
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from lib import utils
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.loss import masked_mae_loss, MAPE, MAE, RMSE


import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DCRNNSupervisor:
    def __init__(self, **kwargs):
        self.datasets = ['RTFMP', 'italian', 'Sepsis', 'InternationalDeclarations', 'DomesticDeclarations', 'BPI2019', 'BPI2018', 'BPI2017', 'bpi12']
        # self.dataset = 'RTFMP'
        # self.agg_type = 'equisize'
        # self.intervals = 500
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.dataset = self._data_kwargs.get('dataset')
        self.agg_type = self._data_kwargs.get('agg_type')
        self.intervals = self._data_kwargs.get('intervals')


        self.time_step = self._model_kwargs.get('seq_len')
        self.unit = self._model_kwargs.get('num_nodes')
        self.alpha = 0.2

        print(self.dataset, self.agg_type, self.intervals)


        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))

        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        self.GRU = nn.GRU(self.time_step, self.unit)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=0.2)
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists(f'models/{self.dataset}/{self.intervals}_{self.agg_type}/'):
            os.makedirs(f'models/{self.dataset}/{self.intervals}_{self.agg_type}/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.dcrnn_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, f'models/{self.dataset}/{self.intervals}_{self.agg_type}/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return f'models/{self.dataset}/{self.intervals}_{self.agg_type}/epo%d.tar' % epoch

    def load_model(self):
        self._setup_graph()
        assert os.path.exists(f'models/{self.dataset}/{self.intervals}_{self.agg_type}/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load(f'models/{self.dataset}/{self.intervals}_{self.agg_type}/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                break


    def self_graph_attention(self, input):
        # input shape here: (batch, sequence, output_size)
        input = input.permute(0, 2, 1).contiguous()
        # after trans：(batch, output_size, sequence)
        # this is why input == output ?
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        # key shape: torch.Size([32, 140, 1])
        query = torch.matmul(input, self.weight_query)
        # torch.repeat 当参数有三个时：（通道数的重复倍数，行的重复倍数，列的重复倍数）
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        # data shape:  torch.Size([32, 140 *140, 1])
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        # attention shape: torch.Size([32, 140, 140])
        return attention
    
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
        # multi_order_laplacian shape: torch.Size([4, 140, 140])
        
        multi_order_laplacian = torch.mean(multi_order_laplacian, axis=0)
        return multi_order_laplacian
    

    def latent_correlation_layer(self, x):
        # is there has a question ? self.windows
        # x: (batch, sequence, features)
        # GRU default input: (sequence, batch, features),  default batch_first=False
        # However, input is (features, batch, sequence) here.
        # torch.Size([140, 32, 12]) sequence(window_size)=12, but it equals 140 here.
        # print("----GRU input shape: ", x.permute(2, 0, 1).contiguous().shape)
        input, _ = self.GRU(x.contiguous())
        
        # input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        # print("----GRU output shape: ", input.shape)
        # last state output shape of GRU in doc: (D * num_layers, batch, output_size(self.unit))
        # However, (sequence, batch, D∗Hout(output_size)) when batch_first=False here ???
        # Only all output features is senseful in this situation
        # torch.Size([140, 32, 140])
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        # attention shape: torch.Size([140, 140])

        degree = torch.sum(attention, dim=1)
        # degree shape: torch.Size([140])
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        # 返回一个以degree为对角线元素的2D矩阵，torch.Size([140, 140])

        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        # diagonal_degree_hat shape: torch.Size([140, 140])
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        # laplacian shape: torch.Size([140, 140])
        mul_L = self.cheb_polynomial(laplacian)
        # mul_L shape: torch.Size([4, 140, 140])
        return mul_L, attention
    
    def MAPE(v, v_, axis=None):
        '''
        Mean absolute percentage error.
        :param v: np.ndarray or int, ground truth.
        :param v_: np.ndarray or int, prediction.
        :param axis: axis to do calculation.
        :return: int, MAPE averages on all elements of input.
        '''
        mape = (np.abs(v_ - v) / np.abs(v)+1e-5).astype(np.float64)
        mape = np.where(mape > 5, 5, mape)
        # mape = mape * 100
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

        
    # setup model
    # dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
    # self.dcrnn_model = dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model
    # self._logger.info("Model created")

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []

            y_truths = []
            y_preds = []

            maes = []
            mapes = []
            rmses = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output = self.dcrnn_model(x)
                loss, mae, mape, rmse = self._compute_loss(y, output)
                losses.append(loss.item())
                maes.append(mae.item())
                mapes.append(mape.item())
                rmses.append(rmse.item())

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            mean_loss = np.mean(losses)
            mae_loss = np.mean(maes)
            mape_loss = np.mean(mapes)
            rmse_loss = np.mean(rmses)
            # print('Before concat prediction', y_preds.shape, 'truth', y_truths.shape)

            self._writer.add_scalar('{} loss'.format(dataset), mean_loss, batches_seen)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

            print('prediction', y_preds.shape, 'truth', y_truths.shape)

            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)          
                # .cpu().detach().numpy()
                # mae = MAE(y_truth, y_pred)
                # mape = MAPE(y_truth, y_pred)
                # rmse = RMSE(y_truth, y_pred)
                # print(mae, mape, rmse)

            # print('After scaled prediction', y_preds_scaled.shape, 'truth', y_truths_scaled.shape)

            return mean_loss, mae_loss, mape_loss, rmse_loss, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}

    def _train(self, base_lr,
               steps, patience=50, epochs=20, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=5, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        
        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(self._epoch_num, epochs):

            # self.dcrnn_model = self.dcrnn_model.train()

            train_iterator = self._data['train_loader'].get_iterator()

            print("train iterator: \n", train_iterator)
            losses = []

            start_time = time.time()

            for _, (x0, y0) in enumerate(train_iterator):
                

                #  (64, 12, 325, 2) (64, 12, 325, 2)
                # print("before prepare x y shape :\n", x.shape, y.shape)
                x, y = self._prepare_data(x0, y0)

                latent_x, _ = self._prepare_latent(x0, y0)
                # print(latent_x.shape)
                mul_L, attention = self.latent_correlation_layer(latent_x)
                # print('mul shape', mul_L.shape)

                self.dcrnn_model = DCRNNModel(mul_L, self._logger, **self._model_kwargs)
                self.dcrnn_model = self.dcrnn_model.cuda() if torch.cuda.is_available() else self.dcrnn_model

                optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)

                optimizer.zero_grad()
    
                self.dcrnn_model = self.dcrnn_model.train()

                # x: ([12, 64, 650])
                # y: ([12, 64, 325])
                # print('train start x: \n', x.shape)
                # print('train start y \n: ', y.shape)

                output = self.dcrnn_model(x, y, batches_seen)

                #([12, 64, 325])
                # print('output shape \n', output.shape)

                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

                loss, _, _, _ = self._compute_loss(y, output)

                self._logger.debug(loss.item())

                losses.append(loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)

                optimizer.step()
            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")

            val_loss, val_mae_loss, val_mape_loss, val_rmse_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen)

            end_time = time.time()

            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, mae: {:.4f}, mape: {:.4f}, rmse: {:.4f},lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), val_loss,val_mae_loss, val_mape_loss, val_rmse_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, test_mae, test_mape, test_rmse, _ = self.evaluate(dataset='test', batches_seen=batches_seen)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f}, mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, epochs, batches_seen,
                                           np.mean(losses), test_loss, test_mae, test_mape, test_rmse, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message) 
                print(f'test metrics on {epoch_num}\n', test_rmse, test_mape, test_mae)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _prepare_latent(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        # print('_get_x_y shape: \n', x.shape, y.shape)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(2, 0, 1, 3)
        y = y.permute(2, 0, 1, 3)

        x = torch.mean(x, axis = 3)
        y = torch.mean(y, axis = 3)
        # print('_get_latent input shape: \n', x.shape, y.shape)
        return x, y
    
    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        # print('_get_x_y shape: \n', x.shape, y.shape)
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        # print('_get_x_y output shape: \n', x.shape, y.shape)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        
        # print('_get_x_y_in_correct output shape: \n', x.shape, y.shape)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        masked_mae = masked_mae_loss(y_predicted, y_true)
        # v.cpu().detach().numpy()
        mae = MAE(y_true.cpu().detach().numpy(), y_predicted.cpu().detach().numpy())
        mape = MAPE(y_true.cpu().detach().numpy(), y_predicted.cpu().detach().numpy())
        rmse = RMSE(y_true.cpu().detach().numpy(), y_predicted.cpu().detach().numpy())
        # print('metric in _compute_loss',  rmse, mape, mae)

        return masked_mae, mae, mape, rmse