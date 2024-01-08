import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from .utils import ZernikeDataset, LossFn, PhaseMetric, WarmupCosineAnnealingLR
from .auxiliary import *
from tqdm import tqdm
import scipy.io as io


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell

        Parameters: 
            input_dim: int, number of channels of input tensor
            hidden_dim: int, number of channels of hidden state
            kernel_size: (int, int), size of the convolutional kernel
            bias: bool, whether or not to add the bias
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1) 
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Generate a multi-layer convolutional LSTM
    
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        torch.Tensor, (b,t,c,h,w) or (t,b,c,h,w)

    Output: (A tuple of two lists of length num_layers, or length 1 if return_all_layers is False)
        layer_output_list: list of lists of length T of each output
        last_state_list: list of last states, each element is tuple (h, c) for hidden state and memory
    """

    def __init__(self, input_dim=32, hidden_dim=32, kernel_size=(3,3), num_layers=5,
                 batch_first=True, bias=True, return_all_layers=True, dropout=0):
        """
        Initialize ConvLSTM network
        
        Parameters:
            input_dim: int, number of channels of input tensor
            hidden_dim: int, number of channels of hidden state
            kernel_size: (int, int), size of the convolutional kernel
            num_layers: int, number of layers of ConvLSTM
            batch_first: bool, whether or not dimension 0 is the batch or not
            bias: bool, whether or not to add the bias
            return_all_layers: bool, whether or not to return all layers
            dropout: float, dropout rate    
        """
        super(ConvLSTM, self).__init__()
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.ModuleList(cell_list)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor):
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            if layer_idx == self.num_layers - 1:
                layer_output = self.dropout(layer_output)
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list) or (isinstance(param, list) and len(param) == 1):
            param = [param] * num_layers
        return param


class ResConvLSTM(nn.Module):
    """
    Generate a multi-layer convolutional LSTM with residual connection

    Parameters:
    ----------
    n_dim: list of list of int
        each list contains dimensions of one block, of which the first element is input dimension and the rest are hidden dimensions for each layer
    kernel_size: list of list of int
        each list contains kernel sizes of one block, of which one element is kernel size for each layer
    dropout: float
        dropout rate

    Input:
    ----------
    input_data: torch.Tensor, (b,t,c,h,w)

    Output:
    ----------
    output_data: torch.Tensor, (b,t,c,h,w)
    """

    def __init__(self, n_dim=[[32,128,32]], kernel_size=[[5,3]], dropout=0.2):
        """
        Initialize ResConvLSTM network

        Parameters:
        ----------
        n_dim: list of list of int
            each list contains dimensions of one block, of which the first element is input dimension and the rest are hidden dimensions for each layer
        kernel_size: list of list of int
            each list contains kernel sizes of one block, of which one element is kernel size for each layer
        dropout: float
            dropout rate
        """
        super(ResConvLSTM, self).__init__()
        self.n_dim = n_dim
        self.kernel_size = kernel_size
        self.n_layers = [len(k_list) for k_list in kernel_size]
        self.dropout = dropout

        cells = []
        convs = []
        for i in range(len(self.n_layers)):
            cells.append(ConvLSTM(n_dim[i][0], n_dim[i][1:], [(k,k) for k in kernel_size[i]], self.n_layers[i], dropout=dropout))
            if n_dim[i][-1] != n_dim[i][0]:
                convs.append(nn.Conv2d(in_channels=n_dim[i][0], out_channels=n_dim[i][-1], kernel_size=(1,1)))
            else:
                convs.append(nn.Identity())
        self.lstm = nn.Sequential(*cells)
        self.conv = nn.Sequential(*convs)
    
    def forward(self, input_data):
        for cell_num in range(len(self.lstm)):
            output, _ = self.lstm[cell_num](input_data)
            output = output[-1]
            input_data = torch.stack([self.conv[cell_num](input_data[:,t,:,:,:]) for t in range(input_data.shape[1])], dim=1)
            input_data = output + input_data
        return input_data
    

class PhasePredictionModel(nn.Module):
    """
    Generate a phase prediction model

    Input:
    ----------
    input_data: torch.Tensor, (b,t,c,input_h,input_w)

    Output:
    ----------
    output_data: torch.Tensor, (b,t,c,output_h,output_w)
    """
    def __init__(self, device=None, norm=1, **kwargs):
        """
        Initialize phase prediction model
        
        Parameters:
        ----------
        device: torch.device
            device to run the model
        norm: float
            used to normalize the input data, default = 1
        kwargs: dict
            kwargs["dataset"]:
                input_patch: list, [min_h, max_h, min_w, max_w]
                    choose patch for input data, used as indices [min_h:max_h, min_w:max_w]
                target_patch: list, [min_h, max_h, min_w, max_w]
                    choose patch for target data, used as indices [min_h:max_h, min_w:max_w]
                t_series: int
                    length of input sequence
            kwargs["model"]:
                n_dim: list of list of int
                    each list contains dimensions of one block, of which the first element is input dimension and the rest are hidden dimensions for each layer
                kernel_size: list of list of int
                    each list contains kernel sizes of one block, of which one element is kernel size for each layer
                dropout: float
                    dropout rate
        """
        super().__init__()

        self.device = device
        self.norm = norm
        self.kwargs = kwargs

        self.input_patch = self.kwargs["dataset"]["input_patch"]
        self.input_size = (self.input_patch[1] - self.input_patch[0], self.input_patch[3] - self.input_patch[2])
        self.output_patch = kwargs["dataset"]["target_patch"]
        self.output_size = (self.output_patch[1] - self.output_patch[0], self.output_patch[3] - self.output_patch[2])
        self.t_series = kwargs["dataset"]["t_series"]

        self.backbone = ResConvLSTM(n_dim=self.kwargs["model"]["n_dim"], 
                                    kernel_size=self.kwargs["model"]["kernel_size"], 
                                    dropout=self.kwargs["model"]["dropout"])
        self.head = nn.Linear(self.input_size[0] * self.input_size[1], self.output_size[0] * self.output_size[1])

    def forward(self, input_data):
        """
        Forward pass of the model

        Parameters:
            Input: torch.Tensor, (b,t,c,input_h,input_w)
        
        Returns:
            Output: torch.Tensor, (b,t,c,output_h,output_w)
        """
        output = self.backbone(input_data / self.norm)
        b, t, c, h, w = output.shape
        output = self.head(output.reshape(b, t, c, h*w)).reshape(b, t, c, self.output_size[0], self.output_size[1])
        output = output * self.norm

        return output


class PhasePrediction(nn.Module):
    """
    Generate a phase prediction application
    """
    def __init__(self, **kwargs):
        """
        Initialize phase prediction application

        Parameters:
        ----------
        kwargs: dict, example:
            {
                "info": {
                    "name": "17t7"
                },
                "model": {
                    "n_dim": [[32,128,128],[128,32,32,32,32,32]], 
                    "kernel_size": [[5,5],[3,3,3,3,3]], 
                    "dropout": 0.2
                },
                "optim": {
                    "lr": 3e-4, 
                    "weight_decay": 5e-4
                },
                "scheduler": {
                    "eta_min": 3e-5
                },
                "criterion": {
                    "loss_type": "l1", 
                    "reduction": "mean"
                },
                "dataset": {
                    "data_path": "./data/phase_data/230407/", 
                    "zernike_phase_path": "./data/settings/",
                    "phase_size": 15, 
                    "n_channel": [3,34], 
                    "split": 0.9, 
                    "t_series": 5, 
                    "t_offset": 1, 
                    "t_down": 0,
                    "all_patch":[0,19,0,25], 
                    "input_patch":[0,17,7,24], 
                    "target_patch":[5,12,12,19], 
                    "test_size": [7,7]
                },
                "config": {
                    "device_id": 0, 
                    "seed": 42, 
                    "batch_size": 32,
                    "epoch": 200, 
                    "virtual_epoch":200, 
                    "warmup_epoch": 5,
                    "save_epoch": 1, 
                    "eval_epoch": 1, 
                    "plot_epoch": 1
                }
            }
        """
        super(PhasePrediction, self).__init__()
        self.kwargs = kwargs

        random_seed(seed=self.kwargs["config"]["seed"])
        self.root = "./log/rclstm_%s/"%(kwargs["info"]["name"])
        self.device = try_gpu(kwargs["config"]["device_id"])

        ZernikeDataset.reset()
        self.train_ds = ZernikeDataset(self.device, train=True, **kwargs)
        self.train_dl = DataLoader(self.train_ds, batch_size=kwargs["config"]["batch_size"], shuffle=True)
        self.test_ds = ZernikeDataset(self.device, train=False, **kwargs)
        self.test_dl = DataLoader(self.test_ds, batch_size=kwargs["config"]["batch_size"], shuffle=False)

        self.all_patch = self.kwargs["dataset"]["all_patch"]
        self.input_patch = self.kwargs["dataset"]["input_patch"]
        self.target_patch = self.kwargs["dataset"]["target_patch"]
        self.loss, self.epoch = [], []
        self.train_R2, self.train_rmse, self.train_sigma = [], [], []
        self.test_R2, self.test_rmse, self.test_sigma = [], [], []
        
        self.model = PhasePredictionModel(self.device, ZernikeDataset.norm,**self.kwargs).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.kwargs['optim']["lr"], weight_decay=self.kwargs['optim']["weight_decay"])
        self.scheduler = WarmupCosineAnnealingLR(self.optimizer, len(self.train_dl), **self.kwargs)
        self.criterion = LossFn(**self.kwargs)
        self.metric = PhaseMetric(**self.kwargs)
        
        makedirs(self.root + "fig/")
        makedirs(self.root + "ckpt/")
        save_json(self.kwargs, self.root + "config.json")

    def ckpt_path_gen(self, epoch):
        return self.root + 'ckpt/turb_checkpoint_%d_epoch.pkl'%(epoch)
    
    def load(self, epoch):
        ckpt_path = self.ckpt_path_gen(epoch)
        if not op.exists(ckpt_path):
            log_string = 'No checkpoint found at {}\n'.format(ckpt_path)
            log_txt(log_string, self.root + "log.txt")
            tqdm.write(log_string)
            return False
        
        checkpoint = torch.load(ckpt_path)
        self.loss.clear()
        self.loss += checkpoint['loss']
        self.epoch.clear()
        self.epoch += checkpoint['epoch']
        self.train_R2.clear()
        self.train_R2 += checkpoint['train_R2']
        self.train_rmse.clear()
        self.train_rmse += checkpoint['train_rmse']
        self.train_sigma.clear()
        self.train_sigma += checkpoint['train_sigma']
        self.test_R2.clear()
        self.test_R2 += checkpoint['test_R2']
        self.test_rmse.clear()
        self.test_rmse += checkpoint['test_rmse']
        self.test_sigma.clear()
        self.test_sigma += checkpoint['test_sigma']
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        log_string = 'Loaded model from {}\n'.format(ckpt_path)
        log_txt(log_string, self.root + "log.txt")
        tqdm.write(log_string)
        return True
    
    def save(self):
        ckpt_path = self.ckpt_path_gen(self.epoch[-1])
        save_dict = {'loss': self.loss, 'epoch': self.epoch, 'train_R2': self.train_R2,
                     'train_rmse': self.train_rmse, 'train_sigma': self.train_sigma,
                     'test_R2': self.test_R2, 'test_rmse': self.test_rmse, 'test_sigma': self.test_sigma,
                     "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}
        torch.save(save_dict, ckpt_path)

        log_string = 'Saved model to {}\n'.format(ckpt_path)
        log_txt(log_string, self.root + "log.txt")
        tqdm.write(log_string)
        return True
    
    def train_one_epoch(self):
        self.model.train()
        batch_num = len(self.train_dl)
        self.epoch.append(0 if len(self.epoch)==0 else self.epoch[-1] + 1)
        for batch_id, (input, target, _) in enumerate(self.train_dl):
            self.optimizer.zero_grad()
            output = self.model(input)
            l = self.criterion(output, target)
            l.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.loss.append((self.epoch[-1] * batch_num + batch_id, l.item()))
    
    def eval(self, train=False, log=True):
        """
        Parameters:
            train: bool, whether to evaluate on train dataset or test dataset
            log: bool, whether to log the result to log.txt
        """
        self.model.eval()

        with torch.no_grad():
            sum_R2 = 0
            sum_rmse = 0
            sum_sigma = 0
            total_num = 0
            for input, target, _ in (self.train_dl if train else self.test_dl):
                ret_dict = self.metric(self.model(input), target, scalar=True)
                sum_R2 += ret_dict['R2'] * input.shape[0]
                sum_rmse += ret_dict['RMSE'] * input.shape[0]
                sum_sigma += ret_dict['SIGMA'] * input.shape[0]
                total_num += input.shape[0]
            average_R2 = sum_R2 / total_num
            average_rmse = sum_rmse / total_num
            average_sigma = sum_sigma / total_num

        if train:
            self.train_R2.append((self.epoch[-1], average_R2))
            self.train_rmse.append((self.epoch[-1], average_rmse))
            self.train_sigma.append((self.epoch[-1], average_sigma))
            log_string = "epoch:{} - TRAIN R2:{:.4f} RMSE:{:.4f} SIGMA:{:.4f}\n".format(
                          self.epoch[-1], average_R2, average_rmse, average_sigma)
            if log:
                log_txt(log_string, self.root + "log.txt")
            tqdm.write(log_string)

        else:
            self.test_R2.append((self.epoch[-1], average_R2))
            self.test_rmse.append((self.epoch[-1], average_rmse))
            self.test_sigma.append((self.epoch[-1], average_sigma))
            log_string = "epoch:{} - TEST  R2:{:.4f} RMSE:{:.4f} SIGMA:{:.4f}\n".format(
                          self.epoch[-1], average_R2, average_rmse, average_sigma)
            if log:
                log_txt(log_string, self.root + "log.txt")
            tqdm.write(log_string)

    def delete_ckpt(self, n_epoch):
        test_R2_data = np.array(self.test_R2)
        best_id = np.argmax(test_R2_data[:,1])
        best_epoch_R2 = test_R2_data[best_id,0].item()

        test_rmse_data = np.array(self.test_rmse)
        best_id = np.argmin(test_rmse_data[:,1])
        best_epoch_rmse = test_rmse_data[best_id,0].item()

        test_sigma_data = np.array(self.test_sigma)
        best_id = np.argmin(test_sigma_data[:,1])
        best_epoch_sigma = test_sigma_data[best_id,0].item()

        preserve_epoch = [best_epoch_R2, best_epoch_rmse, best_epoch_sigma, n_epoch - 1]

        for i in range(n_epoch):
            if i in preserve_epoch:
                continue
            ckpt_path = self.ckpt_path_gen(i)
            if op.exists(ckpt_path):
                os.remove(ckpt_path)
    
    def train(self, eval_model=True, save_model=True, plot_metric=True, remove_ckpt=True):
        epoch_to_train = list(range(self.kwargs['config']['epoch']))

        for epoch_id in tqdm(epoch_to_train):

            self.train_one_epoch()

            if eval_model:
                if (epoch_id + 1) % self.kwargs['config']['eval_epoch'] == 0:
                    self.eval(train=True, log=True)
                    self.eval(train=False, log=True)
            
            if save_model:
                if (epoch_id + 1) % self.kwargs['config']['save_epoch'] == 0:
                    self.save()

            if plot_metric:
                if (epoch_id + 1) % self.kwargs['config']['plot_epoch'] == 0:
                    plot_loss_list(self.loss, self.root + 'fig/Loss.png', dpi=200, yaxis=None)
                    plot_metric_list(self.train_R2, self.test_R2, "R2", self.root + 'fig/R2.png', dpi=200, yaxis=(0,1), max_best=True)
                    plot_metric_list(self.train_rmse, self.test_rmse, "RMSE", self.root + 'fig/RMSE.png', dpi=200, yaxis=None, max_best=False)
                    plot_metric_list(self.train_sigma, self.test_sigma, "SIGMA", self.root + 'fig/SIGMA.png', dpi=200, yaxis=None, max_best=False)

        if remove_ckpt:
            self.delete_ckpt(self.kwargs['config']['epoch'])

    def test(self, phase_size=55, best_metric="RMSE", save_test_zernike=True, plot_test_zernike=True):
        # find best epoch
        self.load(self.kwargs['config']['epoch'] - 1)
        if best_metric == "RMSE":
            best_epoch = find_best_epoch(self.test_rmse, "RMSE", self.kwargs["config"]["save_epoch"], max_best=False)
        elif best_metric == "R2":
            best_epoch = find_best_epoch(self.test_R2, "R2", self.kwargs["config"]["save_epoch"], max_best=True)
        elif best_metric == "SIGMA":
            best_epoch = find_best_epoch(self.test_sigma, "SIGMA", self.kwargs["config"]["save_epoch"], max_best=False)
        else:
            raise Exception("best_metric should be RMSE or R2 or SIGMA!")
        log_string = "Found best epoch is %d by %s\n"%(best_epoch, best_metric)
        log_txt(log_string, self.root + "log.txt")
        tqdm.write(log_string)
        self.load(best_epoch)

        # test path
        test_path = self.root + "/test%d_ts%d/"%(self.kwargs["info"]["best_epoch"], self.kwargs["dataset"]["test_size"][0])
        makedirs(test_path + "/heatmap/")
        makedirs(test_path + "/zernike/")
        makedirs(test_path + "/phase_image/")

        # eval model on test dataset
        tqdm.write("Eval model on test dataset...\n")
        self.eval(train=False, log=False)

        # plot averaged heatmap
        tqdm.write("Plot averaged heatmap...\n")
        self.model.eval()
        test_size = self.kwargs["dataset"]["test_size"]
        with torch.no_grad():
            R2_cum = torch.zeros(test_size[0], test_size[1]).to(self.device)
            rmse_cum = torch.zeros(test_size[0], test_size[1]).to(self.device)
            sigma_cum = torch.zeros(test_size[0], test_size[1]).to(self.device)
            N = 0
            for input, target,  _ in self.test_dl:
                b = input.shape[0]
                N += b
                ret_dict = self.metric(self.model(input), target, scalar=False)
                R2_cum += ret_dict['R2'] * b
                rmse_cum += ret_dict['RMSE'] * b
                sigma_cum += ret_dict['SIGMA'] * b
            R2_cum = R2_cum / N
            rmse_cum = rmse_cum / N
            sigma_cum = sigma_cum / N
        plt.rcdefaults()
        plt.figure(dpi=200)
        plt.imshow(R2_cum.cpu().numpy(), cmap='jet')
        for i in range(test_size[0]):
            for j in range(test_size[1]):
                plt.text(j, i, '%.3f'%R2_cum[i,j], ha='center', va='center')
        plt.colorbar()
        plt.savefig(test_path +  "/heatmap/" + 'R2_heatmap.png')
        plt.close()       
        plt.rcdefaults()
        plt.figure(dpi=200)
        plt.imshow(rmse_cum.cpu().numpy(), cmap='jet')
        for i in range(test_size[0]):
            for j in range(test_size[1]):
                plt.text(j, i, '%.0f'%rmse_cum[i,j], ha='center', va='center')
        plt.colorbar()
        plt.savefig(test_path +  "/heatmap/" + 'RMSE_heatmap.png')
        plt.close()  
        plt.rcdefaults()
        plt.figure(dpi=200)
        plt.imshow(sigma_cum.cpu().numpy(), cmap='jet')
        for i in range(test_size[0]):
            for j in range(test_size[1]):
                plt.text(j, i, '%.3f'%sigma_cum[i,j], ha='center', va='center')
        plt.colorbar()
        plt.savefig(test_path +  "/heatmap/" + 'SIGMA_heatmap.png')
        plt.close()

        # save predicted zernike
        if save_test_zernike or plot_test_zernike:
            tqdm.write("Save predicted zernike...\n")
            self.model.eval()
            for input, target, target_file in self.test_ds:
                pred = self.model(input.unsqueeze(0)).squeeze(0)
                ret_dict = self.metric(pred.unsqueeze(0), target.unsqueeze(0), scalar=False)
                if save_test_zernike:
                    io.savemat(test_path + "/zernike/" + "_".join(target_file.replace("\\", "/").split("/")[-2:]).replace("_ds_", "_test_"),
                            {"pred": pred[-1].detach().cpu().numpy(), "gt": target[-1].detach().cpu().numpy(), "R2": ret_dict["R2"].detach().cpu().numpy(),
                                "RMSE": ret_dict["RMSE"].detach().cpu().numpy(), "SIGMA": ret_dict["SIGMA"].detach().cpu().numpy()})
                if plot_test_zernike:
                    n_channel = list(range(self.kwargs["dataset"]["n_channel"][0], self.kwargs["dataset"]["n_channel"][1]+1))
                    trans_dict = io.loadmat(self.kwargs["dataset"]["zernike_phase_path"] + "/zernike_phase%d.mat" % (phase_size))
                    Z2P = torch.from_numpy(trans_dict["Z2P"]).type(torch.float32).to(self.device)[n_channel,:,:]
                    plot_phase_img([pred[-1].detach(), Z2P], cmap="coolwarm", caxis=[-525,525],
                                    save_name=test_path + "/phase_image/" + "_".join(target_file.replace("\\", "/").split("/")[-2:]).replace("_ds_zernike.mat", "_pred_phase.png"))
                    plot_phase_img([target[-1].detach(), Z2P], cmap="coolwarm", caxis=[-525,525],
                                    save_name=test_path + "/phase_image/" + "_".join(target_file.replace("\\", "/").split("/")[-2:]).replace("_ds_zernike.mat", "_gt_phase.png"))


