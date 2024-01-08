from .auxiliary import *
import glob
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm
import os
import os.path as op


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize MLP

        Parameters:
            input_size: int, input size
            output_size: int, output size
            hidden_size: tuple of hidden layer sizes, might be any length

        Input: 
            x: torch.Tensor, (*,input_size)

        Output:
            output: torch.Tensor, (*,output_size)
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size[0]))
        for i in range(len(hidden_size)-1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i+1]))
        self.layers.append(nn.Linear(hidden_size[-1], output_size))
        
    def forward(self, x):
        """
        Parameters:
            x: torch.Tensor, (*,input_size)

        Return:
            output: torch.Tensor, (*,output_size)
        """
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)
    

class PhaseProjection(nn.Module):
    """
    PhaseProjection model, which is a MLP model that maps slope to zernike polynomials

    API:
    ----------
    train: train model

    test: evaluate model on test set

    inference: inference zernike polynomials from slope

    save: save model to "path/mlp_ckpt_%d.pkl" % (epoch)

    load: load model from "path/mlp_ckpt_%d.pkl" % (epoch)

    """
    def __init__(self, device, hidden_size, num_zernike, mask_size, mask_path, epoch=10, lr=1e-4, batch_size=128, valid=True, split=0.9):
        """
        Initialize PhaseProjection

        Parameters:
        ----------
        device: torch.device

        hidden_size: Tuple[int], hidden layer sizes, might be any length

        num_zernike: int, number of zernike polynomials

        mask_size: Tuple[int], size of mask (n_view_x, n_view_y), should NOT be None

        mask_path: str, path of mask.mat, size of mask should be the same as mask_size; if None, automatically generate a circular mask with the same size as mask_size

        epoch: int, number of epochs to train

        lr: float, learning rate

        batch_size: int, batch size

        valid: bool, whether to split training set into training and validation set

        split: float, ratio of training set to the whole dataset, only valid when valid is True
        """
        super(PhaseProjection, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_zernike = num_zernike
        self.mask_size = mask_size
        self.mask_path = mask_path
        self.epoch = epoch
        self.lr = lr
        self.batch_size = batch_size
        self.valid = valid
        self.split = split

        self.X_mean, self.X_std, self.y_mean, self.y_std = None, None, None, None
        self.loss = []
        self.epoch_list = []
        self.R2_train = []
        self.R2_valid = []
        
        # set mask and valid views
        self.set_mask_valid_views(self.mask_path)
        self.input_dim = 2 * len(self.valid_views)

        # model
        self.model = MLP(self.input_dim, self.hidden_size, self.num_zernike).to(self.device)
    
    def set_mask_valid_views(self, mask_path):
        """
        Set mask and valid views

        Parameters:
            mask_path: str, path to mask.mat, if None, generate a circular mask with the same size as mask_size
        """
        if mask_path is not None:
            self.mask_path = mask_path
            self.mask, self.valid_views = get_mask(self.mask_path)
        else:
            if self.mask_path is not None:
                self.mask, self.valid_views = get_mask(self.mask_path)
            else:
                self.mask = torch.zeros(self.mask_size)
                self.mask[torch.sqrt((torch.arange(self.mask_size[0]) - self.mask_size[0] // 2).pow(2).unsqueeze(1) + (torch.arange(self.mask_size[1]) - self.mask_size[1] // 2).pow(2)) <= self.mask_size[0] // 2] = 1
                self.mask[self.mask < 0.5] = torch.nan
                n_view_x, n_view_y = self.mask_size
                self.valid_views = [i*n_view_y+j for i in range(n_view_x) for j in range(n_view_y) if not self.mask[i,j].isnan()]

    def reset_model(self):
        """
        Reset shiftmaps to zeros
        """
        self.model = MLP(self.input_dim, self.hidden_size, self.num_zernike).to(self.device)
        self.X_mean, self.X_std, self.y_mean, self.y_std = None, None, None, None
        self.loss = []
        self.epoch_list = []
        self.R2_train = []
        self.R2_test = []

    def shiftmap_v2f(self, shiftmap):
        """
        Convert shiftmap from valid views to full size

        Parameters:
            shiftmap: torch.Tensor, (n_valid_views,m,n,2)

        Return:
            shiftmap_full: torch.Tensor, (n_view_x,n_view_y,m,n,2)
        """
        return shiftmap_convertor_validviews2fullsize(shiftmap, self.mask, self.valid_views)
    
    def shiftmap_f2v(self, shiftmap_full):
        """
        Convert shiftmap from full size to valid views

        Parameters:
            shiftmap_full: torch.Tensor, (n_view_x,n_view_y,m,n,2)
        
        Return:
            shiftmap: torch.Tensor, (n_valid_views,m,n,2)
        """
        return shiftmap_convertor_fullsize2validviews(shiftmap_full, self.mask, self.valid_views)
    
    def train(self, data_path, epoch=None, lr=None, batch_size=None, valid=None, split=None):
        """
        Train model

        Parameters:
            data_path: str, path to data, should contain *_slope.mat and corresponding *_zernike.mat
            lr: float, learning rate, if None, use self.lr
            epoch: int, number of epochs to train, if None, use self.epoch
            batch_size: int, batch size, if None, use self.batch_size
            valid: whether to split training set into training and validation set, if None, use self.valid
            split: float, ratio of training set to the whole dataset, only valid when valid is True, if None, use self.split
        """
        if epoch is None:
            epoch = self.epoch
        if lr is None:
            lr = self.lr
        if batch_size is None:
            batch_size = self.batch_size
        if valid is None:
            valid = self.valid
        if split is None:
            split = self.split
        self.reset_model()

        # files
        slope_files = glob.glob(data_path + "/**/*_slope.mat", recursive=True)
        zernike_files = [file.replace("_slope.mat", "_zernike.mat") for file in slope_files]

        # load data    
        X = []
        y = []
        for slope_file, zernike_file in zip(slope_files, zernike_files):
            slope = torch.tensor(io.loadmat(slope_file)["slope"]).type(torch.float32)
            slope = self.shiftmap_f2v(slope)
            X.append(slope)
            zernike = torch.tensor(io.loadmat(zernike_file)["zernike"]).type(torch.float32)
            y.append(zernike)
        X = torch.stack(X, dim=0).permute(0,2,3,4,1).reshape(-1,2*len(self.valid_views)).to(self.device)
        y = torch.stack(y, dim=0).permute(0,2,3,1).reshape(-1,self.num_zernike).to(self.device)

        # split training set and validation set
        if valid:
            train_indices = np.random.choice(X.shape[0], int(X.shape[0] * split), replace=False)
            valid_indices = np.array(list(set(np.arange(X.shape[0])) - set(train_indices)))
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_valid = X[valid_indices]
            y_valid = y[valid_indices]
        else:
            X_train = X
            y_train = y
            X_valid = None
            y_valid = None
        
        # normalize
        self.X_mean = torch.mean(X_train, dim=0, keepdim=True)
        self.X_std = torch.std(X_train, dim=0, keepdim=True)
        self.y_mean = torch.mean(y_train, dim=0, keepdim=True)
        self.y_std = torch.std(y_train, dim=0, keepdim=True)
        X_train_norm = (X_train - self.X_mean) / self.X_std
        y_train_norm = (y_train - self.y_mean) / self.y_std
        if valid:
            X_valid_norm = (X_valid - self.X_mean) / self.X_std
            y_valid_norm = (y_valid - self.y_mean) / self.y_std
        
        # dataset
        train_dataset = TensorDataset(X_train_norm, y_train_norm)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        if valid:
            valid_dataset = TensorDataset(X_valid_norm, y_valid_norm)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        
        # training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_func = nn.MSELoss()
        for epoch_id in tqdm(range(epoch)):
            self.epoch_list.append(epoch_id)
            
            # train
            self.model.train()
            for _, (batch_x, batch_y) in enumerate(train_loader):
                pred = self.model(batch_x)
                loss = loss_func(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.loss.append(loss.item())

            # test on training set
            self.model.eval()
            R2_cum = 0
            Num = 0
            for _, (batch_x, batch_y) in enumerate(train_loader):
                pred = self.model(batch_x)
                pred = pred * self.y_std + self.y_mean
                batch_y = batch_y * self.y_std + self.y_mean
                R2_cum += torch.sum(1 - torch.sum((pred - batch_y)**2, dim=1) / torch.sum((batch_y - self.y_mean)**2, dim=1)).item()
                Num += batch_x.shape[0]
            self.R2_train.append(R2_cum / Num)

            # test on validation set
            if valid:
                R2_cum = 0
                Num = 0
                for _, (batch_x, batch_y) in enumerate(valid_loader):
                    pred = self.model(batch_x)
                    pred = pred * self.y_std + self.y_mean
                    batch_y = batch_y * self.y_std + self.y_mean
                    R2_cum += torch.sum(1 - torch.sum((pred - batch_y)**2, dim=1) / torch.sum((batch_y - self.y_mean)**2, dim=1)).item()
                    Num += batch_x.shape[0]
                self.R2_valid.append(R2_cum / Num)
            else:
                self.R2_valid.append(self.R2_train[-1])
            tqdm.write("epoch %d, train R2: %.6f, valid R2: %.6f"%(epoch_id, self.R2_train[-1], self.R2_valid[-1]))

    def test(self, data_path):
        """
        Evaluate model on test set

        Parameters:
            data_path: str, path to data, should contain *_slope.mat and corresponding *_zernike.mat
        """
        # files
        slope_files = glob.glob(data_path + "/**/*_slope.mat")
        zernike_files = [file.replace("_slope.mat", "_zernike.mat") for file in slope_files]

        # load data    
        X = []
        y = []
        for slope_file, zernike_file in zip(slope_files, zernike_files):
            slope = torch.tensor(io.loadmat(slope_file)["slope"]).type(torch.float32)
            slope = self.shiftmap_f2v(slope)
            X.append(slope)
            zernike = torch.tensor(io.loadmat(zernike_file)["zernike"]).type(torch.float32)
            y.append(zernike)
        X = torch.stack(X, dim=0).permute(0,2,3,4,1).reshape(-1,2*len(self.valid_views)).to(self.device)
        y = torch.stack(y, dim=0).permute(0,2,3,1).reshape(-1,self.num_zernike).to(self.device)
        
        # normalize
        X_norm = (X - self.X_mean) / self.X_std
        y_norm = (y - self.y_mean) / self.y_std

        # dataset
        test_dataset = TensorDataset(X_norm, y_norm)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # test
        self.model.eval()
        R2_cum = 0
        Num = 0
        for _, (batch_x, batch_y) in enumerate(test_loader):
            pred = self.model(batch_x)
            pred = pred * self.y_std + self.y_mean
            batch_y = batch_y * self.y_std + self.y_mean
            R2_cum += torch.sum(1 - torch.sum((pred - batch_y)**2, dim=1) / torch.sum((batch_y - self.y_mean)**2, dim=1)).item()
            Num += batch_x.shape[0]
        R2_test = R2_cum / Num
        print("test R2: %.6f on %s\n"%(R2_test, data_path))

    def inference(self, data_path):
        """
        Inference zernike polynomials from slope

        Parameters:
            data_path: str, path to data, should contain *_slope.mat
        
        Result:
            save coresponding zernike to *_pp_zernike.mat
        """
        # files
        slope_files = glob.glob(data_path + "/**/*_slope.mat", recursive=True)
        zernike_files = [file.replace("_slope.mat", "_mlp_zernike.mat") for file in slope_files]

        # inference
        for slope_file, zernike_file in zip(slope_files, zernike_files):
            slope = self.shiftmap_f2v(torch.tensor(io.loadmat(slope_file)["slope"]).type(torch.float32))
            b, m, n, _ = slope.shape
            slope = slope.permute(1,2,3,0).reshape(m*n,2*b).to(self.device)
            slope_norm = (slope - self.X_mean) / self.X_std
            zernike = self.model(slope_norm).reshape(m,n,self.num_zernike).permute(2,0,1).detach().cpu().numpy()
            io.savemat(zernike_file, {"zernike": zernike})
        print("inference done!\n")

    def save(self, path):
        """
        Save model to "path/mlp_ckpt_%d.pkl" % (epoch)

        Parameters:
            path: str, path to save model
        """
        if not op.exists(path):
            os.makedirs(path)
        torch.save({"model": self.model.state_dict(),
                    "X_mean": self.X_mean.cpu(),
                    "X_std": self.X_std.cpu(),
                    "y_mean": self.y_mean.cpu(),
                    "y_std": self.y_std.cpu(),
                    "loss": self.loss,
                    "epoch_list": self.epoch_list,
                    "R2_train": self.R2_train,
                    "R2_valid": self.R2_valid}, path + "/mlp_ckpt_%d.pkl" % (self.epoch_list[-1]))
        print("Saved model to %s" % (path + "mlp_ckpt_%d.pkl" % (self.epoch_list[-1])))
    
    def load(self, path, epoch, model_only=False):
        """
        Load model from "path/mlp_ckpt_%d.pkl" % (epoch)

        Parameters:
            path: str, path to load model
            epoch: int, epoch to load
            model_only: bool, whether to load only model
        """
        ckpt_path = path + "/mlp_ckpt_%d.pkl" % (epoch)
        if not op.exists(ckpt_path):
            raise Exception("ckpt not exist!")
        ckpt = torch.load(ckpt_path)
        self.model.load_state_dict(ckpt["model"])
        self.X_mean = ckpt["X_mean"].to(self.device)
        self.X_std = ckpt["X_std"].to(self.device)
        self.y_mean = ckpt["y_mean"].to(self.device)
        self.y_std = ckpt["y_std"].to(self.device)
        if not model_only:
            self.loss.clear()
            self.loss += ckpt["loss"]
            self.epoch_list.clear()
            self.epoch_list += ckpt["epoch_list"]
            self.R2_train.clear()
            self.R2_train += ckpt["R2_train"]   
            self.R2_valid.clear()
            self.R2_valid += ckpt["R2_valid"]
        print("Loaded model from %s" % (ckpt_path))



