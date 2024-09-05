import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random 
import os
import os.path as op
from matplotlib import pyplot as plt
import cv2   


def try_gpu(device_id=0):
    """
    Try to use GPU, if GPU is not available, use CPU instead

    Parameters:
        device_id: int, default=0, GPU device id

    Return:
        torch.device("cuda:device_id") if GPU is available else torch.device("cpu")
    """
    if torch.cuda.is_available() and torch.cuda.device_count() > device_id:
        torch.cuda.set_device("cuda:%d" % device_id)
        return torch.device("cuda:%d" % device_id)
    else:
        return torch.device("cpu")


def random_seed(seed=42):
    """
    set random seed for torch, numpy and random
    
    Parameters:
        seed: int, default=42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_json(jsonpath):
    """
    Read json file
    
    Parameters:
        jsonpath: str, json file path

    Return:
        dict, data in json file
    """
    return json.loads(open(jsonpath, 'r', encoding='utf-8').read())


def save_json(datadict, jsonpath):
    """
    Save dict to json file

    Parameters:
        datadict: dict, data to save
        jsonpath: str, json file path
    
    Return:
        bool, True if save successfully, else False
    """
    filejson = json.dumps(datadict, ensure_ascii=False, indent=1)
    fp = open(jsonpath, "w", encoding='utf-8', newline='\n')
    fp.write(filejson)
    fp.close()
    return True


def makedirs(path):
    """
    Create directory recursively if not exists

    Parameters:
        path: str, path to create
    
    Return:
        bool, True if path is created, else False
    """
    if not op.exists(path):
        os.makedirs(path)
    return True


def log_txt(logstr, logpath):
    """
    Log string to txt file
    
    Parameters:
        logstr: str, log string
        logpath: str, log file path
    """
    if not op.exists(logpath):
        with open(logpath, 'w') as f:
            f.write(logstr)
    else:
        with open(logpath, 'a') as f:
            f.write(logstr)


def plot_metric_list(train_data, test_data, name, save_name=None, dpi=300, yaxis=None, max_best=True, save=True, close=True, **kwargs):
    """
    Plot metric curve (R2, RMSE, SIGMA)

    Parameters:
        train_data: [(epoch, data), ...] or ndarray or torch.Tensor, first column is x, second column is y
        test_data: [(epoch, data), ...] or ndarray or torch.Tensor, first column is x, second column is y
        name: "R2", "RMSE", "SIGMA"
        save_name: str, default=None, save name of figure
        dpi: int, default=300, dpi of figure
        yaxis: [min, max], default=None, yaxis range
        max_best: bool, default=True, whether to choose max value as best value
        save: bool, default=True, whether to save the figure
        close: bool, default=True, whether to close the figure
    """
    if isinstance(train_data, list):
        train_data = np.array(train_data)
    if isinstance(train_data, torch.Tensor):
        train_data = train_data.detach().cpu().numpy()
    
    if isinstance(test_data, list):
        test_data = np.array(test_data)
    if isinstance(test_data, torch.Tensor):
        test_data = test_data.detach().cpu().numpy()

    if save_name is None:
        save_name = './' + name + '.png'
    if name in ["R2", "SSIM", "SNR", "PSNR"]:
        func = np.argmax
    elif name in ["RMSE", "SIGMA"]:
        func = np.argmin
    else:
        func = np.argmax if max_best else np.argmin
    best_index = func(test_data[:,1])
    title_str = "%s = %.4f, Best at Epoch = %d"%(name, float(test_data[best_index,1]), int(test_data[best_index,0]))

    with Plot(dpi=dpi, save_name=save_name, 
                        xlabel="epoch", ylabel=name, title=title_str, 
                        yaxis=yaxis, save=save, close=close, **kwargs):
        plt.plot(train_data[:,0], train_data[:,1], color='blue', label='train')
        plt.plot(test_data[:,0], test_data[:,1], color='red', label='test')


def plot_loss_list(loss, save_name=None, dpi=300, yaxis=None, save=True, close=True, **kwargs):
    """
    Plot loss curve

    Parameters:
        loss: [(iter, loss), ...] or ndarray or torch.Tensor, first column is x, second column is y
        save_name: str, default=None, save name of figure
        dpi: int, default=300, dpi of figure
        yaxis: [min, max], default=None, yaxis range
        save: bool, default=True, whether to save the figure
        close: bool, default=True, whether to close the figure
    """
    if isinstance(loss, list):
        loss = np.array(loss)
    if isinstance(loss, torch.Tensor):
        loss = loss.detach().cpu().numpy()

    if save_name is None:
        save_name = './Loss.png'

    with Plot(dpi=dpi, save_name=save_name,
                        xlabel="iter", ylabel='loss', 
                        yaxis=yaxis, save=save, close=close, **kwargs):
        plt.plot(loss[:,0], loss[:,1], color='blue', label='train', lw=0.3)


def plot_phase_img(data, cmap='jet', save_name=None, dpi=300, caxis=None, show_colorbar=True, transparent=True, show_axis=False, zero_is_nan=True,
                   save=True, close=True, **kwargs):
    """
    Plot phase image

    Parameters:
        data: (h,w,p,p) or [(c,h,w),(c,p,p)] or (h*p,w*p), all torch.Tensor
        cmap: 'jet', 'gray', 'viridis', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'bone', 'copper', ...
        save_name: str, default=None, save name of figure
        dpi: int, default=300, dpi of figure
        caxis: [min, max], default=None, colorbar range
        show_colorbar: bool, default=True, whether to show colorbar
        transparent: bool, default=True, whether to set background to transparent
        show_axis: bool, default=False, whether to show axis
        zero_is_nan: bool, default=True, whether to set zero to nan
        save: bool, default=True, whether to save the figure
        close: bool, default=True, whether to close the figure
        **kwargs: other parameters for "with Plot(**kwargs)..."
    """
    if isinstance(data, list):
        zernike = data[0]
        c, h, w = zernike.shape
        Z2P = data[1]
        c, p, _ = Z2P.shape
        phase = F.linear(zernike.permute(1,2,0), Z2P.reshape(c,p*p).permute(1,0), bias=None).reshape(h,w,p,p)
        img = phase.permute(0,2,1,3).reshape(h*p, w*p)
    elif isinstance(data, torch.Tensor):
        if len(data.shape) == 4:
            phase = data
            img = phase.permute(0,2,1,3).reshape(h*p, w*p)
        elif len(data.shape) == 2:
            img = data
        else:
            raise ValueError("data should be (h,w,p,p) or [(c,h,w),(c,p,p)] or (h*p,w*p)")
    else:
        raise TypeError("data should be list or torch.Tensor")
    if zero_is_nan:
        img[img == 0] = np.nan

    if save_name is None:
        save_name = './phase.png'
    
    with Plot(dpi=dpi, save_name=save_name, show_grid=False, show_box=False, show_axis=show_axis,
                        transparent=transparent, show_legend=False, show_colorbar=show_colorbar, caxis=caxis,
                        save=save, close=close, **kwargs):
        plt.imshow(img / 2 / np.pi * 525, cmap=cmap)



class Plot:
    def __init__(self, dpi=300, save_name=None, show_grid=True, show_box=True, show_axis=True, transparent=False,
                 xlabel=None, ylabel=None, title=None, show_legend=True, legend_loc='best', 
                 show_colorbar=False, yaxis=None, xaxis=None, caxis=None, ggplot=True, fig_size=None, show=False, save=True, close=True):
        """
        Plot class based on matplotlib.pyplot

        Parameters:
            dpi: int, default=300, dpi of figure
            save_name: str, default=None, save name of figure
            show_grid: bool, default=True, whether to show grid
            show_box: bool, default=True, whether to show box
            show_axis: bool, default=True, whether to show axis
            transparent: bool, default=False, whether to set background to transparent
            xlabel: str, default=None, xlabel of figure
            ylabel: str, default=None, ylabel of figure
            title: str, default=None, title of figure
            show_legend: bool, default=True, whether to show legend
            legend_loc: str, default='best', legend location
            show_colorbar: bool, default=False, whether to show colorbar
            yaxis: [min, max], default=None, yaxis range
            xaxis: [min, max], default=None, xaxis range
            caxis: [min, max], default=None, colorbar range
            ggplot: bool, default=True, whether to use ggplot style
            fig_size: (w,h), default=None, figure size
            show: bool, default=False, whether to show the figure
            save: bool, default=True, whether to save the figure
            close: bool, default=True, whether to close the figure
        """
        self.dpi = dpi
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.save_name = save_name if save_name is not None else './plot.png'
        self.show_grid = show_grid
        self.show_box = show_box
        self.show_axis = show_axis
        self.show_legend = show_legend
        self.legend_loc = legend_loc
        self.transparent = transparent
        self.show_colorbar = show_colorbar
        self.yaxis = yaxis
        self.xaxis = xaxis
        self.caxis = caxis
        self.ggplot = ggplot
        self.show = show
        self.save = save
        self.fig_size = fig_size
        self.close = close

    def __enter__(self):
        plt.rcdefaults()
        if self.fig_size is not None:
            plt.figure(figsize=self.fig_size, dpi=self.dpi)
        else:
            plt.figure(dpi=self.dpi)
        if self.ggplot:
            plt.style.use('ggplot')

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        plt.box(self.show_box)
        plt.grid(self.show_grid)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel)
        if self.title is not None:
            plt.title(self.title)
        if self.show_legend:
            plt.legend(loc=self.legend_loc)
        if not self.show_axis:
            plt.axis('off')
        if self.show_colorbar:
            plt.colorbar()
        if self.yaxis is not None:
            plt.ylim(self.yaxis)
        if self.xaxis is not None:
            plt.xlim(self.xaxis)
        if self.caxis is not None:
            plt.clim(self.caxis)
        if self.save:
            plt.savefig(self.save_name, transparent=self.transparent)
        if self.show:
            plt.show()
            cv2.waitKey(0)
        if self.close:
            plt.close()

        if exc_type is not None:
            print(f"An exception of type {exc_type} occurred with value {exc_value}")
            if traceback is not None:
                print("Exception traceback:")
                import traceback as tb
                tb.print_tb(traceback)
            return True
        return False
    
