import os
import os.path as op
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
import random
from matplotlib import pyplot as plt
import cv2   


class Helper():
    """

    random_seed

    device: try_gpu, try_all_gpus, get_cpu, cuda_available, cuda_count, get_cpu

    path: is_path_exist, makedirs, get_timestamp

    json: read_json, save_json, read_txt, read_txt_lines, save_txt

    image: tif_save/read_img

    """
    def __init__(self):
        pass
    
    @classmethod
    def random_seed(cls, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    #region DEVICE
    @classmethod
    def try_gpu(cls, device_id=0):
        if torch.cuda.is_available() and torch.cuda.device_count() > device_id:
            torch.cuda.set_device("cuda:%d" % device_id)
            return torch.device("cuda:%d" % device_id)
        else:
            return torch.device("cpu")
    
    @classmethod
    def try_all_gpus(cls):
        if torch.cuda.is_available():
            return [torch.device("cuda:%d" % i) for i in range(torch.cuda.device_count())]
        else:
            return [torch.device("cpu")]

    @classmethod
    def cuda_available(cls):
        return torch.cuda.is_available()
    
    @classmethod
    def cuda_count(cls):
        return torch.cuda.device_count()

    @classmethod
    def get_cpu(cls):
        return torch.device("cpu")
    #endregion

    #region PATH
    @classmethod
    def is_path_exist(cls, path):
        return op.exists(path)

    @classmethod
    def makedirs(cls, path):
        if not op.exists(path):
            os.makedirs(path)
        return True
    
    @classmethod
    def get_timestamp(cls):
        import datetime
        return datetime.datetime.now().strftime("%m%d%H%M%S")
    #endregion

    #region FILE_IO
    @classmethod
    def read_json(cls, jsonpath):
        return json.loads(open(jsonpath, 'r', encoding='utf-8').read())

    @classmethod
    def save_json(cls, datadict, jsonpath):
        filejson = json.dumps(datadict, ensure_ascii=False, indent=1)
        fp = open(jsonpath, "w", encoding='utf-8', newline='\n')
        fp.write(filejson)
        fp.close()
        return True

    @classmethod
    def read_txt(cls, txtpath):
        with open(txtpath, 'r') as f:
            return f.read()
        
    @classmethod
    def read_txt_lines(cls, txtpath):
        """
        return list of lines, including '\n'
        """
        with open(txtpath, 'r') as f:
            return f.readlines()

    @classmethod
    def save_txt(cls, txt, txtpath):
        with open(txtpath, 'w') as f:
            f.write(txt)
        return True
    #endregion

    #region IMAGE_IO
    @classmethod
    def tif_read_img(cls, imgpath, torch_type=True):
        """
        only support tiff file with 8-bits or 16-bits

        torch_type:

            True: return torch.tensor (float32)

            False: return np.array (raw dtype, e.g. uint8, uint16, float32)

        img shape: (~n, h, w, ~c)
        """
        if torch_type:
            img = torch.tensor(tiff.imread(imgpath).astype(np.float32))
        else:
            img = tiff.imread(imgpath)
        return img

    @classmethod
    def tif_save_img(cls, img, imgpath, save_type="uint"):
        """
        only support tiff file with 8-bits or 16-bits

        img: could be np.array or torch.tensor on cpu or gpu

        save_type: "uint" for adaptively choose uint8 or uint16; None for not change; "uint8", "uint16", "float32"

        img shape: (~n, h, w, ~c)
        """
        if isinstance(img, torch.Tensor):
            if img.device.type == "cuda":
                img = img.cpu()
            img = img.numpy()
        if save_type == "uint8":
            img = img.astype(np.uint8)
        elif save_type == "uint16":
            img = img.astype(np.uint16)
        elif save_type == "float32":
            img = img.astype(np.float32)
        elif save_type == "uint":
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            elif img.max() <= 255:
                img = img.astype(np.uint8)
            else:
                img = img.astype(np.uint16)
        tiff.imsave(imgpath, img)
        return True
    #endregion

    # region PLOT
    @classmethod
    def plot_phase_img(cls, data, cmap='jet', save_name=None, dpi=300, caxis=None, show_colorbar=True, transparent=True, show_axis=False, zero_is_nan=True,
                       save=True, close=True, **kwargs):
        """
        data: (h,w,p,p) or [(c,h,w),(c,p,p)] or (h*p,w*p), all torch.Tensor

        caxis: [min, max]

        cmap: 'jet', 'gray', 'viridis', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'bone', 'copper', 'pink', 'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'
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
            plt.imshow(img, cmap=cmap)
    #endregion





class Plot:
    def __init__(self, dpi=300, save_name=None, show_grid=True, show_box=True, show_axis=True, transparent=False,
                 xlabel=None, ylabel=None, title=None, show_legend=True, legend_loc='best', 
                 show_colorbar=False, yaxis=None, xaxis=None, caxis=None, ggplot=True, fig_size=None, show=False, save=True, close=True):
        """
        dpi: int, default=300
        
        save_name: str, default=None
        
        show_grid: bool, default=True
        
        show_box: bool, default=True
        
        show_axis: bool, default=True
        
        transparent: bool, default=False
        
        xlabel: str, default=None
        
        ylabel: str, default=None
        
        title: str, default=None

        show_legend: bool, default=True

        legend_loc: str, default='best'

        show_colorbar: bool, default=False

        yaxis: [min, max], default=None

        xaxis: [min, max], default=None

        caxis: [min, max], default=None

        ggplot: bool, default=True
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
    

