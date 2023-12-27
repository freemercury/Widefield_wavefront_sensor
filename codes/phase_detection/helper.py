import os
import os.path as op
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tifffile as tiff
import random


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