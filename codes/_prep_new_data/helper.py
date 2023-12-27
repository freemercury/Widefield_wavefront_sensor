# helper.py
import os
import os.path as op
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL.Image as Image
import numpy as np
from torchvision.io import read_image, write_png, ImageReadMode
from torchvision import transforms
import cv2   
import tifffile as tiff
import random
import yaml
from tqdm import tqdm


class Helper():
    """

    random_seed

    device: try_gpu, try_all_gpus, get_cpu, cuda_available, cuda_count, get_cpu

    path: is_path_exist, makedirs, get_timestamp

    json: read_json, save_json, read_csv, save_csv, read_yaml, save_yaml, read_txt, read_txt_lines, save_txt

    image: pil/torch/cv/tif_save/read_img

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
    def read_csv(cls, path):
        dataframe = pd.read_csv(path)
        return {keyname:list(dataframe.iloc[:,i]) for i, keyname in enumerate(dataframe.columns) if i > 0}

    @classmethod
    def save_csv(cls, datadict, path):
        dataframe = pd.DataFrame(datadict)
        dataframe.to_csv(path)
        return True

    @classmethod
    def read_yaml(cls, yamlpath):
        with open(yamlpath, 'r') as f:
            return yaml.safe_load(f)
        
    @classmethod
    def save_yaml(cls, datadict, yamlpath):
        with open(yamlpath, 'w') as f:
            yaml.dump(datadict, f)
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
    
    @classmethod
    def cv_read_img(cls, imgpath, torch_type=True, mode="unchanged"):
        """
        support all image types for 8-bits or 16-bits

        torch_type:

            True: return torch.tensor (float32)

            False: return np.array (raw dtype, e.g. uint8, uint16, float32)

        mode: 

            "unchanged": return raw image

            "grayscale": return gray image

            "color": return rgb image

        img shape: (h, w, ~c)
        """
        read_mode_dict = {"unchanged": cv2.IMREAD_UNCHANGED, "grayscale": cv2.IMREAD_GRAYSCALE, "rgb": cv2.IMREAD_COLOR}
        read_mode = read_mode_dict[mode]
        if torch_type:
            img = torch.tensor(cv2.imread(imgpath, flags=read_mode).astype(np.float32))
        else:
            img = cv2.imread(imgpath, flags=read_mode)
        return img

    @classmethod
    def cv_save_img(cls, img, imgpath, save_type="uint"):
        """
        support all image types for 8-bits or 16-bits

        img: could be np.array or torch.tensor on cpu or gpu

        save_type: "uint" for adaptively choose uint8 or uint16; None for not change; "uint8", "uint16", "float32"

        img shape: (h, w, ~c)
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
        cv2.imwrite(imgpath, img)
        return True
    
    @classmethod
    def pil_read_img(cls, imgpath, torch_type=True):
        """
        only support 8bit image

        torch_type:

            True: return torch.tensor (float32, range: [0, 1]) (c, h, w)

            False: return PIL.Image object
        """
        img = Image.open(imgpath)
        if torch_type:
            img = transforms.ToTensor()(img)
        return img

    @classmethod
    def pil_save_img(cls, img, imgpath):
        """
        img should be PIL.Image object
        """
        img.save(imgpath)
        return True
    
    @classmethod
    def torch_read_img(cls, imgpath, torch_type=True, mode="unchanged"):
        """
        only support png with 8-bits

        torch_type:

            True: return torch.tensor (float32, range: [0,1]) (c, h, w)

            False: return torch.tensor (raw dtype, e.g. uint8, uint16, float32) (c, h, w)

        mode: 

            "unchanged": return raw image

            "gray": return gray image

            "gray_alpha": return gray image with alpha channel

            "rgb": return rgb image

            "rgb_alpha": return rgb image with alpha channel
        """
        read_mode_dict = {"unchanged": ImageReadMode.UNCHANGED, "gray": ImageReadMode.GRAY, "gray_alpha": ImageReadMode.GRAY_ALPHA, "rgb": ImageReadMode.RGB, "rgb_alpha": ImageReadMode.RGB_ALPHA}
        read_mode = read_mode_dict[mode]
        img = read_image(imgpath, mode=read_mode)
        if torch_type:
            img = img.type(torch.float32) / 255
        return img

    @classmethod
    def torch_save_img(cls, img, imgpath):
        """
        only save uint8 png image

        img shape should be (c, h, w)
        """
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        elif img.device.type == "cuda":
            img = img.cpu()
        if torch.dtype is not torch.uint8:
            if img.max() <= 1.0:
                img = (img * 255).type(torch.uint8)
            else:
                img = img.type(torch.uint8)
        if img.shape[0] == 4:
            img = img[:3,:,:]
        write_png(img, imgpath)
    #endregion
    

