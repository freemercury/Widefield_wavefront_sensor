# Wide-field Wavefront Sensor (WWS)

[[paper](https://www.nature.com/articles/s41566-024-01466-3)]

![figure1](https://github.com/freemercury/Widefield_wavefront_sensor/blob/main/docs/figure1.png)

This is the official repository of "Direct Observation of Atmospheric Turbulence with a Video-rate Wide-field Wavefront Sensor". This repository contains codes, video demos and data of our work.

**Authors:** Yuduo Guo, Yuhan Hao, Sen Wan, Hao Zhang, Laiyu Zhu, Yi Zhang,  Jiamin Wu, Qionghai Dai & Lu Fang

**Acknowledgements**: We acknowledge the support of the staff of the Xinglong 80cm Tsinghua - National Astronomical Observatory, Chinese Academic of Sciences. This project is supported in part by the National Natural Science Foundation of China (NSFC) (numbers 62125106, 61860206003 and 62088102, to L.F., number 62222508, to J.W.) and in part by Ministry of Science and Technology of China (contract number 2021ZD0109901, to L.F.).

​    

## Table of Content

1. [Video Demos](#video-demos)
2. [Data](#data)
3. [Codes](#codes)

​    

## Video Demos

### Wavefront Observation

https://github.com/freemercury/Widefield_wavefront_sensor/assets/70796826/5cb20112-dac5-49d5-a9a9-3eca3f946ae7.mp4

### Wavefront Prediction

https://github.com/freemercury/Widefield_wavefront_sensor/assets/70796826/d3069e83-4c06-4039-9aa7-a9e9fdd95133.mp4

​    

## Data

Download demo data and pre-trained model weights [here](https://zenodo.org/records/10476938/files/WWS_data_log.rar?download=1). After unzipping the files, simply copy the `data` and `log` folders directly into the project's root directory.

Download more raw data from the following links:

* https://doi.org/10.5281/zenodo.11063896
* https://doi.org/10.5281/zenodo.11071397

Due to the large size of raw data, we currently only open source a subset of the raw data. Should you require access to any datasets that have not yet been uploaded, please contact fanglu@tsinghua.edu.cn (L.F.).



## Codes

### Installation

1. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/locally/), e.g.

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. Install other requirements by the following command:

```
pip install -r requirements.txt
```

### Phase Detection

This repository provides the code for phase detection, which is to detect the phase of the wavefront from sub-aperture images realigned from light-field images.

#### Estimate slope map based on realigned sub-aperture images

Acquire slope map from realigned sub-aperture images.

```
python src/slope.py --realign_data_path ${REALIGN_DATA_PATH} --phase_data_path ${PHASE_DATA_PATH}
```

`REALIGN_DATA_PATH` is the path of realigned sub-aperture images, the default value of which is `data/realign_data/230406/set2`. `PHASE_DATA_PATH` is the path to save the slope map in `*_slope.mat`, the default value of which is `data/phase_data/230406/set2`. You can simply change the value of `REALIGN_DATA_PATH` and `PHASE_DATA_PATH` to process your own data.

If the debug mode is on, loss curve and input/target/warped images of specific views will be saved in `PHASE_DATA_PATH`. To turn on the debug mode, add `--debug` to the command, e.g.

```
python src/slope.py --realign_data_path ${REALIGN_DATA_PATH} --phase_data_path ${PHASE_DATA_PATH} --debug
```

You can use `--help` to check for more options.

```
python src/slope.py --help
```

#### Integrate slope map to obtain wavefront

To obtain the wavefront from the slope map, you should run the *first chunk* of `src/phase.m` in MATLAB. You can refer to `src/phase.m` for more details. The output wavefront is saved as Coefficients of Zernike polynomials in the same path as `*_slope.mat`, named `*_zernike.mat`.

#### MLP projector for faster integration

We also provide an MLP projector to obtain the wavefront from slope maps. The weights of the MLP projector are in `log/mlp/`. 

To infer the wavefront from slope maps, run the following command:

```
python src/phase.py --job infer --data_path ${PHASE_DATA_PATH}
```

`PHASE_DATA_PATH` is the path of slope maps, the default value of which is `data/phase_data/230406/set2`. The output wavefront is saved as Coefficients of Zernike polynomials in the same path as `*_slope.mat`, named `*_mlp_zernike.mat`.

What's more, you can also try to train the MLP projector on the data provided. To train the MLP projector, run the following command:

```
python src/phase.py --job train --data_path data/phase_data/230406/set3/
```

You can then evaluate the trained MLP projector on the test set provided. To evaluate the MLP projector, run the following command:

```
python src/phase.py --job test --data_path data/phase_data/230406/set2/
```

Use `--help` to check for more options.

```
python src/phase.py --help
```

#### Remove system aberration

System aberration should be removed for further analysis. You can remove the system aberration of the provided data (which is to be used in wavefront prediction) by running the following command:

```
python src/phase.py --job remove_sys --data_path data/phase_data/230408/
```

For visualization of the system-aberration-removed wavefront, try the following command:

```
python src/phase.py --job draw --data_path data/phase_data/230408/set18/
```

The images of the system-aberration-removed wavefront will be saved in `data/phase_data/230408/set18/draw/`.

### Phase Prediction

#### Evaluate the pre-trained model

To evaluate the pre-trained model, run the following command:

```
python src/pred.py --job test --config_path src/config_test.json
```

The pre-trained weights will be loaded from `log/rclstm_17t7/ckpt/`. The R2/RMSE/SIGMA heatmap will be saved in `log/rclstm_17t7/test*_ts*/heatmap/`. The predicted wavefront will be saved in `log/rclstm_17t7/test*_ts*/zernike`/. The image of ground-truth and predicted wavefront will be saved in `log/rclstm_17t7/test*_ts*/gt_phase_img/` and `log/rclstm_17t7/test*_ts*/pred_phase_img/` respectively. 

For more options, use `--help` to check for more options.

```
python src/pred.py --help
```

#### Training on the provided data

Try to train another model on the provided data by running the following command:

```
python src/pred.py --job train --config_path src/config_train.json
```

The training log will be saved in `log/rclstm_17t7_new/`. 

#### Parameters of training

The parameters of the model can be referred to and adjusted in `src/config_train.json`. One example is as follows:

```json
{
    "info": {
        "name": "17t7_new"
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
        "data_path": "./data/phase_data/230408/", 
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
```
The parameters are explained as follows:

- `name`: the name of the model, which will be used to name the log folder.
- `n_dim`: list of list of int. Each list contains dimensions of one block, of which the first element is input dimension and the rest are hidden dimensions for each layer.
- `kernel_size`: list of list of int. Each list contains kernel sizes of one block, of which one element is kernel size for each layer.
- `lr`: the learning rate of the optimizer.
- `weight_decay`: the weight decay of the optimizer.
- `eta_min`: the minimum learning rate of the scheduler.
- `loss_type`: the loss function used in training, could be 'l1' or 'l2'.
- `reduction`: the reduction method of the loss function, could be 'mean' or 'sum'.
- `data_path`: the path of all the data.
- `zernike_phase_path`: the path of the zernike2phase coefficients.
- `phase_size`: the size of the phase map, should be in [15, 35, 55, 75, 95, 115]. 
- `n_channel`: [min_channel, max_channel], choose zernike modes, 0 for piston.
- `split`: the ratio of training set to all data.
- `t_series`: length of input sequence.
- `t_offset`: offset between input and target.
- `t_down`: downsample interval, 0 for no downsample, 1 for 1/2 downsample (interval is 1 frame).
- `all_patch`: [min_h, max_h, min_w, max_w], choose patch for all data, used as indices [min_h:max_h, min_w:max_w]
- `input_patch`: [min_h, max_h, min_w, max_w], choose patch for input data, used as indices [min_h:max_h, min_w:max_w]
- `target_patch`: [min_h, max_h, min_w, max_w], choose patch for target data, used as indices [min_h:max_h, min_w:max_w]
- `test_size`: [h, w], test_size for metric calculation
- `device_id`: the id of the GPU to use.
- `seed`: the random seed.
- `batch_size`: the batch size.
- `epoch`: the number of epochs to train.
- `virtual_epoch`: number of virtual maximum epochs, used to calculate T_max.
- `warmup_epoch`: number of epochs for warmup.
- `save_epoch`: save model every `save_epoch` epochs.
- `eval_epoch`: evaluate model every `eval_epoch` epochs.
- `plot_epoch`: plot metric curve every `plot_epoch` epochs.