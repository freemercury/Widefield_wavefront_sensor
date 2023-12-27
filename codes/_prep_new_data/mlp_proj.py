from helper import *
from models import *
from matplotlib import pyplot as plt
import os
import multiprocessing as mp
import subprocess as sp

# helper & device
devices = Helper.try_all_gpus()
device = devices[0]

# path
BASE_PATH = os.getcwd() + "/"
PREP_PATH = BASE_PATH + "data/prep_data_230407_new_1/"
PREP_PARAM_PATH = BASE_PATH + "data/prep_data_param/"
MASK_PATH = PREP_PARAM_PATH + "S2Zmatrix.mat"


# phase config
HIDDEN_SIZE = [300,500] # hidden layer size of mlp
NUM_ZERNIKE = 35     # number of zernike modes
# mlp training config
P_DEBUG = True  # 存mlp训练loss图, R2图
TRAIN = True    # train or test
META_LEN = 1000
TRAIN_SET = 3
TEST_SET = 2
LR = 1e-4
N_EPOCH = 10
BATCH_SIZE = 128


def train_mlp(device, prep_path, prep_param_path, mask, valid_views, meta_len=1000, train_set=3, test_set=2, lr=1e-3, n_epoch=10, batch_size=128, 
              hidden_size=[300,500], num_zernike=35, to_plot=True):
    
    Helper.makedirs(prep_param_path)

    # load X_train
    X_train = []
    for meta_id in range(meta_len):
        train_shiftmap = torch.tensor(io.loadmat(prep_path + "%d/shiftmap%d.mat"%(train_set, meta_id))["shiftmap"]).type(torch.float32)
        train_shiftmap = shiftmap_convertor_fullsize2validviews(train_shiftmap, mask, valid_views)
        X_train.append(train_shiftmap)
    X_train = torch.stack(X_train, dim=0).permute(0,2,3,4,1).reshape(-1,2*len(valid_views)).to(device)

    # load X_test
    if test_set is not None:
        X_test = []
        for meta_id in range(meta_len):
            test_shiftmap = torch.tensor(io.loadmat(prep_path + "%d/shiftmap%d.mat"%(test_set, meta_id))["shiftmap"]).type(torch.float32)
            test_shiftmap = shiftmap_convertor_fullsize2validviews(test_shiftmap, mask, valid_views)
            X_test.append(test_shiftmap)
        X_test = torch.stack(X_test, dim=0).permute(0,2,3,4,1).reshape(-1,2*len(valid_views)).to(device)

    # load y_train
    y_train = []
    for meta_id in range(meta_len):
        train_zernike = torch.tensor(io.loadmat(prep_path + "%d/zernike_full%d.mat"%(train_set, meta_id))["zernike"]).type(torch.float32)
        y_train.append(train_zernike)
    y_train = torch.stack(y_train, dim=0).permute(0,2,3,1).reshape(-1,num_zernike).to(device)

    # load y_test
    if test_set is not None:
        y_test = []
        for meta_id in range(meta_len):
            test_zernike = torch.tensor(io.loadmat(prep_path + "%d/zernike_full%d.mat"%(test_set, meta_id))["zernike"]).type(torch.float32)
            y_test.append(test_zernike)
        y_test = torch.stack(y_test, dim=0).permute(0,2,3,1).reshape(-1,num_zernike).to(device)

    # normalize
    X_mean = torch.mean(X_train, dim=0, keepdim=True)
    X_std = torch.std(X_train, dim=0, keepdim=True)
    y_mean = torch.mean(y_train, dim=0, keepdim=True)
    y_std = torch.std(y_train, dim=0, keepdim=True)
    X_train_norm = (X_train - X_mean) / X_std
    y_train_norm = (y_train - y_mean) / y_std
    if test_set is not None:
        X_test_norm = (X_test - X_mean) / X_std
        y_test_norm = (y_test - y_mean) / y_std

    # dataset
    train_dataset = TensorDataset(X_train_norm, y_train_norm)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    if test_set is not None:
        test_dataset = TensorDataset(X_test_norm, y_test_norm)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # model
    model = MLP(2*len(valid_views), hidden_size, num_zernike).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.MSELoss()

    # training
    epoch_list = []
    loss_list = []
    R2_train = []
    R2_test = []
    for epoch in tqdm(range(n_epoch)):
        epoch_list.append(epoch)
        # train
        model.train()
        for _, (batch_x, batch_y) in enumerate(train_loader):
            pred = model(batch_x)
            loss = loss_func(pred, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        # test on trainset
        model.eval()
        R2_cum = 0
        Num = 0
        for _, (batch_x, batch_y) in enumerate(train_loader):
            pred = model(batch_x)
            pred = pred * y_std + y_mean
            batch_y = batch_y * y_std + y_mean
            R2_cum += torch.sum(1 - torch.sum((pred - batch_y)**2, dim=1) / torch.sum((batch_y - y_mean)**2, dim=1)).item()
            Num += batch_x.shape[0]
        R2_train.append(R2_cum / Num)
        # test on testset
        if test_set is not None:
            R2_cum = 0
            Num = 0
            for _, (batch_x, batch_y) in enumerate(test_loader):
                pred = model(batch_x)
                pred = pred * y_std + y_mean
                batch_y = batch_y * y_std + y_mean
                R2_cum += torch.sum(1 - torch.sum((pred - batch_y)**2, dim=1) / torch.sum((batch_y - y_mean)**2, dim=1)).item()
                Num += batch_x.shape[0]
            R2_test.append(R2_cum / Num)
        else:
            R2_test.append(R2_train[-1])
        tqdm.write("epoch %d, train R2: %.4f, test R2: %.4f\n"%(epoch, R2_train[-1], R2_test[-1]))
        if to_plot:
            # plot loss
            plt.plot(loss_list)
            plt.xlabel('iter')
            plt.ylabel('loss')
            plt.savefig(prep_param_path + "loss.png")
            plt.close()
            # plot R2
            plt.plot(epoch_list, R2_train, color="blue", label="train")
            plt.plot(epoch_list, R2_test, color="red", label="test")
            plt.xlabel('epoch')
            plt.ylabel('R2')
            plt.legend(loc='best')
            plt.savefig(prep_param_path + "R2.png")
            plt.close()

    # save model
    torch.save({"model": model.state_dict(), "R2_train": R2_train, "R2_test": R2_test, "loss": loss_list, "epoch": epoch_list,
                "X_mean": X_mean.cpu(), "X_std": X_std.cpu(), "y_mean": y_mean.cpu(), "y_std": y_std.cpu()
                }, prep_param_path + "mlp_model.pkl")


def test_mlp(device, prep_path, prep_param_path, mask, valid_views, meta_len=1000, test_set=2, batch_size=128, 
             hidden_size=[300,500], num_zernike=35):
    
    # load X_test
    if test_set is not None:
        X_test = []
        for meta_id in range(meta_len):
            test_shiftmap = torch.tensor(io.loadmat(prep_path + "%d/shiftmap%d.mat"%(test_set, meta_id))["shiftmap"]).type(torch.float32)
            test_shiftmap = shiftmap_convertor_fullsize2validviews(test_shiftmap, mask, valid_views)
            X_test.append(test_shiftmap)
        X_test = torch.stack(X_test, dim=0).permute(0,2,3,4,1).reshape(-1,2*len(valid_views)).to(device)

    # load y_test
    if test_set is not None:
        y_test = []
        for meta_id in range(meta_len):
            test_zernike = torch.tensor(io.loadmat(prep_path + "%d/zernike_full%d.mat"%(test_set, meta_id))["zernike_full"]).type(torch.float32)
            y_test.append(test_zernike)
        y_test = torch.stack(y_test, dim=0).permute(0,2,3,1).reshape(-1,num_zernike).to(device)

    # load model
    model = MLP(2*len(valid_views), hidden_size, num_zernike).to(device)
    data_dict = torch.load(prep_param_path + "mlp_model.pkl")
    X_mean, X_std = data_dict["X_mean"].to(device), data_dict["X_std"].to(device)
    y_mean, y_std = data_dict["y_mean"].to(device), data_dict["y_std"].to(device)
    model.load_state_dict(data_dict["model"])

    # normalize
    X_test_norm = (X_test - X_mean) / X_std
    y_test_norm = (y_test - y_mean) / y_std

    # dataset
    test_dataset = TensorDataset(X_test_norm, y_test_norm)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # test
    model.eval()
    R2_cum = 0
    Num = 0
    for _, (batch_x, batch_y) in enumerate(tqdm(test_loader)):
        pred = model(batch_x)
        pred = pred * y_std + y_mean
        batch_y = batch_y * y_std + y_mean
        R2_cum += torch.sum(1 - torch.sum((pred - batch_y)**2, dim=1) / torch.sum((batch_y - y_mean)**2, dim=1)).item()
        Num += batch_x.shape[0]
    R2_test = R2_cum / Num
    print("test R2: %.4f\n"%(R2_test))



if __name__ == "__main__":
    print("***********************\nPID: %d\n"%(os.getpid()))
    Helper.makedirs(PREP_PATH)
    Helper.makedirs(PREP_PARAM_PATH)

    mask, P, valid_views = get_mask_P(MASK_PATH)
    if TRAIN:
        train_mlp(device, PREP_PATH, PREP_PARAM_PATH, mask, valid_views, META_LEN, TRAIN_SET, TEST_SET,
                    LR, N_EPOCH, BATCH_SIZE, HIDDEN_SIZE, NUM_ZERNIKE, P_DEBUG)
    else:
        test_mlp(device, PREP_PATH, PREP_PARAM_PATH, mask, valid_views, META_LEN, TEST_SET,
                BATCH_SIZE, HIDDEN_SIZE, NUM_ZERNIKE)

    # for i in range(2,4):
    #     for j in range(1000):
    #         zernike_data = io.loadmat("D:/hyh/project/LFM/data/prep_data_230407_new_1/%d/zernike%d.mat"%(i,j))["zernike"]
    #         io.savemat("D:/hyh/project/LFM/data/prep_data_230407_new_1/%d/zernike_full%d.mat"%(i,j), {"zernike": zernike_data})


    print("Done!")