from helper import *     
import scipy.io as io


set_list = [18, 20, 21, 22, 23, 27]
for set_id in set_list:
    Helper.makedirs("./data/phase_data/230408/set%d/" % set_id)
    data = torch.load("./data/open_source_data/pred_data/zernike_data%d.pt" % set_id)
    for meta_id in range(1000):
        tmp = data[meta_id].clone()
        io.savemat("./data/phase_data/230408/set%d/moon%d_mlp_zernike.mat" % (set_id, meta_id), {"zernike": tmp.numpy()})



# set_list = [25, 26]
# for set_id in set_list:
#     Helper.makedirs("./data/phase_data/230408/set%d/" % set_id)
#     for meta_id in range(1000):
#         tmp = io.loadmat("./data/open_source_data/prep_data/set%d/shiftmap%d.mat" % (set_id, meta_id))["shiftmap"]
#         io.savemat("./data/phase_data/230408/set%d/moon%d_slope.mat" % (set_id, meta_id), {"slope": tmp})

#         tmp = io.loadmat("./data/open_source_data/prep_data/set%d/zernike_full%d.mat" % (set_id, meta_id))["zernike"]
#         io.savemat("./data/phase_data/230408/set%d/moon%d_zernike.mat" % (set_id, meta_id), {"zernike": tmp})

# a = torch.load("./data/open_source_data/pred_data/zernike_data26.pt")[123]
# print(a.shape)
# # a = torch.from_numpy(io.loadmat("./data/phase_data/230408/set26/moon12_zernike.mat")["zernike"])
# # print(a.shape)
# b = torch.from_numpy(io.loadmat("./data/phase_data/230408/set26/moon123_mlp_zernike.mat")["zernike"])
# print(b.shape)
# R2 = 1.0 - torch.sum((a - b) ** 2) / torch.sum((a - torch.mean(a)) ** 2)
# print(R2)


