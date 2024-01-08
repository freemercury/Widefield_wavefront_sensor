from helper import *
from phase_detection_model import PhaseProjection
import argparse
from scipy import io
import glob
from tqdm import tqdm
from phase_prediction_model import *

CONFIG_PATH = "./src/config.json"
JOB_TYPE = "train"  # "train" or "test"
EVAL_MODEL = True
SAVE_MODEL = True
PLOT_METRIC = True
REMOVE_CKPT = True
BEST_METRIC = "RMSE"
SAVE_TEST_ZERNIKE = True
PLOT_TEST_ZERNIKE = True
PLOT_PHASE_SIZE = 55



# args
def parse_args():
    parser = argparse.ArgumentParser(description='Phase Prediction Arguments')
    parser.add_argument('--config_path', default=CONFIG_PATH, type=str, help='config path')
    parser.add_argument('--job_type', default=JOB_TYPE, type=str, help='job type, "train" or "test"')
    parser.add_argument('--eval_model', default=EVAL_MODEL, type=bool, help='whether to eval model')
    parser.add_argument('--save_model', default=SAVE_MODEL, type=bool, help='whether to save model')
    parser.add_argument('--plot_metric', default=PLOT_METRIC, type=bool, help='whether to plot metric')
    parser.add_argument('--remove_ckpt', default=REMOVE_CKPT, type=bool, help='whether to remove ckpt')
    parser.add_argument('--best_metric', default=BEST_METRIC, type=str, help='load model based on which metric, could be "RMSE" or "R2" or "SIGMA"')
    parser.add_argument('--save_test_zernike', default=SAVE_TEST_ZERNIKE, type=bool, help='whether to save test zernike')
    parser.add_argument('--plot_test_zernike', default=PLOT_TEST_ZERNIKE, type=bool, help='whether to plot test zernike')
    parser.add_argument('--plot_phase_size', default=PLOT_PHASE_SIZE, type=int, help='plot phase size')
    args = parser.parse_args()
    return vars(args)


def pred_phase():
    args = parse_args()
    config_dict = Helper.read_json(args["config_path"])
    Model = PhasePrediction(**config_dict)
    if args["job_type"] == "train":
        Model.train(eval_model=args["eval_model"], save_model=args["save_model"], plot_metric=args["plot_metric"], remove_ckpt=args["remove_ckpt"])
    elif args["job_type"] == "test":
        Model.test(phaes_size=args["plot_phase_size"], best_metric=args["best_metric"], save_test_zernike=args["save_test_zernike"], plot_test_zernike=args["plot_test_zernike"])
    else:
        raise Exception("job type must be 'train' or 'test'!")



if __name__ == "__main__":
    pred_phase()