from helper import *
import argparse
from phase_prediction_model import PhasePrediction

CONFIG_PATH = "./src/config_test.json"
JOB = "test"  # "train" or "test"
BEST_METRIC = "RMSE"
PLOT_PHASE_SIZE = 55
TEST_EPOCH = -1

# args
def parse_args():
    parser = argparse.ArgumentParser(description='Phase Prediction Arguments')
    parser.add_argument('--config_path', default=CONFIG_PATH, type=str, help='config path')
    parser.add_argument('--job', default=JOB, type=str, help='job type, "train" or "test"')
    parser.add_argument('--no_eval_model', action='store_true', help='DO NOT eval model')
    parser.add_argument('--no_save_model', action='store_true', help='DO NOT save model')
    parser.add_argument('--no_plot_metric', action='store_true', help='DO NOT plot metric')
    parser.add_argument('--no_remove_ckpt', action='store_true', help='DO NOT remove the non-last epoch ckpt')
    parser.add_argument('--no_save_test_zernike', action='store_true', help='DO NOT save test zernike')
    parser.add_argument('--no_plot_test_zernike', action='store_true', help='DO NOT plot test zernike')
    parser.add_argument('--plot_phase_size', default=PLOT_PHASE_SIZE, type=int, help='plot phase size, choose in [15, 35, 55, 75, 95, 115]')
    args = parser.parse_args()
    args.eval_model = not args.no_eval_model
    args.save_model = not args.no_save_model
    args.plot_metric = not args.no_plot_metric
    args.remove_ckpt = not args.no_remove_ckpt
    args.save_test_zernike = not args.no_save_test_zernike
    args.plot_test_zernike = not args.no_plot_test_zernike
    return vars(args)


def pred_phase():
    args = parse_args()
    config_dict = Helper.read_json(args["config_path"])
    Model = PhasePrediction(**config_dict)
    if args["job"] == "train":
        Model.train(eval_model=args["eval_model"], save_model=args["save_model"], plot_metric=args["plot_metric"], remove_ckpt=args["remove_ckpt"])
    elif args["job"] == "test":
        Model.test(phase_size=args["plot_phase_size"], save_test_zernike=args["save_test_zernike"], plot_test_zernike=args["plot_test_zernike"])
    else:
        raise Exception("job must be 'train' or 'test'!")



if __name__ == "__main__":
    pred_phase()