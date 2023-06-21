"""
The main module/function is used to structure the program flow.
From here the other program parts like the data generation are called.
"""

import numpy as np
from ogb.linkproppred import Evaluator
from configparser import ConfigParser, ExtendedInterpolation
import time
import utils as ut
from myfm import MyFMClassifier, MyFMOrderedProbit, MyFMRegressor
from sklearn.metrics import mean_squared_error
from data_generation import get_data
from logger import logging_setup, save_pred
import statistics as stats


logger = logging_setup(__name__)

config = ConfigParser(interpolation=ExtendedInterpolation())
config.read("config.ini")


def main():
    """
    Run Experiments with defined parameters.
    Computed files are stored in data directory.
    Saves results in Excel file.
    """
    logger.info("Program started")

    new_neg_samples = config["RUNS"].getboolean("del_neg_edges")
    runs = config["RUNS"].getint("number")
    n_iter = config["RUNS"].getint("iter")
    rank = config["RUNS"].getint("rank")
    # Command is used as indicator in excel file
    # Change when using a different method
    command = f"MyFMRegressor rank:{rank} n_iter:{n_iter} runs: {runs}"

    hits = []
    rmse = []
    for a in range(1, runs + 1):
        if new_neg_samples:
            ut.delete_precomp_files()

        ## Get or Generate Data for Training
        # Type should be train, valid or test
        X_train, y_train, groups = get_data(typ=f"train")
        # X_valid, y_valid = get_daata(name=f'valid_{d_name}')
        X_test, y_test, _ = get_data(typ=f"test")

        start_time = time.time()

        # Model can be exchanged with other myfm-methods
        fm = MyFMRegressor(rank=rank)
        fm.fit(X_train, y_train, n_iter=n_iter, group_shapes=groups)
        p = fm.predict(X_test)
        hits.append(evaluate([p], y_test))
        rmse.append(np.sqrt(mean_squared_error(y_test, p)))

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(
            f"Method FMRegressor took {elapsed_time} to run for {n_iter} iterations"
        )
        print(f"Hits@x: {hits[-1]}")

        logger.info(f"Finished run {a}")

    average = stats.mean(hits)
    if len(hits) > 1:
        stddev = stats.stdev(hits)
    else:
        stddev = 0
    rand = ut.positive_in_top_k_prob()
    save_pred(command, average, stddev, rand)

    logger.info("Programm finished")
    return


def evaluate(preds: list, y_test: np.ndarray) -> float:
    """
    Calculate OGB Metric for specified graph.
    This can be Hits@X for example.
    Takes in predictions of model and groundtruth
    """
    for y_pred in preds:
        y_pred_neg = []
        y_pred_pos = []
        for i, val in enumerate(y_test):
            if val == 0:
                y_pred_neg.append(float(y_pred[i]))
            elif val > 0:
                y_pred_pos.append(float(y_pred[i]))
            else:
                raise ValueError(
                    f"The test data is not in the correct format. Value < 0 : {val}."
                )
        y_pred_pos = np.float_(y_pred_pos)
        y_pred_neg = np.float_(y_pred_neg)
        input_dict = {"y_pred_pos": y_pred_pos, "y_pred_neg": y_pred_neg}
        evaluator = Evaluator(name=config["STANDARD"]["graph_name"])
        result_dict = evaluator.eval(input_dict)
    return list(result_dict.values())[0]


if __name__ == "__main__":
    main()
