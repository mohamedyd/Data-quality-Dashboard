#############################################################################
# Detect: implement a method which executes different error detection methods

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
#############################################################################

import os
import pandas as pd
import sys

from baseline.setup.detectors.detect_method import DetectMethod, EXP_PATH, DATA_PATH
from baseline.setup.detectors.outlierDetector import outlierdetector
from baseline.setup.detectors.rahaDetector import raha
from baseline.setup.detectors.mvDetector import mvdetector
from baseline.setup.detectors.fahesDetector import fahes
from baseline.setup.detectors.kataraDetector import katara
from baseline.setup.detectors.nadeefDetector import nadeef
from baseline.setup.detectors.holoCleanDetector import holoclean
from baseline.setup.detectors.dBoostDetector import dboost
from baseline.setup.detectors.minKDetector import min_k
# from baseline.setup.detectors.ed2Detector import ed2

from baseline.setup.utils import get_all_errors, create_detections_path
from baseline.setup.evaluate import evaluate_detector


def detect(clean_path, dirty_path, detections_path, dataset_path, dataset_name, detect_method, mink_threshold=0.4):
    """
    Detect errors in a dataset using various detection methods
    """
    # if os.path.exists(detections_path):
    #     print("[INFO] Detections already exist")
    #     # Load the detections if they already exist
    #     reader = pd.read_csv(detections_path, names=['i', 'j', 'dummy'])
    #     detections = reader.groupby(['i', 'j'])['dummy'].apply(list).to_dict()
    #     return detections

    # Load the dirty data and its ground truth
    dirty_df = pd.read_csv(dirty_path, header="infer", encoding="utf-8", dtype=str, low_memory=False)
    if clean_path is not None:
        clean_df = pd.read_csv(clean_path, header="infer", encoding="utf-8", low_memory=False)

    if detect_method in [DetectMethod.OUTLIER_DETECTOR_IF, DetectMethod.OUTLIER_DETECTOR_SD,
                         DetectMethod.OUTLIER_DETECTOR_IQR]:
        method = detect_method.__str__()
        detections = outlierdetector(dirtydf=dirty_df, detect_method=method, detections_path=detections_path)
        # Evaluate detections, but first find all errors in the dataset
        # all_errors = get_all_errors(dirty_df=dirty_df, clean_df=clean_df)
        # p, r, f1 = evaluate_detector(all_errors=all_errors, detections=detections)
        # print("Precision: {}, Recall: {}, F1 score: {}".format(p, r, f1))

    elif detect_method == DetectMethod.RAHA:
        detections = raha(dataset_name=dataset_name, clean_path=clean_path, dirty_path=dirty_path,
                          detections_path=detections_path)

    elif detect_method == DetectMethod.MV_DETECTOR:
        detections = mvdetector(dirtydf=dirty_df, detections_path=detections_path)

    elif detect_method == DetectMethod.FAHES_DETECTOR:
        detections = fahes(dirtydf=dirty_df,
                           dirty_path=dirty_path,
                           detections_path=detections_path)

    elif detect_method == DetectMethod.KATARA:
        detections = katara(dirtydf=dirty_df, detections_path=detections_path)

    elif detect_method == DetectMethod.NADEEF:
        detections = nadeef(dirty_df=dirty_df,
                            dataset_name=dataset_name,
                            detections_path=detections_path)

    elif detect_method == DetectMethod.HOLOCLEAN:
       detections = holoclean(dirty_df=dirty_df,
                              dataset_name=dataset_name,
                              dataset_path=dataset_path,
                              detections_path=detections_path)

    elif detect_method == DetectMethod.DBOOST:
        detections = dboost(dirty_df=dirty_df,
                            clean_df=clean_df,
                            dataset_name=dataset_name,
                            detections_path=detections_path)

    elif detect_method == DetectMethod.MIN_K:
        detections = min_k(dataset_name=dataset_name,
                           dirty_path=dirty_path,
                           clean_path=clean_path,
                           detections_path=detections_path,
                           threshold=mink_threshold)

    # elif detect_method == DetectMethod.ED2_DETECTOR:
    #     label_cutoff = 20 * clean_df.shape[1]
    #     detections = ed2(dirty_df=dirty_df,
    #                      clean_path=clean_path,
    #                      dataset_name=dataset_name,
    #                      label_cutoff=label_cutoff,
    #                      detections_path=detections_path)

    else:
        raise NotImplemented

    return detections


if __name__ == "__main__":
    # Get the data path
    dataset_name = 'nasa'
    method = DetectMethod.RAHA

    dataset_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name))
    # Retrieve the dirty and clean data
    clean_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'clean.csv'))
    dirty_path = os.path.abspath(os.path.join(DATA_PATH, dataset_name, 'dirty.csv'))

    # Create a path to the detections.csv file
    detections_path = create_detections_path(EXP_PATH, dataset_name, method.__str__())

    detect(clean_path, dirty_path, detections_path, dataset_path, dataset_name, detect_method=method)
