#####################################################################################
# Fahes: implement the FAHES detector to detect outliers and disguised missing values

# Authors: Mohamed Abdelaal, Christian Hammacher
# Date: July 2022
# Software AG
# All Rights Reserved
#####################################################################################

import sys
import os
import pandas as pd
import baseline.setup.detectors.pFAHES.common as common
import baseline.setup.detectors.pFAHES.patterns as patterns
import baseline.setup.detectors.pFAHES.DV_Detector as DV_Detector
import baseline.setup.detectors.pFAHES.RandDMVD as RandDMVD
import baseline.setup.detectors.pFAHES.OD as OD
from baseline.setup.utils import store_detections


def fahes(dirtydf, dirty_path, detections_path):
    detection_dir = os.path.dirname(detections_path)

    # check output directory
    if not os.path.isdir(detection_dir):
        try:
            os.makedirs(detection_dir)
        except OSError as e:
            print("Error creating directory!")
            sys.exit(1)

    # run fahes and get path to results .csv
    sus_dis_values = []
    sus_dis_values, ptrns = patterns.find_all_patterns(dirtydf, sus_dis_values)
    sus_dis_values = DV_Detector.check_non_conforming_patterns(dirtydf, sus_dis_values)
    sus_dis_values = RandDMVD.find_disguised_values(dirtydf, sus_dis_values)
    sus_dis_values = OD.detect_outliers(dirtydf, sus_dis_values)

    common.print_output_data(detection_dir, dirty_path, sus_dis_values)
    result_path = os.path.join(detection_dir, "DMV_" + os.path.basename(dirty_path))

    # load results .csv as dataframe
    fahes_res_df = pd.read_csv(
        result_path,
        dtype=str,
        header="infer",
        encoding="utf-8",
        keep_default_na=False,
        low_memory=False
    )

    detection_dictionary = {}

    # for each entry in fahes results go through the respective
    # column in dirtydf and mark every cell as detected that
    # has the DMV value defined in the fahes results entry
    for i_fahes, row_fahes in fahes_res_df.iterrows():
        for j_dirty, row_dirty in dirtydf.iterrows():
            # get index of respective column in dirtydf
            col_index = dirtydf.columns.get_loc(row_fahes["Attribute Name"])
            if row_dirty[row_fahes["Attribute Name"]] == row_fahes["DMV"]:
                detection_dictionary[(j_dirty, col_index)] = "JUST A DUMMY VALUE"

    # store detections in detector directory
    store_detections(detection_dictionary, detections_path)

    return detection_dictionary
