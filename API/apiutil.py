from baseline.setup.detectors.raha.raha.detection import Detection as Raha_Detection
import numpy as np
import pandas as pd
import os
import mlflow

from main.utils import evaluate_user_labeling, default_datasets, DATA_PATH

RAHA = None
RAHA_DATASET = None


def initialize_raha(dataset_name, dataset_path, nb_labels=5):
    global RAHA, RAHA_DATASET, RAHA_SIGNAL
    RAHA = Raha_Detection(labeling_budget=nb_labels)
    dd = {"name": dataset_name, "path": dataset_path}

    if dataset_name in default_datasets:
        dd["clean_path"] = os.path.abspath(os.path.join(DATA_PATH, dataset_name, "clean.csv"))

    RAHA_DATASET = RAHA.initialize(dd)

    RAHA.sample_tuple(RAHA_DATASET)
    sample_index = RAHA_DATASET.sampled_tuple
    df = RAHA_DATASET.dataframe.iloc[[sample_index]]
    tuples_left = RAHA.LABELING_BUDGET - len(RAHA_DATASET.labeled_tuples)

    return df, tuples_left


def produce_label(user_label):
    if user_label is not None:
        RAHA_DATASET.labeled_tuples[RAHA_DATASET.sampled_tuple] = 1

        for j in range(RAHA_DATASET.dataframe.shape[1]):
            cell = (RAHA_DATASET.sampled_tuple, j)

            if user_label is None or not any(d['column'] == j for d in user_label):
                RAHA_DATASET.labeled_cells[cell] = [0, 'dummy_value']
            else:
                RAHA_DATASET.labeled_cells[cell] = [1, 'dummy_value']

    if len(RAHA_DATASET.labeled_tuples) < RAHA.LABELING_BUDGET:
        RAHA.sample_tuple(RAHA_DATASET)
        sample_index = RAHA_DATASET.sampled_tuple
        df = RAHA_DATASET.dataframe.iloc[[sample_index]]
        tuples_left = RAHA.LABELING_BUDGET - len(RAHA_DATASET.labeled_tuples)
        return df, tuples_left

    else:
        return pd.DataFrame(), 0


def raha_detect_errors(dataset_name):
    detection_dict = RAHA.detection(RAHA_DATASET)

    p = 0.0
    r = 0.0
    f = 0.0

    if dataset_name in default_datasets:
        p, r, f = evaluate_user_labeling(RAHA_DATASET)

    return p, r, f, detection_dict


def tuple_transform(detection_dict):
    transform_dict = {}
    for key, value in detection_dict.items():
        new_key = '-'.join(map(str, key))
        new_value = (key, value)
        transform_dict[new_key] = new_value

    return transform_dict


def cancel_raha_input():
    global RAHA, RAHA_DATASET
    RAHA = None
    RAHA_DATASET = None
    return True

def compare_sum_df(df1, df2):
    # get the number of different cells in the two dataframes
    number_of_differences = np.sum(df1 != df2).sum()
    

    return number_of_differences

def compile_params(run_id):
    params = mlflow.get_run(run_id).data.params
    return params

def mark_uiltags(uiltags, dirty_path):
    tag_dict = dict()
    df = pd.read_csv(dirty_path, header="infer", encoding="utf-8", dtype=str, low_memory=False, keep_default_na=False)

    result_indices = []
    for tag in uiltags:
        for col in df.columns:
            indices = df.index[df[col] == tag].tolist()

            col_loc = df.columns.get_loc(col)

            result_indices.extend([(i, col_loc) for i in indices])
    
    tag_dict = {index: "JUST A DUMMY VALUE" for index in result_indices}
    return tag_dict