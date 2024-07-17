from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import subprocess
import json
import csv

# ----syspath error (TODO fix syspath naturally)
from pathlib import Path
import sys

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
# --------------

import os
from ydata_profiling import ProfileReport
import time

from typing import List, Tuple, Optional
import numpy as np
from main.utils import *

from API.apiutil import *
from baseline.setup.detectors.detect_method import DetectMethod, EXP_PATH
from baseline.setup.detectors.detect import detect
from baseline.setup.utils import create_detections_path
from baseline.setup.utils import store_detections
from baseline.setup.repairs.repair import repair
from deltalake.writer import write_deltalake
from deltalake import DeltaTable
import mlflow
from mlflow import log_param
MAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main"))

# java_class: path in metanome algorithms github (https://github.com/HPI-Information-Systems/metanome-algorithms)
# --file-key could be found in setRelationalInputConfigurationValue method
#           in each algorithm's java class, check what has to be equal to identifier
fd_detection = {"HyFD": {"java_class": "de.metanome.algorithms.hyfd.HyFD", "--file-key": "INPUT_GENERATOR"},
                "TANE": {"java_class": "de.metanome.algorithms.tane.TaneAlgorithm", "--file-key": "Relational Input"}}

description = ''


class LoadDataRequest(BaseModel):
    dirty_path: str = ""
    dataset_name: str = ""
    dataset_shape: Tuple[int, int] = (0, 0)
    version: Optional[int] = None

class DetectionRequest(BaseModel):
    detection_methods: List[str]
    labeling_budget: int
    uiltags: List = []


class RepairRequest(BaseModel):
    repair_method: str
    detection_method: str


class DetectionResponse(BaseModel):
    error_dict: dict
    raha_tuple: dict
    tuples_left: int


class UserLabel(BaseModel):
    user_label: Optional[List] = None


class TupleResponse(BaseModel):
    raha_tuple: dict
    tuples_left: int


class RahaResponse(BaseModel):
    p: float
    r: float
    f: float
    detection_dict: dict


dataset_info = LoadDataRequest()
error_detection_used = set()
repair_method_used = ''
# list of repaired paths
repaired_path = ''
mlflow_runs_detect = []
mlflow_runs_repair = []
uiltags_store = []

# to run server: uvicorn fastAPImain:app --reload
app = FastAPI(title="Detection Method Interface API",
              version="0.0.1")


@app.post("/file_info_store")
async def store_file(file_info: LoadDataRequest):
    global repair_method_used
    global repaired_path
    dataset_info.dirty_path = file_info.dirty_path
    dataset_info.dataset_name = file_info.dataset_name
    dataset_info.dataset_shape = file_info.dataset_shape
    dataset_info.version = file_info.version

    error_detection_used.clear()
    repair_method_used = ''
    repaired_path = ''
    mlflow_runs_detect.clear()
    mlflow_runs_repair.clear()

    if mlflow.active_run():
        mlflow.end_run()

    return {"message": "Dataset stored successfully"}


@app.post("/detection")
async def perform_detection(request: DetectionRequest):
    dirty_path = dataset_info.dirty_path
    dataset_name = dataset_info.dataset_name
    detection_methods = request.detection_methods
    labeling_budget = request.labeling_budget
    dataset_shape = dataset_info.dataset_shape
    dataset_version = dataset_info.version
    uiltags_store = request.uiltags

    # store detection results at the same location as the dataset
    detections_path = os.path.join(os.path.dirname(dirty_path), 'detections.csv')
    # create_detections_path(EXP_PATH, dataset_name, error_detection[method].__str__())

    # delete previous detection results if exist
    try:
        os.remove(detections_path)
    except OSError:
        pass

    arr = np.zeros(dataset_shape)
    tuples_left = 0
    df = pd.DataFrame()
    detection_dict = dict()
    error_dict = {}

    # track params
    mlflow.set_experiment('detection')
    if mlflow.active_run() is None:
        mlflow.start_run()

    if dataset_version is not None:
        log_param('Data_version', dataset_version)
        print('----------------------------------')
        print(f'Data_version {dataset_version}')
        print('----------------------------------')

    for method in detection_methods:
        
        # define the detections path
        detections_path = Path(os.path.join(MAIN_PATH, "..", 'data', 'processed', dataset_name, method, 'detections.csv'))
        detections_dir = detections_path.parent
        # Create the folder if it does not exist
        detections_dir.mkdir(parents=True, exist_ok=True)
        
        if method != 'Raha':
            # dataset_path: folder containing dataset, dirty_path: dataset file
            detection_dict = detect(clean_path=None, dirty_path=dirty_path, detections_path=detections_path,
                                    dataset_path=os.path.dirname(dirty_path), dataset_name=dataset_name,
                                    detect_method=error_detection[method])
            error_dict[method] = list(detection_dict.keys() if isinstance(detection_dict, dict) else detection_dict)
        else:
            df, tuples_left = initialize_raha(dataset_name, dirty_path, nb_labels=labeling_budget)

        arr = dict_to_arr(arr, detection_dict)
        error_detection_used.add(method)

    #adds the uiltags to the error_dict
    detection_dict = mark_uiltags(uiltags_store, dataset_info.dirty_path)
    error_dict['uiltags'] = list(detection_dict.keys() if isinstance(detection_dict, dict) else detection_dict)

    mlflow.end_run()
    last_run = mlflow.last_active_run()
    print(last_run.info.run_id)
    mlflow_runs_detect.append(last_run.info.run_id)

    response_data = DetectionResponse(
        # error_dict contains detection methods as keys and lists of dirty cell positions as values
        error_dict=error_dict,
        raha_tuple=df.to_dict(),
        tuples_left=tuples_left
    )
    return response_data


@app.post("/repair")
async def perform_repair(request: RepairRequest):
    global repair_method_used
    global repaired_path
    global global_detections_path
    # for multiple repair versions
    timestamp = int(time.time())

    # track params
    mlflow.set_experiment('repair')
    if mlflow.active_run() is None:
        mlflow.start_run()

    dataset_name = dataset_info.dataset_name
    dirty_path = dataset_info.dirty_path
    #target_path = os.path.join(os.path.dirname(dirty_path), dataset_name + f"_repair_{timestamp}.csv")
    detections_path = Path(os.path.abspath(os.path.join(MAIN_PATH, "..", 'data', 'processed', dataset_name, request.detection_method, 'detections.csv')))
    target_path = Path(os.path.abspath(os.path.join(MAIN_PATH, "..", 'data', 'processed', dataset_name, request.detection_method, f"_repair_{timestamp}.csv")))
    #detections_path = os.path.join(os.path.dirname(dirty_path), 'detections.csv')
    dataset_version = dataset_info.version

    repaired_df = repair(clean_path=None, dirty_path=dirty_path, target_path=target_path,
                         detections_path=detections_path, dataset_name=dataset_name,
                         repair_method=error_repair[request.repair_method])

    # df = pd.read_csv(dirty_path, header="infer", encoding="utf-8", keep_default_na=False, low_memory=False)
    # sum_diff = compare_sum_df(df, repaired_df)
    # print(sum_diff)

    if dataset_version is not None:  # means that this is an uploaded dataset, not a default dataset
        # write new data to datalake
        deltalake_path = os.path.join(os.path.dirname(dirty_path), "deltalake")
        write_deltalake(deltalake_path, repaired_df, mode="overwrite", overwrite_schema=True)
        dt = DeltaTable(deltalake_path)
        # log version number of repaired dataset
        log_param('Repaired_data_version', dt.version())
        print('----------------------------------')
        print(f'Repaired_version: {dt.version()}')
        print('----------------------------------')

    repair_method_used = request.repair_method
    repaired_path =target_path
    global_detections_path = detections_path.as_posix()

    mlflow.end_run()
    last_run = mlflow.last_active_run()
    mlflow_runs_repair.append(last_run.info.run_id)

    return JSONResponse(content=os.path.abspath(target_path.as_posix()))


@app.get("/profile")
async def data_profile():
    path = dataset_info.dirty_path
    data_name = dataset_info.dataset_name
    df = pd.read_csv(path, header="infer", encoding="utf-8", keep_default_na=False, low_memory=False)

    timestamp = int(time.time())
    profile_path = os.path.join("assets", f"{data_name}_profile_{timestamp}.html")
    profile_abs_path = os.path.abspath(os.path.join(MAIN_PATH,
                                                    "assets", f"{data_name}_profile_{timestamp}.html"))

    if not os.path.exists(os.path.dirname(profile_abs_path)):
        os.makedirs(os.path.dirname(profile_abs_path))

    profile = ProfileReport(df, html={'navbar_show': False})
    profile.to_file(profile_abs_path)

    return JSONResponse(content=profile_path)


@app.get("/fd")
async def detect_fd():
    detection_methods = list(fd_detection.keys())
    data_path = dataset_info.dirty_path
    jars_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jars", "*"))

    df = pd.read_csv(data_path, header="infer", encoding="utf-8", dtype=str, low_memory=False)  # dirty dataset
    results = []

    for method in detection_methods:
        # run Metanome through subprocess.run (command is specific to Windows, for other OS remove "cmd", "/c")
        # if the dataset doesn't have any FDs, then the output file will be empty
        # output is saved in API/results/data_fds
        subprocess.run(["java", "-cp", jars_path, "de.metanome.cli.App", "--algorithm",
                        fd_detection[method]["java_class"], "--file-key", fd_detection[method]["--file-key"], "--files",
                        data_path, "--separator", ",", "--header", "-o", "file:data"])

        # post process results from algorithm into a dataframe and append to results array
        intermediate = pd.DataFrame(columns=['Determinants', 'Dependant', 'Score'])
        intermediate = read_json(df, dataset_info.dataset_shape[1], intermediate)
        intermediate['Determinants'] = [', '.join(map(str, l)) for l in intermediate['Determinants']]

        # line below is used in the original repo (https://github.com/Marini97/Functional-Dependency)
        #   but I think it's buggy
        # intermediate = get_fds(intermediate, dataset_info.dataset_shape[1])

        results.append(intermediate)

    results_fd = pd.concat(results)

    # delete duplicate fds (there could be multiple rows in results_fd that represent the same FD,
    #   but just have the column names in a different order)
    results_fd[['Determinants', 'Dependant']] = results_fd[['Determinants', 'Dependant']].applymap(
        lambda x: ", ".join(sorted(x.split(", "))))
    results_fd = results_fd.drop_duplicates()

    return JSONResponse(content=results_fd.to_json(orient='records'))


@app.post("/labled-tuple")
async def labled_tuple(request: UserLabel):
    user_label = request.user_label
    df, tuples_left = produce_label(user_label)

    response_data = TupleResponse(
        raha_tuple=df.to_dict(),
        tuples_left=tuples_left
    )

    return response_data


@app.get("/raha-detect")
async def raha_detect():
    dataset_name = dataset_info.dataset_name
    dirty_path = dataset_info.dirty_path
    print(dataset_name)

    p, r, f, detection_dict = raha_detect_errors(dataset_name)

    # store detection results at the same location as the dataset
    #detections_path = os.path.join(os.path.dirname(dirty_path), 'detections.csv')
    detections_path = Path(os.path.join(MAIN_PATH, "..", 'data', 'processed', dataset_name, 'Raha', 'detections.csv'))
    store_detections(detection_dict, detections_path)

    detection_dict = tuple_transform(detection_dict)

    response_data = RahaResponse(
        p=p,
        r=r,
        f=f,
        detection_dict=detection_dict
    )
    return response_data


@app.put('/cancel-raha')
async def cancel_raha():
    cancel_raha_input()
    return {"status": "Raha canceled"}

@app.get('/generate-datasheet')
async def generate_datasheet():
    global repair_method_used
    global repaired_path
    error_count = 0
    data_version = 0
    repaired_data_version = 0
    ed_params = []
    for run in mlflow_runs_detect:
        params = mlflow.get_run(run).data.params
        params = {key: json.loads(value) for key, value in params.items()}
        if "Data_version" in params:
            data_version = params.pop("Data_version")
        ed_params.append(params)
        print(params)

    repair_params = []
    for run in mlflow_runs_repair:
        params_re = mlflow.get_run(run).data.params
        params_re = {key: json.loads(value) for key, value in params_re.items()}
        if "Repaired_data_version" in params_re:
            repaired_data_version = params_re.pop("Repaired_data_version")
        repair_params.append(params_re)
        print(params)

    #detections_path = os.path.join(os.path.dirname(dataset_info.dirty_path), 'detections.csv')
    detections_data = []
    with open(global_detections_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for count, row in enumerate(csv_reader):
            if count % 2 == 0:
                detections_data.append((row[0], row[1]))
            if row != []:
                error_count += 1

    datasheet = {
        "dirty_path": dataset_info.dirty_path,
        "repaired_path": repaired_path.as_posix(),
        "original_data_version": data_version,
        "repaired_data_version": repaired_data_version,
        "dataset_name": dataset_info.dataset_name,
        "dataset_shape": dataset_info.dataset_shape,
        "error_detection_methods": list(error_detection_used),
        "error_repair_method": repair_method_used,
        "error_count": error_count
    }

    datasheet.update({"error_detection_params": ed_params})
    datasheet.update({"error_repair_params": repair_params})
    datasheet.update({"errors": detections_data})

    timestamp = int(time.time())
    datasheet_path = os.path.join("main", "assets", "datasheets", f"{dataset_info.dataset_name}_datasheet_{timestamp}.json")
    datasheet_abs_path = os.path.abspath(os.path.join(MAIN_PATH, "assets", "datasheets", f"{dataset_info.dataset_name}_datasheet_{timestamp}.json"))
    
    if not os.path.exists(os.path.dirname(datasheet_abs_path)):
        os.makedirs(os.path.dirname(datasheet_abs_path))

    with open(datasheet_abs_path, 'w') as json_file:
        json.dump(datasheet, json_file, indent=4)

    print(f'datasheet path: {datasheet_path}')

    return JSONResponse(content=datasheet_path)