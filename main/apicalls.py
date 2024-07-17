import requests
from pydantic import BaseModel
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import json
from main.utils import detransform_detection_dict

API_BASE_URL = "http://localhost:8000"
API_ENDPOINT_STORE = "/file_info_store"
API_ENDPOINT_DETECTION_BASIC = "/detection"
API_ENDPOINT_PROFILE = "/profile"
API_ENDPOINT_FD = "/fd"
API_ENDPOINT_REPAIR = "/repair"

API_ENDPOINT_POST_TUPLE = "/labled-tuple"
API_ENDPOINT_GET_RAHA = "/raha-detect"
API_ENDPOINT_RAHA_CANCEL = "/cancel-raha"

API_ENDPOINT_GET_DET_USED = "/error-detection-used"
API_ENDPOINT_GET_REP_USED = "/repair-methods-used"
API_ENDPOINT_GET_DATASET_INFO = "/dataset-info"
API_ENDPOINT_GENERATE_DATASHEET = "/generate-datasheet"

class UploadReq(BaseModel):
    dirty_path: str
    dataset_name: str
    dataset_shape: Tuple[int, int]
    version: Optional[int] = None


class DetectionRequest(BaseModel):
    detection_methods: List[str]
    labeling_budget: int
    uiltags: List


class RepairRequest(BaseModel):
    repair_method: str
    detection_method: str


class UserLabel(BaseModel):
    user_label: Optional[List] = None


def api_store_ds_info(dirty_path, dataset_name, dataset, version):
    shape = dataset.shape
    data = UploadReq(dirty_path=dirty_path, dataset_name=dataset_name, dataset_shape=shape, version=version).dict()

    try:
        response = requests.post(API_BASE_URL + API_ENDPOINT_STORE, json=data, timeout=20)
        response.raise_for_status()

        return True

    except requests.exceptions.RequestException as e:
        return False


def api_basic_detection(detection_methods, uiltags=[], labeling_budget=5):
    data = DetectionRequest(detection_methods=detection_methods, labeling_budget=labeling_budget, uiltags=uiltags).dict()

    try:
        response = requests.post(API_BASE_URL + API_ENDPOINT_DETECTION_BASIC, json=data, timeout=20)
        response.raise_for_status()

        response_json = response.json()
        error_dict = response_json["error_dict"]
        raha_tuple = response_json["raha_tuple"]
        tuples_left = response_json["tuples_left"]
        # error_array = np.array(error_array)
        return error_dict, raha_tuple, tuples_left

    except requests.exceptions.RequestException as e:
        return None, None, None


def api_repair(repair_method, detection_method='default'):
    data = RepairRequest(repair_method=repair_method, detection_method=detection_method).dict()
    try:
        response = requests.post(API_BASE_URL + API_ENDPOINT_REPAIR, json=data, timeout=20)
        response.raise_for_status()
        response_json = response.json()
        return response_json
    except requests.exceptions.RequestException as e:
        return None


def api_data_profile():
    try:
        response = requests.get(API_BASE_URL + API_ENDPOINT_PROFILE)
        response.raise_for_status()
        response_json = response.json()
        return response_json
    except requests.exceptions.RequestException as e:
        return None


def api_send_user_labels(user_label):
    data = UserLabel(user_label=user_label).dict()
    try:
        response = requests.post(API_BASE_URL + API_ENDPOINT_POST_TUPLE, json=data, timeout=20)
        response.raise_for_status()

        response_json = response.json()
        raha_tuple = response_json["raha_tuple"]
        tuples_left = response_json["tuples_left"]
        return raha_tuple, tuples_left

    except requests.exceptions.RequestException as e:
        return None, None


def api_get_raha_detection():
    try:
        response = requests.get(API_BASE_URL + API_ENDPOINT_GET_RAHA, timeout=20)
        response.raise_for_status()

        response_json = response.json()
        p = response_json['p']
        r = response_json['r']
        f = response_json['f']
        detection_dict = response_json['detection_dict']
        detection_dict = detransform_detection_dict(detection_dict)
        return p, r, f, detection_dict

    except requests.exceptions.RequestException as e:
        return None, None, None, None


def api_cancel_raha():
    try:
        response = requests.put(API_BASE_URL + API_ENDPOINT_RAHA_CANCEL, timeout=20)
        response.raise_for_status()

        return True

    except requests.exceptions.RequestException as e:
        return False


def api_get_fd():
    try:
        response = requests.get(API_BASE_URL + API_ENDPOINT_FD, timeout=20)
        response.raise_for_status()
        response_json = response.json()
        return response_json

    except requests.exceptions.RequestException as e:
        print(e)
        return None

def api_generate_datasheet():
    try:
        response = requests.get(API_BASE_URL + API_ENDPOINT_GENERATE_DATASHEET, timeout=20)
        response.raise_for_status()

        response = response.json()
        return response

    except requests.exceptions.RequestException as e:
        return None