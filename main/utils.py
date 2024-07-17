import base64
import io
import os
import pandas as pd
import numpy as np
import json
import jsonlines
from pathlib import Path
import shutil
from baseline.setup.detectors.detect_method import DetectMethod
from baseline.setup.repairs.repair import RepairMethod
from dash import html, dcc
import dash_bootstrap_components as dbc
import urllib
from datetime import datetime

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../", "data", "raw"))
PROCESSED_PATH = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../", "data", "processed")))

MAX_LEN = 10000  # maximum number of rows when displaying a table on the dashboard
PAGE_SIZE = 10  # number of ro:ws displayed on a datatable in one page

# Define the path for the datasheets directory within the 'assets' folder
DATASHEET_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets", "datasheets"))

default_datasets = ['beers', 'flights', 'hospital', 'movies_1', 'rayyan', 'tax', 'toy', 'nasa']

error_detection = {'Outlier detection (SD)': DetectMethod.OUTLIER_DETECTOR_SD,
                   'Outlier detection (IQR)': DetectMethod.OUTLIER_DETECTOR_IQR,
                   'Outlier detection (IF)': DetectMethod.OUTLIER_DETECTOR_IF,
                   'Fahes': DetectMethod.FAHES_DETECTOR,  # missing values
                   'Nadeef': DetectMethod.NADEEF,  # rule violation
                   #'Katara': DetectMethod.KATARA,  # pattern violation
                   'HoloClean': DetectMethod.HOLOCLEAN,  # rule violation
                   'MV Detector': DetectMethod.MV_DETECTOR,  # missing values
                   'Raha': DetectMethod.RAHA,  # holistic
                   'Min K': DetectMethod.MIN_K  # holistic
                   }

# stores which error detection methods correspond to which metrics
error_detection_metrics = {'Outlier detection (SD)': "Outlier",
                           'Outlier detection (IQR)': "Outlier",
                           'Outlier detection (IF)': "Outlier",
                           'Fahes': 'Missing Value',
                           'Nadeef': "Rule Violation",
                           #'Katara': "Pattern Violation",
                           'HoloClean': "Rule Violation",
                           'MV Detector': 'Missing Value',
                           'Raha': "Others",
                           'Min K': "Others",
                           'uiltags': "User tagging"
                           }

error_repair = {"ML imputer": RepairMethod.ML_IMPUTER,
                "Standard imputer": RepairMethod.STANDARD_IMPUTER}

# Define machine learning models
ml_models = {
    'Neural Network': 'neural_network',
    'Decision Tree': 'decision_tree',
    'Random Forest': 'random_forest',
    # ... potentially more models ...
}


def combine_csv_files(dataset_name, combined_dir_name='combined_detections'):
    # Define the base processed directory
    processed_dir = PROCESSED_PATH / dataset_name

    # Create a combined directory if it doesn't exist
    combined_dir = PROCESSED_PATH / dataset_name / combined_dir_name
    combined_dir.mkdir(exist_ok=True)
    
    # Initialize a list to hold the DataFrames to concatenate
    df_list = []

    # Iterate through each subdirectory in the processed directory
    for folder in processed_dir.iterdir():
        if folder.is_dir():
            # Iterate through each CSV file in the subdirectory
            for csv_file in folder.glob('detections.csv'):
                # check if file has any entries
                if os.stat(csv_file).st_size == 0:
                    continue
                # Read the current CSV file
                df_list.append(pd.read_csv(csv_file, sep=',', header=None))

    # Concatenate all the dataframes in the list, and remove duplicates
    combined_df = pd.concat(df_list, ignore_index=True).drop_duplicates()

    # Save the combined data to a CSV file in the combined directory
    output_file = combined_dir / 'detections.csv'
    combined_df.to_csv(output_file, index=False, header=False)
    print(f"Combined CSV saved to {output_file}")
    

def clear_processed_data():
    
    # Check if the directory exists
    if PROCESSED_PATH.exists() and PROCESSED_PATH.is_dir():
        # Iterate over each item within the directory
        for item in PROCESSED_PATH.iterdir():
            if item.is_file():
                # If it's a file, delete it
                item.unlink()
            elif item.is_dir():
                # If it's a directory, delete it and all its contents
                shutil.rmtree(item)
        print(f"All contents of '{PROCESSED_PATH}' have been removed.")
    else:
        print(f"The directory '{PROCESSED_PATH}' does not exist or is not a directory.")
    

def split_error_dict_per_metric(error_dict):
    """ Converts dict containing error detection methods as keys and list of tuples containing index of
    dirty cells as values into dict containing error detection metrics as keys and corresponding list of
    position of dirty cells as values """
    new_dict = {}

    for k, v in error_dict.items():
        metric = error_detection_metrics[k]
        if metric in new_dict:
            new_dict[metric] = new_dict[metric] + v
        else:
            new_dict[metric] = v

    return new_dict


def error_dict_to_arr(memory_error, dataset_shape):
    """ Converts dict containing data quality metrics as keys and list of tuples containing index of dirty cells
    as values into array containing 1s in locations of dirty cells """

    arr = np.zeros(dataset_shape)
    for val in memory_error.values():
        for loc in val:
            arr[loc[0], loc[1]] = 1

    return arr


def color_cells(color_array, data, max_length):
    """
        Converts numpy array containing locations of dirty cells into style data that specifies style of each cell
        in the DataTable displayed on the dashboard

        Parameters:
            color_array: numpy array
            data: dataframe
            max_length: maximum number of rows showed at once
        Returns:
            a list of dictionaries to be passed to the 'style_data_conditional' attribute of a DataTable
    """

    data = data.to_dict(orient='records')
    style_data = []  # list of dictionaries specifying the style for each cell

    for i, row in enumerate(data):
        for j, value in enumerate(row.values()):
            if i >= max_length:
                break
            if color_array[i, j] == 1:
                style_data.append({'if': {'row_index': i, 'column_id': list(row.keys())[j]},
                                   'backgroundColor': '#FF0000', 'color': 'white'})

    return style_data


def parse_contents(contents, filename):
    """ Parse uploaded dataset from encoded string to dataframe """

    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename or 'xlsx' in filename:
            # Assume that the user uploaded an Excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except:
        print('There was an error processing this file.')

    return df


def dict_to_arr(error_array, detection_dict):
    """
        Gets locations of dirty cells from detection_dict (where keys represent i,j of dirty cells)
        and transfers it to the given error_array.
    """
    for key in detection_dict:
        error_array[key] = 1
    return error_array


def evaluate_user_labeling(d):
    """
        Evaluate raha detection results when ground truth dataset is available.
        Method accepts Dataset object
    """

    # dicts with index (i, j) as key and dummy value
    detection_dict = d.detected_cells  # errors detected by raha
    actual_errors_dictionary = d.get_actual_errors_dictionary()  # errors from ground truth dataframe

    detection_keys = set(detection_dict.keys())
    actual_errors_keys = set(actual_errors_dictionary.keys())

    precision = len(detection_keys & actual_errors_keys) / len(detection_keys)
    recall = len(detection_keys & actual_errors_keys) / len(actual_errors_keys)
    f = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f


def read_json(df: pd.DataFrame, df_cols: int, results: pd.DataFrame) -> pd.DataFrame:
    """Read the output file from Metanome and return a dataframe with the results.

    Args:
        df : pd.DataFrame
            The dataframe with the data.
        df_cols : int
            The number of columns in the dataset.
        results : pd.DataFrame
            The dataframe with the results.

    Returns:
        results : pd.DataFrame
            A pandas dataframe with the results.
    """
    try:
        # todo may have to delete 'results/data_fds' before new dataset is uploaded
        result_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "API", "results", "data_fds"))

        # read the output file from Metanome
        with jsonlines.open(result_path) as lines:
            for line in lines:

                # get the determinants and dependant attributes
                determinants = line['determinant']['columnIdentifiers']
                for i in range(len(determinants)):
                    determinants[i] = determinants[i]['columnIdentifier']

                dependant = line['dependant']['columnIdentifier']

                # if the union of Determinants and Dependant contains all attributes, then the FD is filtered out
                if len(determinants) + 1 < df_cols:
                    score = df[determinants + [dependant]].drop_duplicates().shape[0]
                    row = {'Determinants': determinants, 'Dependant': dependant, 'Score': score}
                    results.loc[len(results)] = row

    except Exception as exception:
        print(exception)
    if len(results) == 0:
        return pd.DataFrame(columns=['Determinants', 'Dependant', 'Score'])
    return results


def get_fds(results: pd.DataFrame, df_cols: int) -> pd.DataFrame:
    """This function takes the results of the FD algorithm and returns a dataframe with the FDs as columns and the attributes as rows.

    Args:
        results : pd.DataFrame
            The results of the FD algorithm.
        df_cols : int
            The number of columns in the dataframe.

    Returns:
        rows : pd.DataFrame
            A dataframe with Determinants, Dependant and Score as attributes and FDs as rows.
    """
    # Convert the determinants to a string
    results['Determinants'] = [', '.join(map(str, l)) for l in results['Determinants']]

    # Create a new dataframe with the columns 'Determinants', 'Dependant' and 'Score'
    # rows = pd.DataFrame(columns=['Determinants', 'Dependant', 'Score'])
    rows = results.copy()

    # Group the dataframe by the determinants
    rows = rows.groupby('Determinants').agg({'Dependant': ', '.join, 'Score': 'max'}).reset_index()

    # Loop through all rows of the dataframe
    for row in rows.itertuples():
        # Loop through all rows of the dataframe again
        for row2 in rows.itertuples():
            # If the determinants of row2 are a subset of the determinants of row
            if row.Index != row2.Index and set(row2.Determinants.split(", ")).issubset(
                    set(row.Determinants.split(", "))):
                # Add the dependants of row2 to the dependants of row
                rows.loc[row.Index, 'Dependant'] = rows.loc[row.Index, 'Dependant'] + ", " + row2.Dependant

    # Remove all rows with more than df_cols columns
    for row in rows.itertuples():
        if len(row.Determinants.split(", ")) + len(row.Dependant.split(", ")) >= df_cols:
            rows = rows.drop(row.Index)

    # Return the dataframe
    return rows


def detransform_detection_dict(detection_dict):
    detransformed_dict = {}
    for key, value in detection_dict.items():
        tuple_key = tuple(int(x) for x in key.split('-'))
        detransformed_value = value[1]
        detransformed_dict[tuple_key] = detransformed_value

    return detransformed_dict

def load_datasheet(path):
    """Load datasheet from given path"""
    with open(path, 'r') as f:
        datasheet = json.load(f)

    rel_dirty_path = os.path.relpath(datasheet['dirty_path'], '')
    rel_repaired_path = os.path.relpath(datasheet['repaired_path'], '')

    # repaired_path_list = [
    #     html.P(f"{name}: {os.path.relpath(repair_path, '')} \n")
    #     for name, repair_path in zip(datasheet["error_repair_methods"], datasheet["repaired_paths"])
    # ]

    detection_methods_list = [
        html.P(f"{name}\n")
        for name in datasheet["error_detection_methods"]
    ]

    error_detection_parameter_list = datasheet.get("error_detection_params", [])

    error_repair_parameter_list = datasheet.get("error_repair_params", [])    

    if datasheet['original_data_version'] != 0:
        dash_component = [
            html.Strong("Dataset Name:"),
            html.P(datasheet["dataset_name"]),
            html.P(f"Original dataset version: {datasheet['original_data_version']}"),
            html.P(f"Repaired dataset version: {datasheet['repaired_data_version']}"),
            html.Strong("Dirty Dataset:"),
            html.P(rel_dirty_path),
            html.Strong(f"Repaired Dataset ({datasheet['error_repair_method']}):"),
            html.P(rel_repaired_path),
            html.Strong("Original Dataset Shape:"),
            html.P(f"Collumns: {datasheet['dataset_shape'][1]} \n Rows: {datasheet['dataset_shape'][0]}"),
            html.Strong('Error Detection Methods:'),
            html.Div(detection_methods_list),
            html.Strong("Miscellaneous:"),
            html.P(f'Number of Errors found by Detection Methods: {datasheet["error_count"]}'),
            *[html.Div([
                html.P(f'{method}:'),
                html.Ul([html.Li(f"{param}: {value}") for param, value in parameters.items()])
            ]) for detection_params in error_detection_parameter_list for method, parameters in detection_params.items()],
            *[html.Div([
                html.P(f'{method}:'),
                html.Ul([html.Li(f"{param}: {value}") for param, value in parameters.items()])
            ]) for repair_params in error_repair_parameter_list for method, parameters in repair_params.items()],
            dcc.Input(id='datasheet-file-name-input', type='text', placeholder='Enter file name', style={'border-radius': '5px', 'borderWidth': '1px', 'margin-bottom': '10px', 'margin-right': '10px'}),
            dbc.Button("Download", id='download-data-sheet-button', color='primary', outline=True,
                    style={"marginTop": 10, "marginBottom": 10}),
            dcc.Download(id='download-data-sheet'),
            dcc.Download(id='download-dirty-dataset'),
            html.Div(id='download-data-sheet-confirm'),
        ]

    else:
        dash_component = [
            html.Strong("Dataset Name:"),
            html.P(datasheet["dataset_name"]),
            html.P(f"Unspecified data version"),
            html.Strong("Dirty Dataset:"),
            html.P(rel_dirty_path),
            html.Strong(f"Repaired Dataset ({datasheet['error_repair_method']}):"),
            html.P(rel_repaired_path),
            html.Strong("Original Dataset Shape:"),
            html.P(f"Collumns: {datasheet['dataset_shape'][1]} \n Rows: {datasheet['dataset_shape'][0]}"),
            html.Strong('Error Detection Methods:'),
            html.Div(detection_methods_list),
            html.Strong("Miscellaneous:"),
            html.P(f'Number of Errors found by Detection Methods: {datasheet["error_count"]}'),
            *[html.Div([
                html.P(f'{method}:'),
                html.Ul([html.Li(f"{param}: {value}") for param, value in parameters.items()])
            ]) for detection_params in error_detection_parameter_list for method, parameters in detection_params.items()],
            *[html.Div([
                html.P(f'{method}:'),
                html.Ul([html.Li(f"{param}: {value}") for param, value in parameters.items()])
            ]) for repair_params in error_repair_parameter_list for method, parameters in repair_params.items()],
            dcc.Input(id='datasheet-file-name-input', type='text', placeholder='Enter file name', style={'border-radius': '5px', 'borderWidth': '1px', 'margin-bottom': '10px', 'margin-right': '10px'}),
            dbc.Button("Download", id='download-data-sheet-button', color='primary', outline=True,
                    style={"marginTop": 10, "marginBottom": 10}),
            dcc.Download(id='download-data-sheet'),
            dcc.Download(id='download-dirty-dataset'),
            html.Div(id='download-data-sheet-confirm'),
        ]

    return dash_component

def get_file_names(directory_path):
    file_names = []
    for filename in os.listdir(directory_path):
        file_names.append(filename)
    
    return file_names

def transform_file_names(file_names):
    transformed_file_names = []
    for filename in file_names:
        # name_datasheet_timestamp to "name (transformed_timestamp)"
        name = filename.split("_")[0]
        timestamp = filename.split("_")[2]
        timestamp = timestamp.split(".")[0]
        # transform timestamp to string with readable timestamp
        timestamp = datetime.fromtimestamp(int(timestamp)).strftime("%H:%M:%S %Y-%m-%d")
        transformed_file_names.append(f"{name} {timestamp}")
    
    return transformed_file_names

def transform_file_name_to_timestamp(name):
    # name (datetime) to name_datasheet_timestamp
    name_data = name.split(" ")[0]
    timestamp_read = name.split(" ")[1] + " " + name.split(" ")[2]
    timestamp_unix = int(datetime.strptime(timestamp_read, "%H:%M:%S %Y-%m-%d").timestamp())
    timestamp = str(int(timestamp_unix))
    return f"{name_data}_datasheet_{timestamp}.json"