import os
from dash import html, dcc
import dash_bootstrap_components as dbc

from main.utils import get_file_names, transform_file_names, DATASHEET_PATH

# Create the datasheet directory
os.makedirs(DATASHEET_PATH, exist_ok=True)
# Extract the file names
file_names = get_file_names(DATASHEET_PATH)
transformed_file_names = transform_file_names(file_names)

# ===================== DATASHEET ==============================

DATA_SHEET = html.Div(
    children=[
        html.H5("Datasheet"),
        #html.Div(id='data-sheet'), todo
        dcc.Loading(children=[dbc.Button("Generate", id='generate-data-sheet', color='primary', outline=True,
                   style={"marginTop": 10, "marginBottom": 10}),
                   html.Div(id='data-sheet-output')], type="circle")
    ],
    className="p-3 bg-light border rounded-3",
    style={"marginTop": 10, "marginBottom": 10})

# =================== LOAD DATASHEET ===========================

LOAD_DATA_SHEET = html.Div(
    children=[
        html.H5("Load Datasheet"),
        html.Div(id='data-sheet-loadable'),
        dcc.Loading(children=[dcc.Dropdown(id='data-sheet-dropdown', options=[{'label': file_name, 'value': file_name} for file_name in transformed_file_names], multi=False, style={"marginTop": 10, "marginBottom": 10}),
                   html.Div(id='data-sheet-load-output')], type="circle")
    ],
    className="p-3 bg-light border rounded-3",
    style={"marginTop": 10, "marginBottom": 10})