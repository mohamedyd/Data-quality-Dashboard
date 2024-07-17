from dash import html, dcc
import dash_bootstrap_components as dbc


# ========================== DATASET =======================

DATASET = dbc.Card(
    [
        dbc.CardHeader(html.H5("Dataset")),
        dbc.CardBody(
            [
                html.Div(id='output-data-upload'),
            ],
            style={"marginTop": 0, "marginBottom": 0},
        ),
        dcc.Checklist(
            options=[{'label': 'Only show dirty rows', 'value': 'dirty'}],
            style={"marginLeft": 20, "marginBottom": 10},
            inputStyle={"margin-right": "5px"},  # Ensure consistency in quotation marks and units
            id='dirty-row-checkbox'
        ),
    ],
    style={"marginTop": 10, "marginBottom": 0},
)

# =========================== DATA QUALITY ====================

DATA_QUALITY = dbc.Card([
    dbc.CardHeader(html.H5("Data Quality")),
    dbc.CardBody(
        [
            html.Div(id="donut", style={"marginTop": 0, "marginBottom": 0}),
            html.P('Error rate: proportion of dirty cells per column'),
            dcc.Loading(children=[html.Div(id='error-detection-result')], type="circle")
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
], style={"marginTop": 10, "marginBottom": 0, "marginRight": 20}, )