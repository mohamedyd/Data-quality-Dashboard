from dash import html, dcc
import dash_bootstrap_components as dbc

from main.utils import error_detection, error_repair, ml_models

# =========================== UILTAGS ==============================

UILTAGS = html.Div([
    html.Label("Tagging values as dirty (press 'Enter' to record each tagged value).", style={'marginRight': '10px'}),
    dcc.Input(id='tag-input', type='text', value='', placeholder='Enter tags', style={'width': '100%', 'height': '30px', 'lineHeight': '30px', 'borderWidth': '1px',
                         'textAlign': 'left', 'margin': '5px'}),
    html.Div(id='tag-container', children=[], style={'display': 'flex', 'flexWrap': 'wrap'}),
], style={"marginTop": 10, "marginLeft": 20})

# ======================= CLEANING ============================

CLEANING = html.Div(
    [
        html.H5("Pick error detection methods", style={"marginTop": 10}),
        dcc.Dropdown(list(error_detection.keys()), id='detection-method', multi=True),
        dbc.Button("Detect errors", id='detect-button', color='primary', outline=True, style={"marginTop": 10}, n_clicks=0), 
        html.H5("Pick error repair methods", style={"marginTop": 10}),
        dcc.Dropdown(list(error_repair.keys()), id='repair-methods'),
        dbc.Button("Repair errors", id='repair-button', color='primary', outline=True,
                   style={"marginTop": 10}),
        html.Div(id='repair-message')
    ],
    className="p-3 bg-light border rounded-3",
    style={"marginTop": 10, "marginLeft": 20},
)

# ===================== ITERATIVE CLEANING ======================

ITERATIVE_CLEANING = html.Div(
    [
        html.Label("Optimizing an ML model performance by auto-selecting premier data cleaning tools." , style={'marginRight': 10}),
        html.H5("Select an ML model", style={"marginTop": 10}),
        dcc.Dropdown(list(ml_models.keys()), id='ml-model-dropdown'),
        html.Div([
        html.Label("Number of Search Iterations (default: 10):", style={"marginTop": 10}),
        dcc.Input(id='num-optuna-trials', type='number', value=10, placeholder='Enter a number', style={"width": "100%"}),
        ], style={"marginTop": 10}),
        html.Div([
        html.Label("Cross-Validation (CV, default: 5):", style={"marginTop": 10}),
        dcc.Input(id='num-optuna-cv', type='number', value=10, placeholder='Enter a number', style={"width": "100%"}),
        ], style={"marginTop": 10}),
        html.Div([
        html.Label("Number of ML Training Epochs (default: 100):", style={"marginTop": 10}),
        dcc.Input(id='num-training-iterations', type='number', value=100, placeholder='Enter a number', style={"width": "100%"}),
        ], style={"marginTop": 10}),
        dbc.Button("Run Iterative Cleaning", id='iterative-clean-button', color='primary', outline=True, n_clicks=0,
                   style={"marginTop": 10}),
        # An output element to display results
        html.Div(id='ic-message')
    ],
    className="p-3 bg-light border rounded-3",
    style={"marginTop": 10, "marginLeft": 20},
)

# =================== CLEANING TABS ===============================

CLEANING_TABS = html.Div([
    dcc.Tabs(
        id="tabs",
        value='tab-1',
        children=[
            dcc.Tab(label='Cleaning Tools', value='tab-1', children=[UILTAGS, CLEANING]),
            dcc.Tab(label='Iterative Cleaning', value='tab-2', children=[ITERATIVE_CLEANING]),
        ],
        style={"marginTop": 20}
    ),
    html.Div(id='tabs-content')
])

# ======================= DETECTION RESULTS ======================

DETECTION_RESULTS = html.Div(
    children=[
        html.H4("Detection Results"),
        dbc.Row(children=[
            dbc.Col(html.P("Graph orientation:"), width=2),
            dbc.Col(html.Div([
                dcc.RadioItems(
                    id='graph-orientation',
                    options=[
                        {'label': 'Vertical', 'value': 'Vertical'},
                        {'label': 'Horizontal', 'value': 'Horizontal'},
                    ],
                    value='Vertical',
                    labelStyle={'display': 'inline-block'},
                    inputStyle={"margin-right": "5px", "margin-left": "15px"}
                ),
            ], className='two columns'), width=4
            )
        ]),
        html.Div(id="bar")
    ],
    className="p-4 bg-light border rounded-3",
    style={"marginTop": 9, "marginBottom": 10})

# ======================= USER LABELING ======================

USER_LABELING = html.Div(
    children=[
        html.H5("User Labeling"),

        # Add a new Div to allow user input for the labeling budget
        html.Div([
            html.Label("Labeling Budget (how many data instances you can label):" , style={'marginRight': 10}),  # A label for the numeric input field
            dcc.Input(
                id='labeling-budget',       # A unique identifier for the input component
                type='number',              # Specifies the input type as a number
                min=1,                      # Optional: sets a minimum value
                step=1,                     # Optional: sets the step size to allow only integers
                placeholder='Enter a Number' # Placeholder text inside the input box when empty
            )
        ], style={'marginBottom': 10}),      # Add some margin below the input field for spacing

        html.Div(id='tuples-left-to-label'),
        dcc.Loading(children=[html.Div(id='user-labeling')], type="circle"),
        html.Div(id='selected-cells'),

        dbc.Row([
            dbc.Col(dbc.Button("Skip", id='submit-button', color='primary', outline=True,
                               style={"marginTop": 10}), width=2),
            dbc.Col(dbc.Button("Cancel", id='cancel-raha-button', color='primary', outline=True,
                               style={"marginTop": 10}), width=2),
        ]),
    ],
    className="p-3 bg-light border rounded-3",
    style={"marginTop": 10, "marginBottom": 10}
)