from dash import html, dcc
import dash_bootstrap_components as dbc

from main.utils import default_datasets


# ============================= NAVBAR =============================

NAVBAR = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(dbc.Col(dbc.NavbarBrand("Data Quality Dashboard", style={"font-size": 22})), justify='start'),
            ),
        ]
    ),
    color="dark",
    dark=True,
)

# ========================== UPLOAD ================================

UPLOAD = html.Div(
    [
        html.H5("Upload a dataset", style={"marginTop": 10}),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={'width': '100%', 'height': '60px', 'lineHeight': '60px', 'borderWidth': '1px',
                   'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'margin': '5px'},
            multiple=False
        ),
        dcc.Input(id='data-version', type='text', placeholder='(Optional) Data version id',
                  style={'width': '100%', 'height': '30px', 'lineHeight': '30px', 'borderWidth': '1px',
                         'textAlign': 'left', 'margin': '5px'},
                  ),
        dbc.Row([dbc.Col(dbc.Button('Discard', id='discard-data', color='primary', outline=True,
                                    style={'font-size': '11px', 'height': '30px'}), width=3),
                 dbc.Col(html.Div(id='file-name', style={'textAlign': 'center', "marginTop": 2, "width": "100%"}),
                         width=4)]),

        html.P("Or use one of our default datasets", style={"marginTop": 20}),
        dcc.Dropdown(
            options=default_datasets,
            id="default-dataset", clearable=True, style={"marginTop": 10, "marginBottom": 10, "font-size": 12},
            persistence=True
        ),
        dbc.Button(children='Upload', id='upload-button', color='primary', outline=True, style={ "marginTop": 10}),
        html.Div(id='data-version-error-msg'),
    ],
    className="p-3 bg-light border rounded-3",
    style={"marginTop": 10, "marginLeft": 20},
)

# ====================== LOAD SQL =================================

LOAD_SQL = html.Div(
    [
        html.H5("Connect to a database", style={"marginTop": 10}),
        dcc.Dropdown(
            id='server-dropdown',
            options=[
                {'label': 'SQL Server', 'value': 'sqlserver'},
                {'label': 'MySQL', 'value': 'mysql'},
                {'label': 'PostgreSQL', 'value': 'postgresql'}
            ],
            value='sqlserver',
            clearable=False,
            style={'border-radius': '5px', 'borderWidth': '1px', 'margin-bottom': '10px'}
        ),
        dcc.Input(id='server-input', type='text', placeholder='servername',
                  style={'border-radius': '5px', 'borderWidth': '1px', 'margin-bottom': '10px'}),
        dcc.Input(id='database-input', type='text', placeholder='database',
                  style={'border-radius': '5px', 'borderWidth': '1px', 'margin-bottom': '10px'}),
        dcc.Input(id='table-input', type='text', placeholder='table',
                  style={'border-radius': '5px', 'borderWidth': '1px', 'margin-bottom': '10px'}),
        dcc.Input(id='uid-input', type='text', placeholder='username',
                  style={'border-radius': '5px', 'borderWidth': '1px', 'margin-bottom': '10px'}),
        dcc.Input(id='password-input', type='password', placeholder='password',
                  style={'border-radius': '5px', 'borderWidth': '1px', 'margin-bottom': '10px'}),
        dcc.Input(id='port-input', type='text', placeholder='port',
                  style={'border-radius': '5px', 'borderWidth': '1px', 'margin-bottom': '10px'}),
        dbc.Button("Connect", id='connect-button', color='primary', outline=True),
        dcc.ConfirmDialog(id='connection-alert',
                          message='Connection failed!',
                          displayed=False,
                          )
    ],
    id='load-sql',
    className="p-3 bg-light border rounded-3",
    style={"marginTop": 10, "marginLeft": 20, "display": "flex", "flex-direction": "column"},
)
