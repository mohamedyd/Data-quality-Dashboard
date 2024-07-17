from dash import html, dcc
import dash_bootstrap_components as dbc


# ==================== DATA PROFILE =============================

DATA_PROFILE = html.Div(
    children=[
        html.H5("Data Profile"),
        html.H6("Functional dependencies", style={"marginTop": 10, "marginBottom": 10}),
        html.Div(id='fd'),
        dbc.Button("Create new FD", id='open-fd-modal', color='primary', outline=True,
                   style={"marginTop": 10, "marginBottom": 10, "marginRight": 5}),
        dbc.Button("Save selected FDs", id='fd-button', color='primary', outline=True,
                   style={"marginTop": 10, "marginBottom": 10, "marginRight": 5}),
        html.Label("Only one-to-one FDs constraints can be stored (to be used with NADEEF & HoloClean detection tools)." , style={'marginRight': 10}),
        html.H6("Dataset properties", style={"marginTop": 10, "marginBottom": 10}),
        dcc.Loading(children=[html.Iframe(id='data-profile', style={"height": "600px", "width": "100%"})],
                    type="circle")
    ],
    className="p-3 bg-light border rounded-3",
    style={"marginTop": 10, "marginBottom": 10})

# ==================== New FDs Popup =============================

FD_MODAL = dbc.Modal(
    [
        dbc.ModalHeader("Create new FDs"),
        dbc.ModalBody(
            html.Div([
                html.Label('Determinants'),
                dcc.Dropdown(id='determinants-dropdown', multi=False),
                html.Label('Dependants'),
                dcc.Dropdown(id='dependants-dropdown', multi=False)
            ])
        ),
        dbc.ModalFooter(
            html.Div([
                dbc.Button("Add FD", id="add-fd-modal", color='primary', outline=True,
                   style={"marginRight": 5}),
                dbc.Button("Cancel", id="cancel-fd-modal", color='primary', outline=True)
            ])
        ),
    ],
    id="fd-modal",
)