from dash import Dash, html, dcc 
import dash_bootstrap_components as dbc

from main.utils import clear_processed_data

from .callbacks import callbacks
from .layouts import layouts

tab_style = {'padding': '-1', 'line-height': '5vh'}

############################# APP LAYOUT ################################################

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True,
            meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])

app.layout = html.Div(
    [
        layouts['NAVBAR'],
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Tabs(
                            [
                                dcc.Tab(
                                    label='Upload',
                                    children=[layouts['UPLOAD']],
                                    style=tab_style,
                                    selected_style=tab_style
                                ),
                                dcc.Tab(
                                    label='From Database',
                                    children=[layouts['LOAD_SQL']],
                                    style=tab_style,
                                    selected_style=tab_style
                                )
                            ]
                        ),
                        layouts['CLEANING_TABS']
                    ],
                    xs=12, sm=12, md=12, lg=3, xl=3
                ),
                dbc.Col(
                    dcc.Tabs(
                        [
                            dcc.Tab(
                                label='Data Overview',
                                children=[layouts['DATASET'], layouts['USER_LABELING']],
                                style=tab_style,
                                selected_style=tab_style
                            ),
                            dcc.Tab(
                                label='Data Profile',
                                children=[layouts['DATA_PROFILE'], layouts['FD_MODAL']],
                                style=tab_style,
                                selected_style=tab_style
                            ),
                            dcc.Tab(
                                label='Detection Results',
                                children=[layouts['DETECTION_RESULTS']],
                                style=tab_style,
                                selected_style=tab_style
                            ),
                            dcc.Tab(
                                label='Datasheets',
                                children=dcc.Tabs(
                                    [
                                        dcc.Tab(
                                            label='Datasheet',
                                            children=[layouts['DATA_SHEET']],
                                            style=tab_style,
                                            selected_style=tab_style
                                        ),
                                        dcc.Tab(
                                            label='Load Datasheet',
                                            children=[layouts['LOAD_DATA_SHEET']],
                                            style=tab_style,
                                            selected_style=tab_style
                                        )
                                    ]
                                ),
                                style=tab_style,
                                selected_style=tab_style
                            )
                        ]
                    ),
                    xs=12, sm=12, md=12, lg=6, xl=6
                ),
                dbc.Col(layouts['DATA_QUALITY'], xs=12, sm=12, md=12, lg=3, xl=3)
            ]
        ),
        dcc.ConfirmDialog(
            id='alert-dialog',
            displayed=False,
            message=''
        ),
        dcc.Interval(
            id='interval-component',
            interval=2*1000,  # in milliseconds (2*1000ms = 2s)
            n_intervals=0,  # number of times the interval has passed
            max_intervals=0, 
        ),
        # The following stores are for internal state management
        dcc.Store(id='alert-message', storage_type='memory'), # value used to control if an alert message should be shown
        dcc.Store(id='data-sheet-path', storage_type='session'), # stores current datasheet path
        dcc.Store(id='memory-dataset', storage_type='session'), # Uploaded dataset is stored here as JSON string, could potentially be slow for large datasets?
        dcc.Store(id='memory-filename', storage_type='session'),
        dcc.Store(id='memory-filepath', storage_type='session'),
        dcc.Store(id='memory-error', storage_type='session'), # dict containing detection methods as keys and lists of dirty cell positions as values
        dcc.Store(id='memory-uiltags', storage_type='memory', data=[]), # stores user in the loop tags
        dcc.Store(id='raha-signal', storage_type='memory'), # value used to control raha error detection process
        dcc.Store(id='user-label', storage_type='memory'), # stores user labels of currently displayed tuple as list of active_cell dicts
        dcc.Store(id='memory-fd', storage_type='memory'),
        dcc.Store(id='raha-tuple', storage_type='memory'),
        dcc.Store(id='tuples-left', storage_type='memory'),
        dcc.Store(id='ic-signal', storage_type='memory'),
        dcc.Store(id='upload-state', data=True, storage_type='memory'),
    ]
)

############################# CALLBACKS #################################################

for cb_function in callbacks:
    cb_function(app)
    

if __name__ == '__main__':
    clear_processed_data()
    app.run_server(debug=True)
