import os
import io
import json
import yaml
import pandas as pd
from dash import dash_table, html, callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from main.utils import PROCESSED_PATH
from main.apicalls import api_get_fd


def get_cb_fd_detection(app):

    # ============================= FD DETECTION ========================================

    @app.callback([Output('fd', 'children', allow_duplicate=True),
                Output('memory-fd', 'data', allow_duplicate=True), 
                Output('interval-component', 'max_intervals'),
                Output('upload-state', 'data', allow_duplicate=True)],
                Input('memory-dataset', 'data'),
                Input('interval-component', 'n_intervals'), prevent_initial_call=True)
    def fd_detection(data, max_intervals):
        """ Detects functional dependencies as soon as user uploads a dataset """
        if data is None:
            raise PreventUpdate

        api_response = api_get_fd()

        if api_response:
            # deactivate the Interval
            max_intervals = 0
            fd_json = json.loads(api_response)
            fd_df = pd.DataFrame.from_records(fd_json)

            return dash_table.DataTable(
                fd_json,
                [{'name': i, 'id': i} for i in fd_df.columns],
                style_table={'overflowX': 'auto', 'height': '210px', 'overflowY': 'auto'},
                row_selectable="multi",
                selected_rows=[],
                id='fd-table'
            ), fd_json, 0, False
        else:
            # Check again after five seconds
            max_intervals = max_intervals + 1
            return html.P("No FDs found"), None, max_intervals, False

    # ================================ SAVE SELECTED FDS ====================================
    
    @app.callback(Output('fd-table', 'selected_rows', allow_duplicate=True),
              Input('fd-button', 'n_clicks'),
              [State('fd-table', 'selected_rows'),
              State('memory-fd', 'data'),
              State('memory-filepath', 'data')], prevent_initial_call=True)
    def save_selected_fds(n_clicks, selected_fds, fds, dataset_filepath):
        """ The user can select functional dependencies in the table shown in the "Data Profile" tab and save it to a file
                to be later used for error detection methods such as nadeef
        """
        fd_df = pd.DataFrame.from_records(fds)

        # selected_fds: array of indices of selected rows (0 to n_rows)
        selected_fd_df = fd_df.iloc[selected_fds]

        # filter fds (only one to one allowed)
        selected_fd_df = selected_fd_df[['Determinants', 'Dependant']].copy()
        selected_fd_df = selected_fd_df[~selected_fd_df.Determinants.str.contains(",")]
        selected_fd_df = selected_fd_df[~selected_fd_df.Dependant.str.contains(",")]
        selected_fd_arr = selected_fd_df.values.tolist()

        # save fds for nadeef
        nadeef_config = {"fd_constraints": {"functions": selected_fd_arr}}

        yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(dataset_filepath), 'parameters.yaml'))
        existing_data = {}
        
        # Read the existing YAML file
        if os.path.exists(yaml_file_path):
            with io.open(yaml_file_path, 'r', encoding='utf8') as infile:
                existing_data = yaml.safe_load(infile)  # Load the existing data, if any
        # Update the existing data with new data
        if existing_data is not None:
            existing_data.update(nadeef_config)
        else:
            existing_data = nadeef_config       
        
        with io.open(yaml_file_path, 'w', encoding='utf8') as outfile:
            yaml.dump(existing_data, outfile, default_flow_style=False, allow_unicode=True)

        # save fds for holoclean
        fd_string_template = "t1&t2&EQ(t1.{left_side},t2.{left_side})&IQ(t1.{right_side},t2.{right_side})\n"

        with io.open(os.path.abspath(os.path.join(os.path.dirname(dataset_filepath), "constraints.txt")),
                    "w", encoding='utf8') as text_file:
            for fd in selected_fd_arr:
                text_file.write(fd_string_template.format(left_side=fd[0], right_side=fd[1]))

        return []
    
    # ================================ CUSTOM FD DROPDOWN ====================================
    
    @app.callback(Output('fd-modal', 'is_open'),
                  [Input('open-fd-modal', 'n_clicks'),
                  Input('add-fd-modal', 'n_clicks'),
                  Input('cancel-fd-modal', 'n_clicks')],
                  [State('fd-modal', 'is_open')], prevent_initial_call=True)
    def add_fd_popup(create_button, add_new_fd_button, cancel_modal, is_open):
        """ Opens a popup to allow the user to add a custom functional dependency """
        if create_button or add_new_fd_button or cancel_modal:
            return not is_open
        return is_open
    
    @app.callback([Output('determinants-dropdown', 'options', allow_duplicate=True),
                   Output('dependants-dropdown', 'options', allow_duplicate=True)],
                  [Input('memory-dataset', 'data')], prevent_initial_call=True)
    def update_dropdowns(data):
        """ Updates the dropdowns with the columns of the dataset """
        if data is None:
            # If there's no data, return empty options
            return [], []

        columns = [{'label': col, 'value': col} for col in data[0].keys()]

        return columns, columns
    
    @app.callback([Output('open-fd-modal', 'disabled'),
                   Output('fd-button', 'disabled')],
                 [Input('fd', 'children')])
    def disable_open_fd_button(fd_children):
        """ Disables the open-fd-modal button and the fd-button when the fds have not been detected yet """
        # Check if the fd component has been created
        if isinstance(fd_children, html.P) and fd_children.children == "No FDs found":
            return False, False

        if isinstance(fd_children, dict) and fd_children.get('type') == 'DataTable':
            return False, False

        return True, True
    
    @app.callback(Output('dependants-dropdown', 'options'),
                  Input('determinants-dropdown', 'value'),
                  State('memory-dataset', 'data'))
    def update_dependants_dropdown(selected_determinants, data):
        """ Updates the dependants dropdown based on the selected determinants """
        if selected_determinants is None:
            raise PreventUpdate

        # Filter out the selected determinants from the columns
        columns = [{'label': col, 'value': col} for col in data[0].keys() if col not in selected_determinants]

        return columns
    
    @app.callback(Output('determinants-dropdown', 'options'),
                  Input('dependants-dropdown', 'value'),
                  State('memory-dataset', 'data'))
    def update_determinants_dropdown(selected_dependants, data):
        """ Updates the determinants dropdown based on the selected dependants """
        if selected_dependants is None:
            raise PreventUpdate
        
        # Filter out the selected dependants from the columns
        columns = [{'label': col, 'value': col} for col in data[0].keys() if col not in selected_dependants]

        return columns
    
    # ================================ ADD CUSTOM FD ====================================
    
    @app.callback([Output('fd', 'children'),
                   Output('memory-fd', 'data'),
                   Output('fd-table', 'selected_rows'),
                   Output('alert-message', 'data', allow_duplicate=True),
                   Output('determinants-dropdown', 'value'),
                   Output('dependants-dropdown', 'value')],
                   [Input('add-fd-modal', 'n_clicks')],
                   [State('determinants-dropdown', 'value'),
                   State('dependants-dropdown', 'value'),
                   State('memory-fd', 'data'),
                   State('fd-table', 'selected_rows')], prevent_initial_call=True)
    def handle_add_fd(add_new_fd_button, determinants, dependant, existing_fds, selected_rows):
        """ Handles the functionality of the add-fd-modal button """
        if add_new_fd_button:
            new_fd = {'Determinants': determinants, 'Dependant': dependant}
            
            # Check if new_fd already exists in existing_fds
            if any(new_fd['Determinants'] == fd['Determinants'] and new_fd['Dependant'] == fd['Dependant'] for fd in existing_fds):
                fd_df = pd.DataFrame.from_records(existing_fds)
                fd_table = dash_table.DataTable(
                    existing_fds,
                    [{'name': i, 'id': i} for i in fd_df.columns],
                    style_table={'overflowX': 'auto', 'height': '210px', 'overflowY': 'auto'},
                    row_selectable="multi",
                    selected_rows=selected_rows,
                    id='fd-table'
                )
                return fd_table, existing_fds, selected_rows, "FD already exists!", [], []
            
            updated_fds = [new_fd] + existing_fds
            
            if selected_rows is None:
                selected_rows = [0]
            else: 
                selected_rows =  [i + 1 for i in selected_rows]  # increment each index by 1
                selected_rows.insert(0, 0)  # add the new row at the beginning
            
            fd_df = pd.DataFrame.from_records(updated_fds)
            fd_table = dash_table.DataTable(
                updated_fds,
                [{'name': i, 'id': i} for i in fd_df.columns],
                style_table={'overflowX': 'auto', 'height': '210px', 'overflowY': 'auto'},
                row_selectable="multi",
                selected_rows=selected_rows,
                id='fd-table'
            )
            
            return fd_table, updated_fds, selected_rows, None, [], []
            
        raise PreventUpdate