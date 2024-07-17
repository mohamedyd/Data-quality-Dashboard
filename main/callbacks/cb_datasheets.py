import os
import json

from dash import html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from main.apicalls import api_generate_datasheet
from main.utils import load_datasheet, transform_file_names, transform_file_name_to_timestamp, get_file_names, DATASHEET_PATH


def get_cb_datasheets(app):
    
    # ================================== GENERATE DATASHEET =====================================

    @app.callback(Output('data-sheet-output', 'children'),
              Output('data-sheet-dropdown', 'options'),
              Output('data-sheet-path', 'data', allow_duplicate=True),
              Output('alert-message', 'data'),
              Input('generate-data-sheet', 'n_clicks'), prevent_initial_call=True)
    def generate_data_sheet(n_clicks):
        
        if n_clicks is None:
            raise PreventUpdate
                
        datasheet_path = api_generate_datasheet()

        new_file_names = get_file_names(DATASHEET_PATH)
        new_transformed_file_names = transform_file_names(new_file_names)
        #return [html.A('Download Data Sheet', href=datasheet_path, download='dataset.json'), html.P(f'File saved at {datasheet_path}')]
        if datasheet_path is not None:
            component = load_datasheet(datasheet_path)
            return component, new_transformed_file_names, datasheet_path, None
        
        return None, None, None, "Error generating datasheet."

    # ================================ LOAD DATASHEET =========================================        
  
    @app.callback(Output('data-sheet-load-output', 'children'),
                Output('data-sheet-path', 'data'),
                Output('alert-message', 'data', allow_duplicate=True),
                Input('data-sheet-dropdown', 'value'), prevent_initial_call=True)
    def load_data_sheet(value):
        if value is None:
            raise PreventUpdate
        datasheet_name_time = transform_file_name_to_timestamp(value)
        datasheet_path = os.path.abspath(os.path.join(DATASHEET_PATH, datasheet_name_time))
        if datasheet_path is not None:    
            component = load_datasheet(datasheet_path)
            return component, datasheet_path, None
        
        return None, None, "Datasheet does not exist."


    # =============================== DOWNLOAD DATASHEET =======================================

    @app.callback(Output('download-data-sheet-confirm', 'children'),
                Output('download-data-sheet', 'data'),
                Output('download-dirty-dataset', 'data'),
                Input('download-data-sheet-button', 'n_clicks'), 
                State('data-sheet-path', 'data'),
                State('datasheet-file-name-input', 'value'),
                prevent_initial_call=True)
    def download_data_sheet(n_clicks, datasheet_path, datasheet_name):
        if n_clicks is None:
            raise PreventUpdate
        
        if datasheet_path is not None:

            with open(datasheet_path, 'r') as file:
                json_data = json.load(file)

            dirty_path = os.path.relpath(json_data['dirty_path'], '')   
            dirty_dataset_name = os.path.basename(os.path.dirname(dirty_path))

            with open(dirty_path, 'r') as file:
                dirty_data = file.read()

            if datasheet_name == '' or datasheet_name is None:
                datasheet_def_name = os.path.basename(datasheet_path)
                datasheet_downloadable = dict(content=json.dumps(json_data, indent=4), filename=datasheet_def_name)
                dirty_dataset_downloadable = dict(content=dirty_data, filename=f'{dirty_dataset_name}.csv')
                return html.P('Downloaded Data Sheet'), datasheet_downloadable, dirty_dataset_downloadable
            else:
                datasheet_downloadable = dict(content=json.dumps(json_data, indent=4), filename=f'{datasheet_name}.json')
                dirty_dataset_downloadable = dict(content=dirty_data, filename=f'{dirty_dataset_name}.csv')
                return html.P('Downloaded Data Sheet'), datasheet_downloadable, dirty_dataset_downloadable
        
        raise PreventUpdate              
