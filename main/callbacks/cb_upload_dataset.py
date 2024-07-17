import os
import pandas as pd
from main.apicalls import api_store_ds_info
from main.utils import parse_contents, DATA_PATH, MAX_LEN, PAGE_SIZE
from main.sqlconnection import switch_for_service

from deltalake import DeltaTable
from deltalake.writer import write_deltalake

from dash import dash_table, html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State


def get_cb_upload_dataset(app):
    
    # ============================ UPLOAD DATASET ======================================
    
    @app.callback([Output('memory-dataset', 'data', allow_duplicate=True),
                Output('memory-filename', 'data', allow_duplicate=True),
                Output('memory-filepath', 'data', allow_duplicate=True),
                Output('memory-error', 'data', allow_duplicate=True),
                Output('alert-message', 'data', allow_duplicate=True),
                Output('data-version-error-msg', 'children'),
                Output('upload-state', 'data', allow_duplicate=True)],
                Input('upload-button', 'n_clicks'),
                [State('data-version', 'value'),
                State('upload-data', 'contents'),
                State('upload-data', 'filename'),
                State('default-dataset', 'value')], prevent_initial_call=True)
    def upload_dataset(n_clicks, version_id, content, filename, default_dataset):
        """ Store uploaded dataset in a dcc.Store when user clicks the 'Upload' button """

        if n_clicks is None:
            raise PreventUpdate

        # first checks if user uploaded a file, then checks if a default dataset is selected
        if content is not None:
            df = parse_contents(content, filename)

            if ".csv" in filename:
                filename = filename[:-4]
            elif '.xlsx' in filename:
                filename = filename[:-5]

            path = os.path.abspath(os.path.join(DATA_PATH, filename, "dirty.csv"))

            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            # delete file containing detected errors from previous runs:
            # user has to perform detection first on uploaded dataset before errors can be repaired
            # prevent repair using errors detected from different version of dataset
            detections_path = os.path.abspath(os.path.join(DATA_PATH, filename, "detections.csv"))
            if os.path.exists(detections_path):
                os.remove(detections_path)

            #########################
            # deltalake

            deltalake_path = os.path.join(os.path.dirname(path), "deltalake")
            os.makedirs(deltalake_path, exist_ok=True)
            print(f'Version given: {version_id}')

            # we only track versions of uploaded datasets but not default datasets
            if version_id is None or version_id == "":
                write_deltalake(deltalake_path, df, mode="overwrite", overwrite_schema=True)
                dt = DeltaTable(deltalake_path)
                err_msg = None
            else:
                try:
                    dt = DeltaTable(deltalake_path, version=int(version_id))
                    print(f'Loading version {version_id}')
                    err_msg = None
                except:
                    # 2 possible exceptions here: invalid version, datatable doesn't exist
                    print('Loading version failed')
                    write_deltalake(deltalake_path, df, mode="overwrite", overwrite_schema=True)
                    dt = DeltaTable(deltalake_path)
                    err_msg = 'Loading version failed, using uploaded dataset instead'

                df = dt.to_pandas()

            print('----------------------------------')
            print(f'Dataset: {filename}, Version: {dt.version()}')
            print('----------------------------------')
            #########################

            df.to_csv(path, index=False, encoding="utf-8")  # save dataset to directory

            if api_store_ds_info(path, filename, df, dt.version()):
                return df.to_dict('records'), filename, path, None, None, err_msg, True

            else:
                return None, None, None, None, "Error uploading the dataset", None, False

        elif default_dataset is not None:
            path = os.path.abspath(os.path.join(DATA_PATH, default_dataset, "dirty.csv"))
            df = pd.read_csv(path, header="infer", encoding="utf-8", keep_default_na=False, low_memory=False, )

            if api_store_ds_info(path, default_dataset, df, None):
                return df.to_dict('records'), default_dataset, path, None, None, None, True

            else:
                return None, None, None, None, "Error storing the dataset.", None, False
        return None, None, None, None, "No dataset selected or uploaded.", None, False
    
    #================================== DISPLAY DATA =================================
    
    @app.callback(Output('output-data-upload', 'children'),
                Input('memory-dataset', 'data'), prevent_initial_call=True)
    def display_data(data):
        """ Display uploaded data as a DataTable on dashboard after user clicks the 'Upload' button """
        if data is None:
            raise PreventUpdate

        df = pd.DataFrame(data)
        page_count = int(len(df) / PAGE_SIZE) if len(df) < MAX_LEN else MAX_LEN / PAGE_SIZE

        return dash_table.DataTable(
            df.head(PAGE_SIZE).to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns],
            style_table={'overflowX': 'auto'},
            id='display-table',
            page_action='custom',
            page_size=PAGE_SIZE,
            page_count=page_count
        )
    
    # =============================== DISCARD FILE NAME ===============================
    
    @app.callback(
    [Output('upload-data', 'contents', allow_duplicate=True),
    Output('upload-data', 'filename', allow_duplicate=True),
    Output('file-name', 'children', allow_duplicate=True),
    Output('data-version', 'value')],
    Input('discard-data', 'n_clicks'), prevent_initial_call=True
    )
    def discard_file(n_clicks):
        return None, None, None, None


    # ================================ DISPLAY FILE NAME ===============================
    
    @app.callback(Output('file-name', 'children'),
              Input('upload-data', 'filename'), prevent_initial_call=True)
    def display_file_name(filename):
        """ After a file is uploaded, the file name is displayed below the upload box """
        return filename

    # =============================== ADD UILTAG =======================================

    @app.callback(Output('tag-container', 'children'),
              Output('tag-input', 'value'),
              Output('memory-uiltags', 'data'),
              Input('tag-input', 'n_submit'),
              State('tag-container', 'children'),
              State('tag-input', 'value'),
              State('memory-uiltags', 'data'), prevent_initial_call=True)
    def add_uiltag(n_submit, existing_tags, tag_value, data_existing_tags):
        if data_existing_tags is None:
            data_existing_tags = []
        if n_submit > 0 and tag_value:
            data_existing_tags.append(tag_value)
            print("Tage: ", data_existing_tags)
            new_tags = existing_tags + [html.Span(tag_value, className='tag', style={'margin': '0 5px 5px 0', 'border': '1px solid red', 'border-radius': '5px', 'padding': '1px 2px', 'color': 'white', 'background-color': 'red', 'line-height': '1.2'})]
            return new_tags, '', data_existing_tags
        else:
            return existing_tags, tag_value, data_existing_tags


    # =============================== CONNECT DATABASE ==================================
    
    @app.callback(Output('memory-dataset', 'data'),
              Output('memory-filename', 'data'),
              Output('memory-filepath', 'data'),
              Output('memory-error', 'data'),
              Output('alert-message', 'data', allow_duplicate=True),
              Output('upload-state', 'data', allow_duplicate=True),
              Input('connect-button', 'n_clicks'),
              Input('server-dropdown', 'value'),
              State('server-input', 'value'),
              State('database-input', 'value'),
              State('table-input', 'value'),
              State('uid-input', 'value'),
              State('password-input', 'value'),
              State('port-input', 'value'), prevent_initial_call=True)
    def connect_database(n_clicks, dropdownselect, servername, databasename, tablename, username, password, port):
        """ Loads a table from a specified dataset from different database systems """

        if n_clicks is None:
            raise PreventUpdate

        # checks if something was entered into the input fields
        if [x for x in (servername, databasename, tablename, username) if x is None]:
            return None, None, None, None, 'Please enter all required server details and credentials!', False

        # connects to the SQL database and retrieves the table
        df, errormessage = switch_for_service(dropdownselect, servername, databasename, tablename, username, password, port)

        # checks for error in connection, if there is none it proceeds
        if errormessage is None:
            # as in upload_dataset
            path = os.path.abspath(os.path.join(DATA_PATH, tablename, "dirty.csv"))

            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            # delete file containing detected errors from previous runs:
            # user has to perform detection first on uploaded dataset before errors can be repaired
            # prevent repair using errors detected from different version of dataset
            detections_path = os.path.abspath(os.path.join(DATA_PATH, tablename, "detections.csv"))
            if os.path.exists(detections_path):
                os.remove(detections_path)

            df.to_csv(path, index=False, encoding="utf-8")

            if api_store_ds_info(path, tablename, df, None):
                return df.to_dict('records'), tablename, path, None, None, True

            else:
                return None, None, None, None, "Error: Could not store dataset information", False

        return None, None, None, None, errormessage, False
