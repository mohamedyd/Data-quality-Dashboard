import dash_bootstrap_components as dbc
from dash import dash_table, html, callback_context
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from main.utils import PROCESSED_PATH
from main.apicalls import api_get_fd


def get_cb_upload_state_management(app):
    
    # ============================= DISABLE/ENABLE BUTTONS ========================================
    
    @app.callback([Output('detect-button', 'disabled'),
                 Output('repair-button', 'disabled'),
                 Output('iterative-clean-button', 'disabled'),
                 Output('generate-data-sheet', 'disabled'),],
                 [Input('upload-state', 'data')])
    def update_buttons(upload_state):
        if upload_state:
            # Disable the buttons if the upload state is True
            return True, True, True, True
        else:
            # Enable the buttons otherwise
            return False, False, False, False
        
    # ================================ DISABLE RAHA SUBMIT/CANCEL BUTTON =================================
    
    @app.callback(Output('submit-button', 'disabled'),
                Input('user-labeling', 'children'))
    def disable_submit_button(t):
        """ Disables the button for submitting labels if no tuple is showed """
        if t is None:
            return True
        
    @app.callback(Output('cancel-raha-button', 'disabled'),
                  Input('raha-signal', 'data'))
    def disable_cancel_button(signal):
        """ Disables the button for canceling if it is not active """
        if signal is None:
            return True
        else:
            return False
        
    # ================================ SHOW LOADING SPINNER =================================
        
    @app.callback([Output('upload-button', 'children'),
                   Output('upload-button', 'disabled')],
                  [Input('upload-state', 'data')], prevent_initial_call=True)
    def update_upload_button(upload_state):
        if upload_state:
            # If upload_state is True, show the spinner and the text "Loading..."
            return [dbc.Spinner(size="sm"), " Uploading..."], True
        else:
            # If upload_state is False, show the text "Upload"
            return "Upload", False
        
    # ================================ DISABLE DISCARD BUTTON =================================
    
    @app.callback(Output('discard-data', 'disabled'),
                  Input('upload-data', 'filename'))
    def update_button(filename):
        if filename is None:
            return True
        else:
            return False