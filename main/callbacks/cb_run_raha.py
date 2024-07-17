import pandas as pd

import dash
from dash import html, dash_table
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State

from main.utils import default_datasets
from main.apicalls import api_cancel_raha, api_send_user_labels, api_get_raha_detection

def get_cb_raha(app):
    
    # ============================= RAHA SAMPLE TUPLE ========================================

    @app.callback(Output('user-labeling', 'children'),
                Input('raha-signal', 'data'),
                State('raha-tuple', 'data'), prevent_initial_call=True)
    def raha_sample_tuple(signal, raha_tuple):
        """ Raha samples a tuple and displays it on the dashboard """

        if signal is None:
            return None

        if signal:  # sampled tuple
            df = pd.DataFrame.from_dict(raha_tuple)

            # display tuple on dashboard
            return dash_table.DataTable(
                df.to_dict('records'),
                [{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                id='tuple-labeling')
        else:
            raise PreventUpdate

    # ============================= TUPLES LEFT TO LABLE ======================================

    @app.callback(Output('tuples-left-to-label', 'children'),
                Input('raha-signal', 'data'),
                State('tuples-left', 'data'), prevent_initial_call=True)
    def tuples_left_to_label(signal, tuples_left):
        """ Tells user how many tuples left to label according to Raha's labeling budget"""

        if signal:
            return html.P(f'{tuples_left} tuples left to label')
        else:
            return None


    # ============================ SELECTED CELLS ==============================================

    @app.callback(Output('selected-cells', 'children'),
                Input('user-label', 'data'), prevent_initial_call=True)
    def selected_cells(user_labels):
        """ Tells user which cells are currently selected """

        if user_labels is not None:
            s = [f"({u['row']}, {u['column_id']})" for u in user_labels]
            return html.P(f'Selected cells (row, column): {", ".join(s)}')
        else:
            return None

    # =========================== RAHA USER LABEL ==============================================

    @app.callback(Output('user-label', 'data'),
                  Output('tuple-labeling', 'active_cell'),
                  Input('tuple-labeling', 'active_cell'),
                  State('user-label', 'data'), prevent_initial_call=True)
    def raha_user_label(active_cell, user_labels):
        """ Stores dirty cells labeled by user """

        if active_cell is None:
            raise PreventUpdate

        if user_labels is None:
            return [active_cell], None
        else:
            if active_cell in user_labels:
                user_labels.remove(active_cell)
            else:
                user_labels.append(active_cell)
            return user_labels, None

    # ============================== RAHA USER LABEL STYLE ======================================

    @app.callback(Output('tuple-labeling', 'style_data_conditional'),
                Input('user-label', 'data'), prevent_initial_call=True)
    def raha_user_label_style(user_labels):
        """ Marks cells if clicked by user (red cells are cells labeled by user as dirty) """

        style_data = []  # list of dictionaries specifying the style for each cell
        if user_labels is not None:
            for active_cell in user_labels:
                style_data.append({'if': {'row_index': active_cell['row'], 'column_id': active_cell['column_id']},
                                'backgroundColor': '#FF0000', 'color': 'white'})

            return style_data
        else:
            raise PreventUpdate

    # =============================== RAHA SUBMIT USER LABEL =====================================

    @app.callback(Output('raha-signal', 'data', allow_duplicate=True),
                Output('user-labeling', 'children', allow_duplicate=True),
                Output('user-label', 'data', allow_duplicate=True),
                Output('raha-tuple', 'data', allow_duplicate=True),
                Output('tuples-left', 'data', allow_duplicate=True),
                Input('submit-button', 'n_clicks'),
                State('user-label', 'data'), prevent_initial_call=True)
    def raha_submit_user_label(n_clicks, user_label):
        """ Submit user labels to raha after user clicks "Submit" button """

        raha_tuple, tuples_left = api_send_user_labels(user_label)

        # check if we still want to sample more tuples
        if tuples_left > 0:
            return True, dash.no_update, None, raha_tuple, tuples_left
        else:
            return False, None, None, None, None

    # ================================== RAHA DETECTION =============================================

    @app.callback(Output('memory-error', 'data', allow_duplicate=True),
                  Output('ic-signal', 'data'),
                Input('raha-signal', 'data'),
                State('memory-error', 'data'),
                State('memory-dataset', 'data'),
                State('memory-filename', 'data'), prevent_initial_call=True)
    def raha_detection(signal, error, dataset, filename):
        """ After user labels all required tuples, raha proceeds to detect errors """

        if signal is None or signal:
            raise PreventUpdate

        # receive terminate signal
        if not signal:
            p, r, f, detection_dict = api_get_raha_detection()

            # evaluate user labeling for datasets with ground truth
            if filename in default_datasets:
                print(f"Detection finished, Results: \n Precision: {p:.3f}, Recall: {r:.3f}, F1 Score: {f:.3f}")

            if error is None:
                error = {}
            error['Raha'] = list(detection_dict.keys())

            return error, True

    # =============================== CANCEL RAHA ============================================

    @app.callback(Output('raha-signal', 'data', allow_duplicate=True),
                Output('user-label', 'data', allow_duplicate=True),
                Input('cancel-raha-button', 'n_clicks'), prevent_initial_call=True)
    def cancel_raha(n_clicks):
        """ Cancel Raha process """
        api_cancel_raha()
        return None, None
    
    # =============================== CANCEL RAHA ============================================
    
    @app.callback(Output('submit-button', 'children'),
                  Input('user-label', 'data'), prevent_initial_call=True)
    def change_submit_button_text(user_labels):
        if user_labels is not None:
            return "Submit"
        return "Skip"