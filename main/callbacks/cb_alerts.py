from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output


def get_cb_alerts(app):

    # ================ POST ALERT =========================================

    @app.callback(Output('alert-dialog', 'displayed'),
                Output('alert-dialog', 'message'),
                Input('alert-message', 'data'), prevent_initial_call=True
                )
    def post_alert(message):
        if message is None:
            raise PreventUpdate
        
        return True, message

# ================= RESET ALERT ===========================================

    @app.callback(Output('alert-dialog', 'submit_n_clicks'),
                Output('alert-dialog', 'cancel_n_clicks'),
                Input('alert-dialog', 'submit_n_clicks'),
                Input('alert-dialog', 'cancel_n_clicks'), prevent_initial_call=True
                )
    def reset_alert(submit, cancel):
        return None, None
