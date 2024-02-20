#%% imports
from convenience import get_collection
from construct import reconstruct_from_folder

import pandas as pd

from pathlib import Path

collection = ('mito_ideas_models')
folder = Path("//lebnas1.epfl.ch/microsc125/deep_events/data/original_data/training_data")

#%% Reconstruct the model database from the training_data folder
# reconstruct_from_folder(folder, collection)

#%%
collection = get_collection('mito_ideas_models')
print(collection)

data = pd.DataFrame(list(collection.find()))
# %%
# import plotly.graph_objects as go
# fig = go.Figure()
from dash import Dash, dcc, html

app = Dash(__name__)
app.layout = html.Div([["Hello world"]])
if __name__ == '__main__':
    app.run_server(debug=False)
# %%
