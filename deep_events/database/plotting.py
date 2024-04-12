#%% imports
from convenience import get_collection
from construct import reconstruct_from_folder
from sys import platform
import pandas as pd

from pathlib import Path

collection = 'mito_ideas_models'
if platform == "linux" or platform == "linux2":
    folder = Path("/mnt/deep_events/data/original_data/training_data")
else:
    folder = Path("//lebnas1.epfl.ch/microsc125/deep_events/data/original_data/training_data")

#%% Reconstruct the model database from the training_data folder
# reconstruct_from_folder(folder, collection)

#%%
collection = get_collection(collection)

data = pd.DataFrame(list(collection.find()))
data.drop("brightness_range", axis=1, inplace=True)
data.drop('_id', axis=1, inplace=True)

event_collection = get_collection("mito_events")
event_data = pd.DataFrame(list(event_collection.find()))
# %%
import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import altair as alt
import numpy as np



cols = st.columns((1, 2))
cols[1].title("Model performance")

# Left column
c = np.empty(1, object)
c[0] = ['brightfield']
bf_events = event_data[event_data.contrast.values == c]
text = f"""
## Current data\n
    number of models: {len(data)}
    number of events: {len(event_data)}
    brightfield events: {len(bf_events)}"""

cols[0].markdown(text)
cols[0].write(data)

## Main filtering step
query = cols[1].text_input('Main filtering (_applies to all below_)',
                           value="data_corruption != True and contrast == 'brightfield'")
data['n_timepoints'] = data['n_timepoints'].astype(str)
for index, row in data.iterrows():
    data.loc[index, 'date'] = pd.to_datetime(row['time'][:8])
color_by_list = sorted(data.columns)
color_by = cols[1].selectbox('Color by', color_by_list, index=0)
color_by = alt.value("white") if color_by == "None" else color_by
color_by = alt.Color('date:T', scale=alt.Scale(
    scheme=alt.SchemeParams(name='yellowgreenblue', extent=[-1, 2])
                    )) if color_by == "date" else color_by

data0 = data.copy()
if query:
    data0 = data0.query(query)
c0 = alt.Chart(data0).mark_circle(size=80).encode(
    x='frames',
    y='p_f1',
    color=color_by,
    tooltip=['frames', 'p_f1', 'n_timepoints', 'fps', 'time', 'contrast', 'n_event']
)
cols[1].altair_chart(c0, use_container_width=True)


subquery = cols[1].text_input('Subquery', value="fps == 1")
axis_cols = cols[1].columns(2)
x_axis = axis_cols[0].selectbox('X axis', ['frames', 'p_f1', 'p_tpr', 'p_precision', 'n_event'], index=0)
y_axis = axis_cols[1].selectbox('Y axis', ['p_f1', 'p_tpr', 'p_precision'], index=0)
data1 = data0.copy()
if subquery:
    data1 = data1.query(subquery)

c1 = alt.Chart(data1).mark_circle(size=80).encode(
    x=x_axis,
    y=y_axis,
    color=color_by,
    tooltip=['frames', 'p_f1', 'n_timepoints', 'fps', 'time']
)
cols[1].altair_chart(c1, use_container_width=True)


# data2 = data1.query("fps == 1 and (date > '2024-01-13' or n_timepoints == "3.0") and (frames > 1300 and frames < 4800)

# For tooltips in fullscreen mode
st.markdown('<style>#vg-tooltip-element{z-index: 1000051}</style>',
             unsafe_allow_html=True)

# %%


def dark_boxplot():
    #TODO: put this into something recallable
    import matplotlib.pyplot as plt
    data_noneg = [1,2,3]
    data_neg = [2,3,4]
    boxprops = dict(color='#999999', linewidth=3)
    flierprops = dict(marker='o', markerfacecolor='#999999')
    medianprops = dict(linestyle='-', color='r', linewidth=3)
    meanlineprops = dict(linestyle='-', linewidth=2.5, color='purple')
    whiskerprops = dict(color="#999999")
    capprops = dict(color="#999999")
    plt.boxplot([data_noneg, data_neg],capprops=capprops, boxprops=boxprops, flierprops=flierprops, medianprops=medianprops, meanprops=meanlineprops, whiskerprops=whiskerprops)
    plt.ylabel("F1",fontdict={'size': 24})
    plt.ylim(0.4, 1)
    plt.yticks(fontdict={'size': 18})
    plt.xticks(ticks=[1,2], labels=["pre", "post"], fontdict={'size': 18})
    plt.xlabel("Negative events in training data", fontdict={'size': 24, 'weight': 'normal'})