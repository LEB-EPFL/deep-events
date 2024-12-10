#%% imports
from convenience import get_collection
from construct import reconstruct_from_folder
from sys import platform
import pandas as pd
import yaml

from pathlib import Path

collection = 'mito_ideas_models'
if platform == "linux" or platform == "linux2":
    main_folder = Path("/mnt/LEB/Scientific_projects/deep_events_WS/data/original_data")
else:
    main_folder = Path("//sb-nas1.rcp.epfl.ch/LEB/Scientific_projects/deep_events_WS/data/original_data")

#%% Reconstruct the model database from the training_data folder
folder = main_folder / 'training_data'
with open(folder / "collection.yaml", "r") as f:
    collection = yaml.safe_load(f)["collection"]
# reconstruct_from_folder(folder, collection)

#%%
if isinstance(collection, str):
    collection = get_collection(collection)

data = pd.DataFrame(list(collection.find()))
print(data)
data.drop("brightness_range", axis=1, inplace=True)
data.drop('_id', axis=1, inplace=True)

event_collection = get_collection("ld_events")
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
                           value="collection == 'ld_events'")
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
    y='n_event',
    color=color_by,
    tooltip=['frames', 'p_f1', 'n_timepoints', 'fps', 'time', 'contrast', 'n_event']
)
cols[1].altair_chart(c0, use_container_width=True)


subquery = cols[1].text_input('Subquery', value="fps == 1")
axis_cols = cols[1].columns(2)
x_axis = axis_cols[0].selectbox('X axis', ['frames', 'f1', 'mcc', 'n_event'],
                                index=0)
y_axis = axis_cols[1].selectbox('Y axis', ['f1', 'precision', 'recall', 'mcc', 'kappa', 'w_mcc'],
                                index=0)
data1 = data0.copy()
if subquery:
    data1 = data1.query(subquery)

def get_nested_value_y(row, y_axis=y_axis):
    try:
        value = row['performance'].get(y_axis, 0) 
        return value
    except (KeyError, TypeError, AttributeError) as e:
        return 0.0

def get_nested_value_x(row, x_axis=x_axis):
    try:
        value = row['performance'].get(x_axis, 0) 
        return value
    except (KeyError, TypeError, AttributeError) as e:
        return 0.0

try:
    data1['y_axis'] = data1[y_axis]
except:
    data1['y_axis'] = data1.apply(get_nested_value_y, axis=1)

try:
    data1['x_axis'] = data1[x_axis]
except:
    data1['x_axis'] = data1.apply(get_nested_value_x, axis=1)
data1 = data1.convert_dtypes()

c1 = alt.Chart(data1).mark_circle(size=80).encode(
    x='x_axis:Q',
    y='y_axis:Q',
    color=color_by,
    tooltip=['frames', 'n_timepoints', 'fps', 'date']
)
cols[1].altair_chart(c1, use_container_width=True)


# data2 = data1.query("fps == 1 and (date > '2024-01-13' or n_timepoints == "3.0") and (frames > 1300 and frames < 4800)

# For tooltips in fullscreen mode
st.markdown(
    """
    <style>
    #vg-tooltip-element {
        z-index: 1000051 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
