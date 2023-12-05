import pandas as pd
import streamlit as st
import pickle
import plotly.graph_objects as go
from sklearn.neighbors import NearestNeighbors




# if "my_client" not in st.session_state:
#     st.session_state ["my_client"]=chosen_client

fulldata=pd.read_csv('./Datas/df_accepted_for_app.csv')

st.write (st.session_state.chosen_client)

model_selection_column, display_column = st.columns(2)


     # Choose a client
    # chosen_client = st.text_input(
    #     "Enter some text 👇",
    #             disabled=st.session_state.disabled,
    #     placeholder=st.session_state.placeholder,
    # )


client = int(st.session_state.chosen_client)
X = fulldata[fulldata['SK_ID_CURR'] == client]

st.write(X)

st.write(fulldata)

from sklearn.neighbors import NearestNeighbors

def nearestneighbors (data, chosen_client, k):
    X = data[data['SK_ID_CURR'] == chosen_client]
    voisins = NearestNeighbors(n_neighbors=k)
    voisins.fit(data)
    distances, indexes = voisins.kneighbors(X)
    df_neighbors = data.loc[indexes[0]].T
    return df_neighbors


st.write (nearestneighbors(fulldata, client, 4).head(20))

