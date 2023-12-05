import streamlit as st
import st_pages
from st_pages import show_pages_from_config, add_page_title
import pandas as pd
import numpy as np
import pickle
from urllib.request import urlopen
import json
# from PIL import Image
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


#  pour les raisons et les modalités https://docs.python.org/fr/3/library/warnings.html

def main():
    
        st.title("Make sure a client is a sure client!") 

if __name__ == '__main__':
    
    main()

show_pages_from_config (".streamlit/pages.toml")

# Create an instance of SessionState


# Main sections
header = st.container()
dataset= st.container()
# features = st.container()

model_training = st.container()
model = pickle.load(open('./Datas/model.pkl','rb'))

# #Load Dataframe
dataframe=pd.read_csv('./Datas/dFreduced.csv')


def credit(id_client):
    
    ID = int(id_client)   
    
# récupération des données clients
    X = dataframe[dataframe['SK_ID_CURR'] == ID]
    X = X.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    
    prediction = model.predict(X)
    
    y_probabiliste = model.predict_proba(X)
    
    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(y_probabiliste[0][0])
        }

    print('Nouvelle Prédiction : \n', dict_final)


### Sidebar
st.sidebar.title("Menu")
sidebar_selection = st.sidebar.radio(
    'Select your activity :',
    ['Overview', 'Model & Prediction','Explication and Comparison'],
    )


with header:
    st.title("Make sure a client is a sure client!") 
    

    
with dataset:
    st.header("The train dataset is made of a selection of relevant features chosen after EDA")
    credit_data = dataframe
    st.write (credit_data.head())



st.subheader("Caractéristiques influençant le score")
          
          
#MLFLOW tracking    
# Set the experiment
# Mlflow tracking

    # track_with_mlflow = st.checkbox(
    #     "📈 Track with mlflow? ", help="Mark to track experiment with MLflow"
    # )

#     # Model training
#     start_training = st.button("💪 Start training", help="Train and evaluate ML model")
#     if not start_training:
#         st.stop()

#     if track_with_mlflow:
#         mlflow.set_experiment(data_choice)
#         mlflow.start_run()
#         mlflow.log_param("model", model_choice)
#         mlflow.log_param("features", feature_choice)


# mlflow.set_experiment("optimized_RF_Classifier")

# # Log a metric
# accuracy = 0.9
# mlflow.log_metric("accuracy", accuracy)

# # Log an artifact
# model = pickle.dumps(my_model)
# mlflow.log_artifact("model", model)

# # Display the metrics and artifacts
# st.write("Accuracy:", accuracy)
# st.write("Model:", model)
    
    