import pandas as pd
import streamlit as st
import pickle
from urllib.request import urlopen
import json
import plotly.graph_objects as go

# load our best model
# PATH = "C:\Users\DELL\Formation OC\API_P7\Datas\"

#Load Dataframe


# X_test=pd.read_csv(PATH+'X_test.csv')
# y_test=pd.read_csv(PATH+'y_test.csv')
dataframe=pd.read_csv('./Datas/dFreduced.csv')


# Main sections
header = st.container()
dataset= st.container()
features = st.container()
model_training = st.container()

model = pickle.load(open('./Datas/model.pkl','rb'))
with header:
    st.title("Make sure a client is a sure client!") 
    

    
with dataset:
   
    st.header("The train dataset is made of a selection of relevant features chosen after EDA")
    credit_data = dataframe
    st.write (credit_data.head())


with model_training:
         
    model_selection_column, display_column = st.columns(2)
    test_clients = pd.read_csv('./Datas/dFreduced.csv')
    liste_id = test_clients['SK_ID_CURR'].tolist()
    
    

    if "chosen_client" not in st.session_state:
        st.session_state.chosen_client = None

    #  Choose a client
    # st.session_state.chosen_client = st.text_input(
    #     "Enter some text 👇",
    #             disabled=st.session_state.disabled,
    #     placeholder=st.session_state.placeholder,
    # )
  
    # 
    st.session_state.chosen_client = str(model_selection_column.selectbox("Please chose your client ID", test_clients['SK_ID_CURR']))
    st.session_state.chosen_client = int(st.session_state.chosen_client)



    if "my_client" not in st.session_state:
        st.session_state ["my_client"]=st.session_state.chosen_client


    
    
    # if chosen_client == '':

    #     st.write('Veuillez recommencer')

    if int(st.session_state.chosen_client) not in liste_id:

        st.write('Veuillez recommencer')
        
    elif (int(st.session_state.chosen_client) in liste_id) :

        st.success("client chosen")
        
        with st.spinner('Attente du score du client choisi ...'):   
            
            ID = int(st.session_state.chosen_client)   
        
            # récupération des données clients
            X = dataframe[dataframe['SK_ID_CURR'] == ID]
            X = X.drop(['TARGET', 'SK_ID_CURR'], axis=1)
            
            prediction = model.predict(X)
            
            y_probabiliste = model.predict_proba(X)
            
            dict_final = {
                'prediction' : int(prediction),
                'proba' : float(y_probabiliste[0][0])
                }
            if dict_final["prediction"] == 1:
                resultat = "client dangereux"
            else:
                resultat = "client sûr"
            
            
            print('Nouvelle Prédiction : \n', dict_final) 


                
            chaine = 'Prédiction : **' + resultat +  '** avec **' + str(round((1-dict_final['proba'])*100)) + '%** de risque d''erreur '
            
            st.markdown(chaine)

        fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = dict_final['proba']*100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                delta = {'reference': 100},
    
                title = {'text': "Confidence"}))

        fig.update_layout(paper_bgcolor='white',
                                height=400, width=600,
                                font={'color': 'darkblue', 'family': 'Arial'})

        st.plotly_chart(fig, use_container_width=True)


        #   classe_predite = API_data["prediction"]
        #   if classe_predite == 1:
        #       resultat = "client dangereux"
        #   else:
        #       resultat = "client peu risqué"
          
        #   proba = 1- API_data["proba"]
          
        #   #affichage de la prédiction
        #   prediction = API_data['proba']
        #   # classe_reelle = dataframe[dataframe['SK_ID_CURR']==int(chosen_client)]['LABELS'].values[0]
        #   # classe_reelle = str(classe_reelle).replace('0', 'sans défaut').replace('1', 'avec défaut')
        #   chaine = 'Prédiction : **' + resultat +  '** avec **' + str(round(proba*100)) + '%** de risque d''erreur '

        # st.markdown(chaine)