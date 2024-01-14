import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns 


data = pd.read_csv('df_test.csv')

data.drop(columns=['Unnamed: 0'], inplace=True)

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'id_client': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json, verify=False)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()




def main():
    api_pred = f"https://projet7-obzi.onrender.com/predict"
    api_interpre = f"https://projet7-obzi.onrender.com/interpretability"


    st.title('Bank credit Prediction')
    
    client_choice = st.selectbox(
        'client id',
        data.SK_ID_CURR.tolist())
    
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    def click_button():
        st.session_state.clicked = True
    
    predict_btn = st.button('Prédire', on_click=click_button)
    if st.session_state.clicked:
        pred = request_prediction(api_pred, client_choice)
        if pred['classe'] == 0:
            st.write(
            "The credit is accepted")
            
        else:
            st.write(
            "The credit is not accepted")
            
        fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred['probabilité'] * 100,
        title={'text': "Client score"},
        domain={'x': [0, 1], 'y': [0, 1]}
        ))
        fig.update_layout(yaxis={'range': [0, 100]})
        st.plotly_chart(fig)
            
        
        response = requests.request(method='POST', 
                                   headers={"Content-Type": "application/json"}, 
                                   url=api_interpre, 
                                   json={"id_client" : client_choice},
                                   stream=True
                                   )
        with open(str(client_choice)+'.png', 'wb') as outputfile : 
            outputfile.write(response.content)
        
        
        st.image(str(client_choice)+'.png')
    
        var = st.sidebar.multiselect(
            'select variables',
            data.columns)
        
        index = data[data.SK_ID_CURR == client_choice].index
        
        #client2 = data.loc[index, var2][index[0]]

        for i in range(len(data[var].columns)):
            client = data.loc[index, var[i]][index[0]]
            fig1 = px.histogram(data[var[i]])
            fig1.add_vline(x=client, annotation_text="client")
            st.plotly_chart(fig1)
    

if __name__ == '__main__':
    main()
