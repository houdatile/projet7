# API

# Importing Necessary modules
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse, StreamingResponse
import io
#%matplotlib inline
 

app = FastAPI()

data = pd.read_csv("df_test.csv")

data.drop(columns=['Unnamed: 0'], inplace=True)

model = joblib.load('best_model.joblib')

model_fi = joblib.load('feature_importance.joblib')
shap_values = model_fi(data)

class request_body(BaseModel):
    id_client : int
    
 
# Defining path operation for root endpoint
@app.get('/')
def main():
    return {'message': 'Home Credit Default Risk'}
 
# Defining path operation for /name endpoint
@app.get('/{name}')
def hello_name(name : str): 
    # Defining a function that takes only string as input and output the
    # following message. 
    return {'message': f'Welcome {name}'}

@app.post('/predict')
def predict(id_client : request_body):
    id_client = id_client.id_client
    
    pred = model.predict(data[data.SK_ID_CURR == id_client])[0]
    proba = model.predict_proba(data[data.SK_ID_CURR == id_client])
    return {
        "classe"  : pred,
        "probabilité" : proba[0, int(pred)]
        }
    
    
@app.post('/interpretability')
async def interpretability(id_client : request_body):
    id_client = id_client.id_client
    
    
    fig = plt.figure()
    #fig.set_size_inches(5, 5)
    fig = shap.plots.waterfall(shap_values[data[data.SK_ID_CURR == id_client].index[0]], show=False)
    w, _ = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(w*(5/3), w)
    plt.tight_layout()
    #plt.savefig(str(id_client)+'.png')
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png")   
    #return FileResponse(str(id_client)+'.png')
    buf.seek(0)
        
    return StreamingResponse(buf, media_type="image/png")
    





