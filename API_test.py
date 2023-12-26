from fastapi.testclient import TestClient
from fastapi import status
from API_app import app

client=TestClient(app=app)

def test_predict_correct():
    response = client.post('/predict', json={'id_client' : 100001})
    
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {'classe': 0.0, 'probabilité': 0.9860117143850826}


def test_correct():
    response = client.get('/')
    
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {'message': 'Home Credit Default Risk'}
    
  
def test_FeatImp_correct():
    response = client.post('/interpretability', json={'id_client' : 100001})
    
    assert response.status_code == status.HTTP_200_OK
