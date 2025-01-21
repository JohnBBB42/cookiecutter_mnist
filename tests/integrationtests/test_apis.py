from fastapi.testclient import TestClient
from mnist_project.app import app

client = TestClient(app)

# add model
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the MNIST model inference API!"}

# for lifespan
#def test_read_root():
    #with TestClient(app) as client:
        #response = client.get("/")
        #assert response.status_code == 200
        #assert response.json() == {"message": "Welcome to the MNIST model inference API!"}
