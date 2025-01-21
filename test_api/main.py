import re
from enum import Enum
from http import HTTPStatus

import anyio
import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response

class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/restric_items/{item_id}")
def read_item(item_id: ItemEnum):
    return {"item_id": item_id}

@app.get("/query_items")
def read_item(item_id: int):
    return {"item_id": item_id}

database = {'username': [ ], 'password': [ ]}

@app.post("/login/")
def login(username: str, password: str):
    username_db = database['username']
    password_db = database['password']
    if username not in username_db and password not in password_db:
        with open('database.csv', "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"

@app.get("/text_model/")
def contains_email(data: str):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data) is not None
    }
    return response


class DomainEnum(Enum):
    """Domain enum."""

    gmail = "gmail"
    hotmail = "hotmail"


class Item(BaseModel):
    """Item model."""

    email: str
    domain: DomainEnum


@app.post("/text_model/")
def contains_email_domain(data: Item):
    """Simple function to check if an email is valid."""
    if data.domain is DomainEnum.gmail:
        regex = r"\b[A-Za-z0-9._%+-]+@gmail+\.[A-Z|a-z]{2,}\b"
    if data.domain is DomainEnum.hotmail:
        regex = r"\b[A-Za-z0-9._%+-]+@hotmail+\.[A-Z|a-z]{2,}\b"
    return {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data.email) is not None,
    }

@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: Optional[int] = 28, w: Optional[int] = 28):
    # Save uploaded file synchronously to ensure it's written before reading
    content = await data.read()
    with open("image.jpg", "wb") as image_file:
        image_file.write(content)

    # Read the saved image using OpenCV
    img = cv2.imread("image.jpg")
    if img is None:
        return {"error": "Failed to load the image. Please check the file and try again."}

    # Resize the image
    res = cv2.resize(img, (w, h))
    cv2.imwrite("image_resize.jpg", res)

    return {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "output": FileResponse("image_resize.jpg")
    }
