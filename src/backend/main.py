import base64
import os.path
import uuid
from http.client import responses

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from . import modelAi

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.exists("xray_model.h5"):
   print("model already exists, using the saved model")
   #modelAi.runAi()#file_Location)
else:
    print("model does not exist, training the model...")
    modelAi.trainModel()

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/api/uploadFile",
          responses = {
              200: {
                  "content": {"image/png": {}}
              }
          },
          response_class=Response
          )
async def upload_file(file: UploadFile):
    if not file:
        return {"message": "File not uploaded"}

    file_Location = f"C:\\Users\\guilherme.zografos\\PycharmProjects\\FastAPIProject\\temp\\{uuid.uuid4()}.png"#{file.filename}"
    with open(file_Location, "wb+") as file_object:
        file_object.write(file.file.read())

    result = modelAi.runAi(file_Location)


    #####
    #
    #   TODO: upload image to aws S3 and build the AWS side of the project.
    #   TODO: Return the image through the API response so that it can be rendered on the frontend.
    #
    #####

    #image_bytes: bytes = file.file
    return {
        "message": "success",
        "results": {
            "class": "test",
            "file": "a",#Response(content=image_bytes, media_type="image/png"),
            "boneAge": 10
        }
    }

