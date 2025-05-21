import base64

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/api/uploadFile")
async def upload_file(file: UploadFile):
    if not file:
        return {"message": "File not uploaded"}

    #####
    #
    #   TODO: upload image to aws S3 and build the AWS side of the project.
    #   TODO: Return the image through the API response so that it can be rendered on the frontend.
    #
    #####

    image_bytes: bytes = file.file;
    return {
        "message": "success",
        "results": {
            "file": "a",
            "boneAge": 10
        }
    }

