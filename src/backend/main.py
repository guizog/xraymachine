from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow React frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
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
    debugToggle = True
    if not file:
        return {"message": "File not uploaded"}

    if debugToggle: ##info for debug and dev purposes
        return {
            "results": {
                "file": "temp",
                "boneAge": 10
            }
        }

    print(file)
    return {"message": f"File uploaded successfully. Filename: {file.filename}"}
