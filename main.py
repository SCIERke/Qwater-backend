from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

@app.get("/")
def hello_world():
    return { "msg" : "ok!"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    print("Filename:",file.filename)
    return {"filename": file.filename}