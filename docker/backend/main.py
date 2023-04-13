from fastapi import FastAPI, File, UploadFile
from prediction import predict

app = FastAPI()


@app.get('/')
async def root():
    return {"message": "Hello World"}

@app.post("/audio")
async def create_upload_file(file: UploadFile = File(...)):
    if file.content_type[:6] != "audio/":
        return {"error": "File must be a WAV file"}
    
    prediction = predict(file.file)

    return prediction
