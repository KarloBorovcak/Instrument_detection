from fastapi import FastAPI, File, UploadFile
from prediction import predict

app = FastAPI()

@app.post("/predict-instruments")
async def create_upload_file(file: UploadFile = File(...)):
    """
    Predict the instruments present in an audio file.

    - **file**: An audio file in WAV format.

    Returns a JSON response with the predicted instruments.
    """

    if file.content_type[:6] != "audio/":
        return {"error": "File must be a WAV file"}
    
    prediction = predict(file.file)

    return prediction
