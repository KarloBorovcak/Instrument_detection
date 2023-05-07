from typing import List
from fastapi import FastAPI, File, UploadFile
from predict import predict

app = FastAPI()

@app.post("/upload-file")
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

@app.post("/upload-files")
async def predict_instruments(files: List[UploadFile] = File(...)):
    """
    Predict the instruments present in one or more audio files.

    - **files**: A list of audio files in WAV format.

    Returns a JSON response where keys are names of the file, with each response containing 
    the predicted instruments for a single audio file.
    """

    predictions = {}
    for file in files:
        if file.content_type[:6] != "audio/":
            return {"error": "File must be a WAV file"}

        prediction = predict(file.file)
        predictions[file.filename[:-4]] = prediction

    return predictions
