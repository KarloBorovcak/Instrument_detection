from flask import Flask, request
from prediction import predict

app = Flask(__name__)

@app.route('/audio', methods=['GET', 'POST'])
def get_audio():
    if request.method == 'POST':

        # Get the file from post request
        if request.content_type == 'audio/wave':
            f = request.get_data()

        elif request.content_type[:19] == 'multipart/form-data':
            f = request.files['file']
        
        else:
            return 'Invalid content type'


        response = predict(f)

        return response


if __name__ == '__main__':
    app.run()