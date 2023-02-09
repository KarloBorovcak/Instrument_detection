from flask import Flask
from flask_restful import Resource, Api, reqparse
import base64

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('file', type=str, help='File to be uploaded')

class Receive(Resource):
    def get(self):
        return {'data': 'Hello World'}

    def post(self):
        args = parser.parse_args()
        file = args['file']
        with open("temp.wav", "wb") as wav_file:
            decode_string = base64.b64decode(file)
            wav_file.write(decode_string)
            wav_file.close()
            # model(decode_string)
    
        return {'Message': 'File uploaded successfully'}

api.add_resource(Receive, '/receive')

if __name__ == '__main__':
    app.run()