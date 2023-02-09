import requests
import base64

url = 'http://localhost:5000/receive'

with open('test_file.wav', 'rb') as wav:

    wav_send = base64.b64encode(wav.read())
    files = { "file": wav_send.decode('ascii') }
    print(files)
    req = requests.post(url, files=files)

    print(req.status_code)
    print(req.text)