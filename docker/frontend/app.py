import streamlit as st
import requests
import json
st.title("Instrument classifier")

st.write("This is an instrument classifier which uses a CNN. It can recognize 11 instruments: cello, clarinet, flute, acoustic guitar, electric guitar, organ, piano, saxophone, trumpet, violin and voice.")
upload_message = "Upload a .wav file to classify the instruments in it"
st.markdown(f"<p style='font-size: 20px; margin-bottom:-50px'>{upload_message}</p>", unsafe_allow_html=True)
wavfile = st.file_uploader(label=" ", type=("wav", "wave"))
  
# headers = {'Content-Type': 'multipart/form-data'}
instruments_name = {"cel": "Cello", "cla": "Clarinet", "flu": "Flute", "gac": "Acoustic guitar", "gel": "Electric guitar", "org": "Organ", "pia": "Piano", "sax": "Saxophone", "tru": "Trumpet", "vio": "Violin", "voi": "Voice"}
headers = {"accept": "application/json",}

if wavfile is not None:
    cols = st.columns(3)
    with cols[1]:
        clicked = st.button("Predict", use_container_width=True)
    if clicked:
        st.write("")
        cols2 = st.columns(3)
        with cols2[1]:
            with st.spinner("Predicting..."):
                response = requests.post("http://backend:8000/audio", headers=headers, files={"file": (wavfile.name, wavfile.read(), wavfile.type)})

        data = json.loads(response.json())
        counts = {'0': 0, '1': 0}
        for key in data:
            counts[str(data[key])] += 1
        if counts['0'] == 11:
            st.write("No instruments detected.")
        else:
            st.write("Detected instruments:")
            cols = st.columns(counts['1'])
            cntr = 0
            for instrument in data:
                if data[instrument] == 0:
                    continue
                with cols[cntr]:
                    st.image(f"slike/{instrument}.jpg", caption=f"{instruments_name[instrument]}", use_column_width=True)
                    cntr += 1
