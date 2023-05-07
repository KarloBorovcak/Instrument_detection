import streamlit as st
import requests
import json

st.set_page_config(layout="wide")
st.title("Instrument classifier")

st.write("""Our innovative web application allows you to effortlessly upload any WAV file and receive an accurate instrument classification in a matter of seconds. Using a state-of-the-art Convolutional Neural Network (CNN) model, our app is capable of recognizing 11 different instruments including cello, clarinet, flute, acoustic guitar, electric guitar, organ, piano, saxophone, trumpet, violin, and even voice!
No longer will you have to spend countless hours manually identifying the various instruments within an audio recording. Our powerful AI-powered technology utilizes advanced machine learning techniques to deliver reliable and precise instrument classifications. Simply upload your audio file and let our app do the rest!
Whether you're a musician, audio engineer, or simply have an interest in audio analysis, our web application is the perfect solution for instrument classification. Try it out today and experience the unparalleled accuracy and efficiency of our innovative tool!""")
         
upload_message = "Upload a .wav file to classify the instruments in it"
st.markdown(f"<p style='font-size: 20px; margin-bottom:-50px'>{upload_message}</p>", unsafe_allow_html=True)
wavfile = st.file_uploader(label=" ", type=("wav", "wave"))
  
instruments_name = {"cel": "Cello", "cla": "Clarinet", "flu": "Flute", "gac": "Acoustic guitar", "gel": "Electric guitar", "org": "Organ", "pia": "Piano", "sax": "Saxophone", "tru": "Trumpet", "vio": "Violin", "voi": "Voice"}
headers = {"accept": "application/json",}

if wavfile is not None:
    cols = st.columns(5)
    with cols[2]:
        clicked = st.button("Predict", use_container_width=True)
    if clicked:
        st.write("")
        cols2 = st.columns(5)
        with cols2[2]:
            with st.spinner("Predicting..."):
                response = requests.post("http://backend:8000/upload-file", headers=headers, files={"file": (wavfile.name, wavfile.read(), wavfile.type)})

        data = json.loads(response.json())
        counts = {'0': 0, '1': 0}
        for key in data:
            counts[str(data[key])] += 1
        if counts['0'] == 11:
            cols = st.columns(5)
            with cols[2]:
                st.markdown(f"<p style='font-size: 20px; margin-bottom:-50px'>No instruments detected</p>", unsafe_allow_html=True)
        else:
            st.write("Detected instruments:")
            cols = st.columns(5)
            cols2 = st.columns(6)
            cntr = 0
            width = 280
            for instrument in data:
                if data[instrument] == 0:
                    continue
                if cntr < 5:
                    with cols[cntr]:
                        st.image(f"images/{instrument}.png", caption=f"{instruments_name[instrument]}", width=width)
                        cntr += 1
                else:
                    with cols2[cntr - 5]:
                        st.image(f"images/{instrument}.png", caption=f"{instruments_name[instrument]}", width=width)
                        cntr += 1
