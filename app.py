import streamlit as st 
from PIL import Image
import numpy as np
from ISR.models import RRDN
from ISR.predict import Predictor
import os

st.title("Super Resolution GAN")
st.text("--"*50)
st.subheader("Upload an image which you want to upscale")   
st.spinner("Testing spinner")

uploaded_file = st.file_uploader("Choose an image...", type=("jpg", "png", "jpeg"))
if uploaded_file is not None:
    file_details = {"FileName":uploaded_file.name,"FileType":uploaded_file.type}
    st.write(file_details)
    img = Image.open(uploaded_file)
    st.image(img)
    with open(os.path.join("input",uploaded_file.name),"wb") as f: 
      f.write(uploaded_file.getbuffer())         
    #st.success("Saved File")


    st.image(img, caption='Uploaded Image.')
    st.write("")
    if st.button('Upscale Now'):
        st.write("upscaling...")
        lr_img = np.array(img)
        model = RRDN(weights='gans') 
        predictor = Predictor(input_dir='input',output_dir='output')
        predictor = predictor.model.load_weights('PSNR_Y_epoch004.hdf5')
        pred = predictor.get_predictions(model=model)# weights_path='PSNR_Y_epoch004.hdf5')
        #sr_img = model.predict(np.array(lr_img))
        pred = Image.fromarray(pred)
        st.image(pred, caption='Upscaled Image', use_column_width=True)  
