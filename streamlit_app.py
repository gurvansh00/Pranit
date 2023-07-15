# importing libraries
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import time

#setting the page
st.title('----')
st.markdown('This webapp uses an state of the art ML model trained on predefined ML model to classify your skin status as Benign or Malignant form of Melanome')
st.markdown(
'''
Steps to use the app:
- Take the image of your skin disorder from your phone camera
- Launch the webapp again
- Upload the image and wait for the processing to be completed
- Get the prelimenary result and consult a doctor'''
)

def set_bg_hack_url():
    '''
    A function to unpack an image from url and set as bg.
    Returns
    -------
    The background.
    '''
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://img.freepik.com/free-vector/abstract-medical-wallpaper-template-design_53876-61802.jpg?w=1800&t=st=1689312423~exp=1689313023~hmac=b81f19f0fa8493e07a3a32fc33ca6365ad7aaf99cdc2bbb02c18eaa5e5c77da0");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

set_bg_hack_url()
#loading the model
model = YOLO('best.pt')

#image loader
img = st.file_uploader('Select the image', type=['jpg','png','jpeg'])
if img is not None:
	img = Image.open(img)
	st.markdown('Image Visualization')
	st.image(img)
	st.header('Melanoma Form Classification')
	res = model.predict(img)
	label = res[0].probs.top5
	conf = res[0].probs.top5conf
	conf = conf.tolist()
	col1,col2 = st.columns(2)
	col1.subheader(res[0].names[label[0]].title() +' with '+ str(conf[0])+' Confidence')
	col2.subheader(res[0].names[label[1]].title() +' with '+ str(conf[1])+' Confidence')
