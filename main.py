import streamlit as st
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import base64

backend_url = "http://localhost:8000/process_image"

def process_image(image_file):
    files = {"file": image_file.getvalue()}
    response = requests.post(backend_url, files=files)
    return response.json()

def display_segmented_image(segmented_image_data):
    segmented_image = Image.open(BytesIO(base64.b64decode(segmented_image_data)))
    return segmented_image

def main():
    st.title("Face Segmentation App")
    uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_rgb = img.convert('RGB')
        
        if st.button('Process'):
            with st.spinner('Processing...'):
                result = process_image(uploaded_file)
                segmented_image_data = result["segmented_image"]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img_rgb, caption="Original", use_column_width=True)
                with col2:
                    processed_image = display_segmented_image(segmented_image_data)
                    st.image(processed_image, caption="Segmented", use_column_width=True)
                
                buf = BytesIO()
                processed_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(label="Download Segmented Image", data=byte_im, file_name="segmented.png", mime="image/png")

if __name__ == "__main__":
    main()