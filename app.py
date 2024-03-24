import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("Face Segmentation App")
run_on_all_photos = st.sidebar.radio('🚨 Override num_face limit. (Result might be inaccurate)', ["No", "Yes"], horizontal=True)
uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_rgb = img.convert('RGB')

    if st.button('Process'):
        with st.spinner('Processing...'):
            files = {'image': uploaded_file.getvalue()}
            params = {'run_on_all_photos': run_on_all_photos}
            response = requests.post('http://backend:8000/process', files=files, params=params)

            if response.status_code == 200:
                data = response.json()
                original_image = Image.open(BytesIO(uploaded_file.getvalue()))
                detected_image = Image.open(BytesIO(data['detected_image']))
                segmented_image = Image.open(BytesIO(data['segmented_image']))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(original_image, caption="Original", use_column_width=True)
                with col2:
                    st.image(detected_image, caption="Detected", use_column_width=True)
                    st.write(f"Number of faces: {data['n_faces']}")
                with col3:
                    st.image(segmented_image, caption="Segmented", use_column_width=True)

                st.download_button(label="Download Segmented Image", data=data['segmented_image'], file_name="segmented.png", mime="image/png")
            else:
                st.error(response.json()['detail'])