import streamlit as st
import cv2
import numpy as np  
from PIL import Image
import torch 
from torch import nn # for the interpolate function
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation # for semantic segmentation tasks
from io import BytesIO  # to download the image directly from the Streamlit app without needing to save to file

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.to(device)

def run_inference(image):
    """Define the main inference function for segmentation"""
    inputs = image_processor(images=image, return_tensors="pt").to(device) # SegformerImageProcessor
    # return_tensor="pt" specifies that the processed images should be returned as PyTorch tensors (standard data format)
    outputs = model(**inputs) 
    logits = outputs.logits  # model's predictions for each pixel, but are not yet normalized or converted to probabilities.
    # Upsample the logits to the size of the original image for accurate pixel-wise classification
    upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False) # for resizing images or tensors. 'bilinear' for a balance of speed and quality for upscaling
    # The align_corners=False argument specifies how the interpolation handles the corner pixels. nearest, bicubic are alternative modes
    labels = upsampled_logits.argmax(dim=1)[0] 
    labels_viz = labels.cpu().numpy()  # Move the labels to CPU and convert to NumPy array for visualization
    unwanted_labels = [0, 16, 17, 18] # These labels include background, clothes and other unwanted lables for the task
    segment = np.isin(labels_viz, unwanted_labels, invert=True) # Filter out unwanted labels to ignore by making them transparent
    # The invert=True argument inverts the mask, so that pixels with unwanted labels are set to False (transparent) and vice versa.

    # Create an RGBA image where A (Alpha) channel controls opacity of pixels
    rgba_image = np.zeros((*segment.shape, 4), dtype=np.uint8)
    rgb_image = np.array(image.convert('RGB'))  # Convert input image to RGB
    rgba_image[:,:, :3] = rgb_image  # Set RGB channels
    rgba_image[:,:, 3] = segment * 255  # Set alpha channel based on segmentation mask
    return Image.fromarray(rgba_image)  # Return the final RGBA image (image with transparency)

def detect_faces(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale for face detection
    # Load OpenCV's pre-trained Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    return faces

st.title("Face Segmentation App")
run_on_all_photos = st.sidebar.radio('🚨 Override num_face limit. (Result might be inaccurate)', ["No", "Yes"], horizontal=True)
uploaded_file = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_rgb = img.convert('RGB')
    open_cv_image = np.array(img_rgb)
    if st.button('Process'):
        with st.spinner('Processing...'):
            faces = detect_faces(open_cv_image)  # Detect the number of faces in the image
            n_faces = len(faces)
            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(open_cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if n_faces == 1 or run_on_all_photos == "Yes":
                col1, col2, col3 = st.columns(3)  
                with col1:
                    st.image(img_rgb, caption="Original", use_column_width=True)  
                with col2:
                    st.image(open_cv_image, caption="Detected", use_column_width=True)
                    n_faces = len(faces)
                with col3:
                    processed_image = run_inference(img_rgb)  
                    st.image(processed_image, caption="Segmented", use_column_width=True) 
                buf = BytesIO()
                processed_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(label="Download Segmented Image", data=byte_im, file_name="segmented.png", mime="image/png")

            else:
                if n_faces > 1:
                    st.warning("Multiple faces found. Please upload an image with only one face.")
                if n_faces == 0:
                    st.warning("No face found. Please upload an image with a face.")