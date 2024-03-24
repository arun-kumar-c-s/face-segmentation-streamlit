# Face Segmentation
 A Streamlit web application that performs semantic segmentation of facial images using the SegFormer model. It allows users to upload an image, detects faces in the image, and performs pixel-wise segmentation to isolate the facial regions from the background.

## Features
- Face detection using OpenCV's Haar Cascade classifier
- Semantic segmentation using the pre-trained SegFormer model
- Supports uploading images in JPG, JPEG, and PNG formats
- Displays the original image, detected faces, and segmented image
- Option to override the number of faces limit
- Download the segmented image as a PNG file with transparent background


## Installation
Clone the repository:

```
git clone https://github.com/yourusername/face-segmentation-app.git
```
Change into the project directory:
```
cd face-segmentation-app
```
Install the required dependencies:
```
pip install -r requirements.txt
pip install -r requirements-backend.txt
```


## Usage
Run the Streamlit app:

```
streamlit run app_standalone.py
```

- Open the provided URL in your web browser.
- Upload an image containing a face using the file uploader.
- Click the "Process" button to perform face detection and segmentation.
- View the original image, detected faces, and segmented image.
- Optionally, override the number of faces limit using the sidebar option.
- Download the segmented image by clicking the "Download Segmented Image" button.


## Built with
Streamlit, OpenCV, Transformers

## Credits
- SegFormer model: HuggingFace, jonathandinu/face-parsing
- Haar Cascade classifier: OpenCV
