from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from io import BytesIO

app = FastAPI()

device = "cpu"
# image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
# model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")

# For docker
image_processor = SegformerImageProcessor.from_pretrained("/root/.cache/huggingface/hub/models--jonathandinu--face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("/root/.cache/huggingface/hub/models--jonathandinu--face-parsing")
model.to(device)

class RunInferenceParams(BaseModel):
    run_on_all_photos: str

@app.post("/process")
async def process_image(image: UploadFile = File(...), params: RunInferenceParams = None):
    contents = await image.read()
    img = Image.open(BytesIO(contents))
    img_rgb = img.convert('RGB')
    open_cv_image = np.array(img_rgb)

    faces = detect_faces(open_cv_image)
    n_faces = len(faces)

    for (x, y, w, h) in faces:
        cv2.rectangle(open_cv_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if n_faces == 1 or (params and params.run_on_all_photos == "Yes"):
        processed_image = run_inference(img_rgb)
        segmented_image_bytes = BytesIO()
        processed_image.save(segmented_image_bytes, format="PNG")
        segmented_image_bytes.seek(0)

        detected_image_bytes = BytesIO()
        Image.fromarray(open_cv_image).save(detected_image_bytes, format="PNG")
        detected_image_bytes.seek(0)

        return {
            'n_faces': n_faces,
            'detected_image': detected_image_bytes.getvalue(),
            'segmented_image': segmented_image_bytes.getvalue()
        }
    else:
        if n_faces > 1:
            raise HTTPException(status_code=400, detail="Multiple faces found. Please upload an image with only one face.")
        if n_faces == 0:
            raise HTTPException(status_code=400, detail="No face found. Please upload an image with a face.")

def run_inference(image):
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
    labels = upsampled_logits.argmax(dim=1)[0]
    labels_viz = labels.cpu().numpy()
    unwanted_labels = [0, 16, 17, 18]
    segment = np.isin(labels_viz, unwanted_labels, invert=True)
    rgba_image = np.zeros((*segment.shape, 4), dtype=np.uint8)
    rgb_image = np.array(image.convert('RGB'))
    rgba_image[:, :, :3] = rgb_image
    rgba_image[:, :, 3] = segment * 255
    return Image.fromarray(rgba_image)

def detect_faces(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    return faces