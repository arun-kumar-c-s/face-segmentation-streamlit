import os
import io
import base64
from fastapi import FastAPI, UploadFile, File
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import torch
from torch import nn
import numpy as np

app = FastAPI()

# Set the cache directory for Hugging Face models
cache_dir = "/app/model_cache"
os.environ["TRANSFORMERS_CACHE"] = cache_dir

# Load the face segmentation model
device = "cpu"
image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
model.to(device)

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')

    # Perform face segmentation
    segmented_image_data = run_inference(image)

    return {"segmented_image": segmented_image_data}

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
    rgb_image = np.array(image)
    rgba_image[:,:, :3] = rgb_image
    rgba_image[:,:, 3] = segment * 255

    segmented_image = Image.fromarray(rgba_image)
    buffered = io.BytesIO()
    segmented_image.save(buffered, format="PNG")
    segmented_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return segmented_image_base64

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)