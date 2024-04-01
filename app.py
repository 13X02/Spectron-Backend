from flask import Flask, request, jsonify, url_for
from flask_cors import CORS
import os
import io
import numpy as np
from PIL import Image
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'})

    image_file = request.files['image']    
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded image
    image_path = "uploads/" + image_file.filename
    image_file.save(image_path)

    # Perform inference
    processed_result = predict_image(image_path)

    # Delete the uploaded image file
    os.remove(image_path)

    return jsonify(processed_result)

# Initialize the model once outside the function to avoid loading it on every request
cfg = get_cfg()
cfg.merge_from_file("config.yaml")
cfg.MODEL.DEVICE = 'cpu'  # Set to 'cuda' if you have GPU
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for detections
cfg.MODEL.WEIGHTS = "model_final.pth"  # Load your trained model
predictor = DefaultPredictor(cfg)

import base64

def predict_image(image_path):
    # Read the image
    image = np.array(Image.open(image_path).convert("RGB"))

    # Perform inference
    outputs = predictor(image)

    # Get the detected instances
    instances = outputs["instances"].to("cpu")

    # Get the metadata
    metadata = {"thing_classes":["Ignore","Illiminite", "Kynite-0", "Kynite-45", "Quartz-color", "Quartz-colorless", "Rutile-0", "Rutile-90", "Silm-0", "Silm-40", "Silm-60", "Silm-90", "Zircoin"]}

    # Get the category names
    category_names = ["Ignore","Illiminite", "Kynite-0", "Kynite-45", "Quartz-color", "Quartz-colorless", "Rutile-0", "Rutile-90", "Silm-0", "Silm-40", "Silm-60", "Silm-90", "Zircoin"]

    # Initialize a dictionary to store the count of each instance
    instance_counts = {}

    # Loop through each detected instance
    for i in range(len(instances)):
        label = instances.pred_classes[i].item()
        if category_names:
            label_name = category_names[label]
        else:
            label_name = f"Instance {label}"
        # Increment the count for this instance type
        instance_counts[label_name] = instance_counts.get(label_name, 0) + 1

    # Visualize predictions on the image using Detectron2's Visualizer
    v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    processed_image = v.get_image()[:, :, ::-1]

    # Convert processed image to base64
    buffered = io.BytesIO()
    Image.fromarray(processed_image).save(buffered, format="JPEG")
    processed_image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")


    return {
        "numInstances": len(instances),
        "instanceCounts": instance_counts,
        "processedImageBase64": processed_image_base64
    }

if __name__ == '__main__':
    app.run(debug=True)
