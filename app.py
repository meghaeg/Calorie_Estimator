import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import pandas as pd
import json
import torch.nn as nn

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load food class labels
def load_custom_classes():
    try:
        with open("food_classes.json", "r") as f:
            class_mapping = json.load(f)
        class_mapping = {int(k): v for k, v in class_mapping.items()}
        print(f"✅ Loaded Food Classes: {class_mapping}")
        return class_mapping
    except Exception as e:
        print(f"❌ Error loading food_classes.json: {e}")
        return {}

custom_classes = load_custom_classes()

# Load trained model
def load_trained_model():
    if not custom_classes:
        print("❌ No class labels found. Model cannot be initialized.")
        return None, None

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  # Updated weights
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(custom_classes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = "food_model.pth"
    if not os.path.exists(model_path):
        print(f"❌ Model file '{model_path}' not found.")
        return None, None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully.")
    return model, device

model, device = load_trained_model()

# Load food calorie data
# Load food calorie data
def load_calories():
    try:
        df = pd.read_csv("food_calories.csv")
        df.columns = df.columns.str.strip()
        df["Food Item"] = df["Food Item"].str.strip().str.lower()
        print("✅ Calorie data loaded.")
        return df
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

food_calories = load_calories()

# Predict food and return calorie value
def predict_food(image_path):
    image_tensor = transform_image(image_path).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)

    predicted_label = custom_classes.get(predicted_class.item(), "Unknown Food")

    if food_calories is not None and 'Food Item' in food_calories.columns:
        match = food_calories[food_calories['Food Item'] == predicted_label.lower()]
        calories = match['Calories'].values[0] if not match.empty else "Unknown"
    else:
        calories = "Unknown"

    return predicted_label.title(), calories


# Preprocess image
def transform_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Predict food


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    food_name, calories = predict_food(file_path)

    return render_template("result.html", image=file.filename, food_name=food_name, calories=calories)

if __name__ == "__main__":
    app.run(debug=True)
