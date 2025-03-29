import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd

# Load pretrained ResNet50 model trained on Food-101
model = models.resnet50(weights="IMAGENET1K_V1")  # Pretrained on ImageNet
model.fc = torch.nn.Linear(model.fc.in_features, 101)  # Adjust for Food-101 classes

# Load trained model weights
model.load_state_dict(torch.load("food101_resnet50.pth", map_location=torch.device('cpu')))
model.eval()  # Set to evaluation mode

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load food classes from Food-101 dataset
food_classes = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    # Add all 101 class names...
]

# Load food calorie dataset
food_data = pd.read_csv("food_calories.csv")

def predict_food(image_path):
    """Predict the food item from the image and estimate calories."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_index = predicted.item()

    if predicted_index >= len(food_classes):
        return "Unknown Food", "N/A"

    predicted_label = food_classes[predicted_index]

    # Find calories from dataset
    calories = food_data.loc[food_data["Food"] == predicted_label, "Calories"].values
    calorie_estimate = calories[0] if len(calories) > 0 else "Unknown"

    return predicted_label, calorie_estimate
