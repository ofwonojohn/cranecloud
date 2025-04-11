import torch
import torch.nn as nn
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

# Define FastAPI app
app = FastAPI()

# Define the exact model architecture used during training
class CNNModel(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CNNModel, self).__init__()

        self.base_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout_rate),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.classifier = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc_layers(x)
        return self.classifier(x)

# Load the trained model with correct architecture
model = CNNModel()
model.load_state_dict(torch.load("cnn_enhanced.pth", map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define prediction function
def predict(image: Image.Image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    prediction = torch.argmax(output, dim=1).item()
    return "Pothole Detected" if prediction == 1 else "No Pothole"

# FastAPI route for file upload and prediction
@app.post("/predict/")
async def predict_pothole(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    result = predict(image)
    return {"filename": file.filename, "prediction": result}

# Root endpoint
@app.get("/")
def home():
    return {"message": "Pothole Detection API is running!"}
