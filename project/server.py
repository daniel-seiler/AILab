import torch
from fastapi import FastAPI, File, UploadFile, Request
import io
import torchvision.transforms as transforms
from fastapi.templating import Jinja2Templates
import torch.nn as nn
from PIL import Image

# To start: python -m uvicorn server:app --reload

app = FastAPI()

model_path = 'best_acc_model.pth'

width = 224
height = 224


class CNN(nn.Module):
    def __init__(self, num_classes=1):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(16, 16, kernel_size=15, padding=7),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(16 * int(width/(2**7)) * int(height/(2**7)), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x.reshape(-1)

model = CNN(num_classes=1)
# Load model state
model.load_state_dict(torch.load(model_path))
model.eval()



transform = transforms.Compose([
    transforms.Resize((width, height)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


templates = Jinja2Templates(directory="templates")

# FastAPI endpoint to predict the label of an image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Apply transformation to the image
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        predicted = (output>0.5).float()
   
    return { "prediction": predicted.item() }



@app.get("/")
async def get_image_upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

