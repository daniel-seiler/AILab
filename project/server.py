import torch
from fastapi import FastAPI, File, UploadFile, Request
import io
import torchvision.transforms as transforms
import torchvision
from fastapi.templating import Jinja2Templates
import torch.nn as nn
from PIL import Image

# To start: python -m uvicorn server:app --reload

app = FastAPI()

model_path_cnn = '/app/model/Simple_CNN_250x250_Reszize_FullRun_1750364050.5667918.pth'
model_path_small_eff = '/app/model/SmallEfficientNet_250x250_Reszize_FullRun_1750364050.5667918.pth'
model_path_efficientnet = '/app/model/EfficientNet_250x250_Reszize_FullRun_1750364050.5667918.pth'
model_path_squeezenet = '/app/model/SqueezeNet_250x250_Reszize_FullRun_1750364050.5667918.pth'

width = 250
height = 250

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class Simple_CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(Simple_CNN, self).__init__()
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

            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

            nn.Conv2d(32, 32, kernel_size=15, padding=7),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4, 4),

        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(32 * int(width/(2**7)) * int(height/(2**7)), 4 * int(width/(2**7)) * int(height/(2**7))),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4 * int(width/(2**7)) * int(height/(2**7)), num_classes),
            #nn.Sigmoid(),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x #x.reshape(-1)

class FusedMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=1, stride=1, squeeze=0, internal_kernel=3):
        super().__init__()
        expanded_channels = int(in_channels * expansion_factor)
        hidden_channels = expanded_channels
        self.expand = nn.Identity()

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=internal_kernel, stride=stride, padding=int((internal_kernel-1)/2), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU()
        )
        self.has_se = squeeze != 0
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=hidden_channels, out_channels=squeeze, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(in_channels=squeeze, out_channels=hidden_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Projection convolution
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.dropout = torchvision.ops.StochasticDepth(0.025, mode="row")
        # Use residual connection if possible
        self.use_residual = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        if self.has_se:
            x = x * self.se(x)
        x = self.project(x)

        if self.use_residual:
            x = self.dropout(x)
            x = x + identity

        return x

class SmallEfficientNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        self.blocks = nn.Sequential(
            FusedMBConvBlock(32, 32, expansion_factor=1, stride=1, squeeze=4, internal_kernel=3), #112
            nn.MaxPool2d(2),
            FusedMBConvBlock(32, 32, expansion_factor=2, stride=1, squeeze=4, internal_kernel=3), #56
            nn.MaxPool2d(2),
            FusedMBConvBlock(32, 32, expansion_factor=2, stride=1, squeeze=6, internal_kernel=5), #28
            nn.MaxPool2d(2),
            FusedMBConvBlock(32, 32, expansion_factor=2, stride=1, squeeze=8, internal_kernel=7), #14
            nn.MaxPool2d(2),
            FusedMBConvBlock(32, 32, expansion_factor=2, stride=1, squeeze=6, internal_kernel=7), #7
            nn.MaxPool2d(2),
            FusedMBConvBlock(32, 32, expansion_factor=2, stride=1, squeeze=4, internal_kernel=3),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

simple_cnn = Simple_CNN()
# Load model state
simple_cnn.load_state_dict(torch.load(model_path_cnn,map_location=device))
simple_cnn.eval()

small_eff = SmallEfficientNet()
# Load model state
small_eff.load_state_dict(torch.load(model_path_small_eff,map_location=device))
small_eff.eval()

efficientnet = torchvision.models.efficientnet_b0(pretrained=False)
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, 2)
# Load model state
efficientnet.load_state_dict(torch.load(model_path_efficientnet,map_location=device))
efficientnet.eval()

squeezenet = torchvision.models.squeezenet1_1(pretrained=False)
squeezenet.classifier[1] = nn.Conv2d(512, 2, kernel_size=1,stride=1)
# Load model state
squeezenet.load_state_dict(torch.load(model_path_squeezenet,map_location=device))
squeezenet.eval()



transform = transforms.Compose([
    transforms.Resize((width, height)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


templates = Jinja2Templates(directory="templates")

# FastAPI endpoint to predict the label of an image
@app.post("/predict")
async def predict(file: UploadFile = File(...), model_id: int = 2):
    # Read image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    # Apply transformation to the image
    image = transform(image).unsqueeze(0)

    image = image.to(device)
    
    with torch.no_grad():
        if model_id == 1:
            output = simple_cnn(image)
        elif model_id == 2:
            output = small_eff(image)
        elif model_id == 3:
            output = efficientnet(image)
        else:
            output = squeezenet(image)
        if (output.shape[1] == 2):
            predicted = output.argmax(dim=1).float()
        else:
            predicted = (output>0.5).float()
   
    return { "prediction": predicted.item() }



@app.get("/")
async def get_image_upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

