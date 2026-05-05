import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

device = torch.device("cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

img = Image.open("test.jpg").convert("RGB")
x = transform(img).unsqueeze(0)

with torch.no_grad():
    out = model(x)
    pred = torch.argmax(out, 1).item()

classes = ["crack", "normal"]

print("Prediction:", classes[pred])
