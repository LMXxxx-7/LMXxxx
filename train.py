import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device("cpu")

BATCH_SIZE = 16
EPOCHS = 15
LR = 0.001
DATA_DIR = "data"

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"),
    transform=transform
)

val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"),
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
train_accs = []

best_acc = 0

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(train_loader)

    train_losses.append(avg_loss)
    train_accs.append(acc)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {avg_loss:.4f} "
        f"Acc: {acc*100:.2f}%"
    )

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/best_model.pth")

print("\nTraining Finished.")
print("Best model saved.")

# evaluation
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=["crack", "normal"]
))

# loss curve
plt.figure(figsize=(8, 5))
plt.plot(train_losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("outputs/loss_curve.png")
plt.close()

# confusion matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=["crack", "normal"],
    yticklabels=["crack", "normal"]
)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

print("Saved plots to outputs/")
