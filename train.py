import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# -------------------------
# 1. Data transforms
# -------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# -------------------------
# 2. Load dataset
# -------------------------
train_dir = "chest_xray/train"
test_dir = "chest_xray/test"

train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print("Classes:", train_data.classes)

# -------------------------
# 3. CNN Model 1
# -------------------------
class CNNModel1(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.network(x)

# -------------------------
# 4. CNN Model 2
# -------------------------
class CNNModel2(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.network(x)

# -------------------------
# 5. Training function
# -------------------------
def train_model(model, train_loader, test_loader, epochs, learning_rate, model_name):
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images)
            loss = loss_fn(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                loss = loss_fn(predictions, labels)

                total_test_loss += loss.item()
                predicted_classes = predictions.argmax(dim=1)

                correct += (predicted_classes == labels).sum().item()
                total += labels.size(0)

        avg_test_loss = total_test_loss / len(test_loader)
        accuracy = correct / total

        test_losses.append(avg_test_loss)
        test_accuracies.append(accuracy)

        print(f"{model_name} | Epoch {epoch+1}/{epochs}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Test loss: {avg_test_loss:.4f}")
        print(f"Test accuracy: {accuracy*100:.2f}%")
        print("----------------------------")

    return train_losses, test_losses, test_accuracies

# -------------------------
# 6. Train both models
# -------------------------
model1 = CNNModel1()
model2 = CNNModel2()

model1_train_loss, model1_test_loss, model1_acc = train_model(
    model1, train_loader, test_loader, epochs=10, learning_rate=0.001, model_name="CNN Model 1"
)

model2_train_loss, model2_test_loss, model2_acc = train_model(
    model2, train_loader, test_loader, epochs=10, learning_rate=0.001, model_name="CNN Model 2"
)

# -------------------------
# 7. Plot loss curves
# -------------------------
plt.figure()
plt.plot(model1_train_loss, label="Model 1 Train Loss")
plt.plot(model1_test_loss, label="Model 1 Test Loss")
plt.plot(model2_train_loss, label="Model 2 Train Loss")
plt.plot(model2_test_loss, label="Model 2 Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Test Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.show()

# -------------------------
# 8. Plot accuracy curves
# -------------------------
plt.figure()
plt.plot(model1_acc, label="Model 1 Test Accuracy")
plt.plot(model2_acc, label="Model 2 Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracy")
plt.legend()
plt.savefig("accuracy_curve.png")
plt.show()

# -------------------------
# 9. Confusion matrix for Model 2
# -------------------------
all_preds = []
all_labels = []

model2.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model2(images)
        preds = outputs.argmax(dim=1).cpu()

        all_preds.extend(preds)
        all_labels.extend(labels)

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_data.classes)
disp.plot()
plt.title("Confusion Matrix - CNN Model 2")
plt.savefig("confusion_matrix.png")
plt.show()