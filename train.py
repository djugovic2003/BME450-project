import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# -------------------------
# 0. Setup
# -------------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

RESULTS_FILE = "training_results.txt"

with open(RESULTS_FILE, "w") as f:
    f.write("Chest X-Ray Pneumonia Classification Results\n")
    f.write("==========================================\n\n")
    f.write(f"Device used: {device}\n\n")

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
val_dir = "chest_xray/val"
test_dir = "chest_xray/test"

train_data = datasets.ImageFolder(train_dir, transform=transform)
val_data = datasets.ImageFolder(val_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

print("Classes:", train_data.classes)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Testing images:", len(test_data))

with open(RESULTS_FILE, "a") as f:
    f.write(f"Classes: {train_data.classes}\n")
    f.write(f"Training images: {len(train_data)}\n")
    f.write(f"Validation images: {len(val_data)}\n")
    f.write(f"Testing images: {len(test_data)}\n\n")

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
def train_model(model, train_loader, val_loader, epochs, learning_rate, model_name):
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    val_accuracies = []

    with open(RESULTS_FILE, "a") as f:
        f.write(f"\nTraining {model_name}\n")
        f.write("=" * 40 + "\n")

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
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                predictions = model(images)
                loss = loss_fn(predictions, labels)

                total_val_loss += loss.item()
                predicted_classes = predictions.argmax(dim=1)

                correct += (predicted_classes == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        output = (
            f"{model_name} | Epoch {epoch + 1}/{epochs}\n"
            f"Train loss: {avg_train_loss:.4f}\n"
            f"Validation loss: {avg_val_loss:.4f}\n"
            f"Validation accuracy: {val_accuracy * 100:.2f}%\n"
            "----------------------------\n"
        )

        print(output)

        with open(RESULTS_FILE, "a") as f:
            f.write(output)

    return train_losses, val_losses, val_accuracies

# -------------------------
# 6. Final evaluation function
# -------------------------
def evaluate_model(model, test_loader, model_name):
    all_preds = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels,
        all_preds,
        target_names=train_data.classes,
        zero_division=0
    )

    correct = np.trace(cm)
    total = np.sum(cm)
    accuracy = correct / total

    print(f"\nFinal Test Evaluation - {model_name}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(cm)
    print(report)

    with open(RESULTS_FILE, "a") as f:
        f.write(f"\nFinal Test Evaluation - {model_name}\n")
        f.write("=" * 40 + "\n")
        f.write(f"Test accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)
        f.write("\n")

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=train_data.classes
    )
    disp.plot()
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
    plt.close()

    return accuracy

# -------------------------
# 7. Train both models
# -------------------------
EPOCHS = 3
LEARNING_RATE = 0.001

model1 = CNNModel1()
model2 = CNNModel2()

model1_train_loss, model1_val_loss, model1_val_acc = train_model(
    model1,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    model_name="CNN Model 1"
)

model2_train_loss, model2_val_loss, model2_val_acc = train_model(
    model2,
    train_loader,
    val_loader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    model_name="CNN Model 2"
)

# -------------------------
# 8. Plot loss curves
# -------------------------
plt.figure()
plt.plot(model1_train_loss, label="Model 1 Train Loss")
plt.plot(model1_val_loss, label="Model 1 Validation Loss")
plt.plot(model2_train_loss, label="Model 2 Train Loss")
plt.plot(model2_val_loss, label="Model 2 Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("loss_curve.png")
plt.close()

# -------------------------
# 9. Plot validation accuracy curves
# -------------------------
plt.figure()
plt.plot(model1_val_acc, label="Model 1 Validation Accuracy")
plt.plot(model2_val_acc, label="Model 2 Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.legend()
plt.savefig("accuracy_curve.png")
plt.close()

# -------------------------
# 10. Final test evaluation
# -------------------------
model1_test_accuracy = evaluate_model(model1, test_loader, "CNN Model 1")
model2_test_accuracy = evaluate_model(model2, test_loader, "CNN Model 2")

with open(RESULTS_FILE, "a") as f:
    f.write("\nModel Comparison Summary\n")
    f.write("=" * 40 + "\n")
    f.write(f"CNN Model 1 final test accuracy: {model1_test_accuracy * 100:.2f}%\n")
    f.write(f"CNN Model 2 final test accuracy: {model2_test_accuracy * 100:.2f}%\n")

    if model2_test_accuracy > model1_test_accuracy:
        f.write("CNN Model 2 performed better on the final test set.\n")
    elif model1_test_accuracy > model2_test_accuracy:
        f.write("CNN Model 1 performed better on the final test set.\n")
    else:
        f.write("Both models had the same final test accuracy.\n")

print("\nTraining complete.")
print("Saved files:")
print("- training_results.txt")
print("- loss_curve.png")
print("- accuracy_curve.png")
print("- cnn_model_1_confusion_matrix.png")
print("- cnn_model_2_confusion_matrix.png")
