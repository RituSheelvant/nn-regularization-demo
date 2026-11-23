import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import our data and model from the first two files
from step1_data import train_loader, val_loader
from step2_model import ClothingClassifier

# Keep results consistent every time you run it
torch.manual_seed(42)

# Helper function to train a model and record its progress
def train_and_track(model, optimizer, epochs=40):
    loss_fn = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # --- TRAINING PHASE (Studying) ---
        model.train()  # Turns Dropout ON (if the model has it)
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            predictions = model(images)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            correct += (predictions.argmax(1) == labels).sum().item()
            total += labels.size(0)

        # --- VALIDATION PHASE (Taking the exam) ---
        model.eval()   # Turns Dropout OFF so the full network is tested
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                predictions = model(images)
                loss = loss_fn(predictions, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (predictions.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        # Save scores for this round
        history["train_loss"].append(train_loss / total)
        history["val_loss"].append(val_loss / val_total)
        history["train_acc"].append(100 * correct / total)
        history["val_acc"].append(100 * val_correct / val_total)

    return history

# 1. Train Model 1: NO Dropout, NO L2 penalty (weight_decay=0)
print("Training Model 1 (No Regularization)...")
torch.manual_seed(42)
model_1 = ClothingClassifier(use_dropout=False)
opt_1 = optim.Adam(model_1.parameters(), lr=0.001, weight_decay=0.0)
hist_1 = train_and_track(model_1, opt_1, epochs=40)

# 2. Train Model 2: WITH Dropout (0.4) + L2 penalty (weight_decay=0.001)
print("Training Model 2 (Dropout + L2 Regularization)...")
torch.manual_seed(42)
model_2 = ClothingClassifier(use_dropout=True)
opt_2 = optim.Adam(model_2.parameters(), lr=0.001, weight_decay=1e-3)
hist_2 = train_and_track(model_2, opt_2, epochs=40)

# 3. Print final comparison in terminal
print("\n--- FINAL RESULTS (After 40 Rounds) ---")
print(f"Model 1 (No Reg)   -> Train Acc: {hist_1['train_acc'][-1]:.1f}% | Val Acc: {hist_1['val_acc'][-1]:.1f}% | Val Loss: {hist_1['val_loss'][-1]:.3f}")
print(f"Model 2 (With Reg) -> Train Acc: {hist_2['train_acc'][-1]:.1f}% | Val Acc: {hist_2['val_acc'][-1]:.1f}% | Val Loss: {hist_2['val_loss'][-1]:.3f}")

# 4. Draw and save comparison graphs
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
rounds = range(1, 41)

# Left Graph: Loss (Lower is better)
ax1.plot(rounds, hist_1["train_loss"], "--", color="red", label="Model 1 Train Loss")
ax1.plot(rounds, hist_1["val_loss"], "-", color="red", linewidth=2, label="Model 1 Val Loss (Overfitting)")
ax1.plot(rounds, hist_2["train_loss"], "--", color="blue", label="Model 2 Train Loss")
ax1.plot(rounds, hist_2["val_loss"], "-", color="blue", linewidth=2, label="Model 2 Val Loss")
ax1.set_title("Loss (Lower is Better)")
ax1.set_xlabel("Training Rounds (Epochs)")
ax1.set_ylabel("Error / Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right Graph: Accuracy (Higher is better)
ax2.plot(rounds, hist_1["train_acc"], "--", color="red", label="Model 1 Train Acc")
ax2.plot(rounds, hist_1["val_acc"], "-", color="red", linewidth=2, label="Model 1 Val Acc")
ax2.plot(rounds, hist_2["train_acc"], "--", color="blue", label="Model 2 Train Acc")
ax2.plot(rounds, hist_2["val_acc"], "-", color="blue", linewidth=2, label="Model 2 Val Acc")
ax2.set_title("Accuracy % (Higher is Better)")
ax2.set_xlabel("Training Rounds (Epochs)")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results_40_epochs.png", dpi=40)
print("\nDone! Open 'results_40_epochs.png' in your folder to see the comparison graph.")

torch.save(model_1.state_dict(), 'model_1.pth')
torch.save(model_2.state_dict(), 'model_2.pth')
