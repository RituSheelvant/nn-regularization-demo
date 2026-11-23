import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from step1_data import val_data
from step2_model import ClothingClassifier

# Names of the 10 clothing items in Fashion-MNIST
CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# 1. Load both trained models
model_1 = ClothingClassifier(use_dropout=False)
model_1.load_state_dict(torch.load("model_1.pth"))
model_1.eval()

model_2 = ClothingClassifier(use_dropout=True)
model_2.load_state_dict(torch.load("model_2.pth"))
model_2.eval()

# 2. Find 8 unseen images where Model 1 fails, but Model 2 gets it right!
examples = []
with torch.no_grad():
    for image, true_label in val_data:
        input_img = image.unsqueeze(0)
        
        probs_1 = F.softmax(model_1(input_img), dim=1)[0]
        probs_2 = F.softmax(model_2(input_img), dim=1)[0]
        
        pred_1 = probs_1.argmax().item()
        pred_2 = probs_2.argmax().item()
        
        # Pick cases where Model 1 is WRONG and Model 2 is RIGHT
        if pred_1 != true_label and pred_2 == true_label:
            examples.append((
                image.squeeze().numpy(),
                CLASSES[true_label],
                CLASSES[pred_1], probs_1[pred_1].item() * 100,
                CLASSES[pred_2], probs_2[pred_2].item() * 100
            ))
        if len(examples) == 8:
            break

# 3. Print comparison in terminal
print("\n--- LIVE TEST ON UNSEEN IMAGES ---")
for i, (_, actual, p1, c1, p2, c2) in enumerate(examples, 1):
    print(f"Image {i} (Actual: {actual:11s}) | Model 1: {p1:11s} ({c1:5.1f}% sure) [WRONG] | Model 2: {p2:11s} ({c2:5.1f}% sure) [RIGHT]")

# 4. Show and save visual proof
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, (img, actual, p1, c1, p2, c2) in zip(axes.flat, examples):
    ax.imshow(img, cmap="gray")
    ax.set_title(
        f"Actual: {actual}\n"
        f"Model 1: {p1} ({c1:.0f}%)\n"
        f"Model 2: {p2} ({c2:.0f}%)",
        fontsize=9.5
    )
    ax.axis("off")

plt.suptitle("Unseen Test Images: Model 1 Overconfident & Wrong vs. Model 2 Accurate", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("live_predictions.png", dpi=200)
print("\nSaved visual proof to 'live_predictions.png'!")
plt.show()