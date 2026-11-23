import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms
import matplotlib.pyplot as plt
from step1_data import val_data
from step2_model import ClothingClassifier

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load both trained models
model_1 = ClothingClassifier(use_dropout=False)
model_1.load_state_dict(torch.load("model_1.pth"))
model_1.eval()

model_2 = ClothingClassifier(use_dropout=True)
model_2.load_state_dict(torch.load("model_2.pth"))
model_2.eval()

# Transform for custom external image files
custom_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("=== INTERACTIVE MODEL TESTER ===")
print("• Type any test image number (0 to 999) to test from the dataset")
print("• OR type the filename of your own image (e.g., my_shoe.png)")
print("• Type 'q' to quit\n")

while True:
    choice = input("Enter image number (0-999) or image path: ").strip()
    if choice.lower() == "q":
        break

    # Option A: User entered a dataset index (0 to 999)
    if choice.isdigit() and 0 <= int(choice) < len(val_data):
        img_tensor, label_idx = val_data[int(choice)]
        actual_name = CLASSES[label_idx]
    # Option B: User entered a local image file path
    else:
        try:
            raw_img = Image.open(choice).convert("L")
            # Invert colors if image has a bright/white background (Fashion-MNIST uses black background)
            if raw_img.getpixel((0, 0)) > 128:
                raw_img = ImageOps.invert(raw_img)
            img_tensor = custom_transform(raw_img)
            actual_name = f"Custom File ({choice})"
        except Exception:
            print("Invalid input! Enter a number between 0-999 or a valid image filename.\n")
            continue

    # Run both models on the chosen image
    with torch.no_grad():
        probs_1 = F.softmax(model_1(img_tensor.unsqueeze(0)), dim=1)[0]
        probs_2 = F.softmax(model_2(img_tensor.unsqueeze(0)), dim=1)[0]

    p1, c1 = probs_1.argmax().item(), probs_1.max().item() * 100
    p2, c2 = probs_2.argmax().item(), probs_2.max().item() * 100

    print(f"\nActual Item  : {actual_name}")
    print(f"Model 1 (No Reg)   -> {CLASSES[p1]:11s} ({c1:.1f}% sure)")
    print(f"Model 2 (With Reg) -> {CLASSES[p2]:11s} ({c2:.1f}% sure)\n")

    # Pop up the image with both predictions
    plt.figure(figsize=(4, 4))
    plt.imshow(img_tensor.squeeze().numpy(), cmap="gray")
    plt.title(f"Actual: {actual_name}\nM1: {CLASSES[p1]} ({c1:.0f}%) | M2: {CLASSES[p2]} ({c2:.0f}%)")
    plt.axis("off")
    plt.show()