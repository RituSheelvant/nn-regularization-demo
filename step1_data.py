from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# 1. Convert images into numbers that the neural network can read
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 2. Download the Fashion-MNIST dataset automatically
print("Downloading dataset...")
full_train = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
full_val = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# 3. Take a small slice: 2,000 images to train on, 1,000 images to test on
train_data = Subset(full_train, range(2000))
val_data = Subset(full_val, range(1000))

# 4. Group them into small batches of 64 images at a time
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=256, shuffle=False)

print(f"Ready! Trainingg images: {len(train_data)} | Validation images: {len(val_data)}")








#hello 
#hiiiii   vjhgjikhkjhiu
#gjhkupupu;k'k'