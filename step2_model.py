import torch.nn as nn

class ClothingClassifier(nn.Module):
    def __init__(self, use_dropout=False):
        super().__init__()
        
        layers = [
            nn.Flatten(),              # Turns a 28x28 image into a flat list of 784 numbers
            nn.Linear(784, 512),       # First hidden layer (512 neurons)
            nn.ReLU(),                 # Activation function
        ]
        
        # Add Dropout only if turned ON
        if use_dropout:
            layers.append(nn.Dropout(0.4))  # Randomly turns off 40% of neurons
            
        layers.extend([
            nn.Linear(512, 256),       # Second hidden layer (256 neurons)
            nn.ReLU(),
        ])
        
        if use_dropout:
            layers.append(nn.Dropout(0.4))
            
        layers.append(nn.Linear(256, 10))   # Output layer: 10 clothing categories
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Quick test to verify both models build cleanly
if __name__ == "__main__":
    model_normal = ClothingClassifier(use_dropout=False)
    model_regularized = ClothingClassifier(use_dropout=True)
    print("Normal model built successfully!")
    print("Regularized model built successfully!")