from torchvision import datasets, transforms

# Define a simple transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder("dataset", transform=transform)

# Print the classes
print("Classes found in dataset:", dataset.classes)
