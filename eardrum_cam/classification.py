#%%
#%%
import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from collections import defaultdict
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
from torchvision import models  # Import models from torchvision
#%%
# Step 1: Define the root directory and image transformations
root_dir = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/eardrumDs_kaggle'  # Change this to your root directory containing eardrum images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert PIL image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
])

# Step 2: Filter classes that have more than 50 images
def get_valid_classes(root_dir, min_images=50):
    class_counts = defaultdict(int)
    for root, _, files in os.walk(root_dir):
        class_name = os.path.basename(root)
        if class_name not in ['.', '..']:
            class_counts[class_name] += len([f for f in files if f.endswith('.tiff') or f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.png')])
    valid_classes = [cls for cls, count in class_counts.items() if count >= min_images]
    print(f"Classes with more than {min_images} images: {valid_classes}")
    return valid_classes

valid_classes = get_valid_classes(root_dir, min_images=50)

# Step 3: Create a custom dataset class to filter the valid classes
class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, valid_classes, transform=None):
        super().__init__(root, transform=transform)

        # Filter samples to keep only the valid classes
        self.samples = [s for s in self.samples if self.classes[s[1]] in valid_classes]

        # Create a mapping from old class indices to new continuous indices
        old_class_indices = sorted({s[1] for s in self.samples})  # Unique class indices after filtering
        new_index_mapping = {old_index: new_index for new_index, old_index in enumerate(old_class_indices)}
        
        # Debug: Print the mapping to see how the classes are being remapped
        print(f"Old to New Class Index Mapping: {new_index_mapping}")

        # Remap class indices in self.samples to be continuous
        self.samples = [(s[0], new_index_mapping[s[1]]) for s in self.samples]

        # Update the targets to reflect the new indices
        self.targets = [s[1] for s in self.samples]

        # Update self.class_to_idx and self.classes to match the new mapping
        self.class_to_idx = {self.classes[old_index]: new_index for old_index, new_index in new_index_mapping.items()}
        self.classes = [self.classes[old_index] for old_index in old_class_indices]

        # Debugging print statement to show the number of samples and updated class indices
        print(f"Number of samples after filtering and remapping: {len(self.samples)}")
        print(f"New class-to-index mapping: {self.class_to_idx}")

# Load the dataset with only valid classes
dataset = FilteredImageFolder(root=root_dir, valid_classes=valid_classes, transform=transform)
#%%
# Step 4: Split the dataset into train, validation, and test sets
# Train: 70%, Validation: 15%, Test: 15%
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Step 5: Create DataLoaders for each subset
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Step 6: Display a summary of the data split
print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")
print(f"Number of test images: {len(test_dataset)}")

# Step 1: Define the training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=25, device='cuda'):
    train_loss_history = []
    val_loss_history = []
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimize in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Record the history
            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:4f}')
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history

# Step 2: Define the evaluation function
def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    print("Evaluation Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))

    return acc, auc, precision, recall, f1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model: ResNet50 (adjusted for number of classes)
num_classes = len(valid_classes)  # Assuming valid_classes are defined as in the previous example
resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(resnet50.fc.in_features, num_classes)
resnet50 = resnet50.to(device)

# Define the criterion, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = Adam(resnet50.parameters(), lr=0.001)

# Train the model
trained_model, train_loss, val_loss = train_model(resnet50, train_loader, val_loader, criterion, optimizer, num_epochs=25, device=device)

# Plotting training and validation loss
plt.figure()
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("training_validation_loss.png")  # Save the figure before plt.show(
plt.show()

# Step 4: Evaluate the model on the test set
print("Evaluating on the test set...")
evaluate_model(trained_model, test_loader, device=device)

# Step 5: Save the best model
torch.save(trained_model.state_dict(), 'best_resnet50_eardrum.pth')

# %%
