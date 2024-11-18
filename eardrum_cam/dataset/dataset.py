#%%
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import sys,os
import random
if __name__ == '__main__':
    # Change to the root directory (the parent directory of 'dataset')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(current_dir, '..'))
    os.chdir(root_dir)

    # Ensure that the root directory is in sys.path
    sys.path.insert(0, root_dir)
from utils import get_valid_classes
#%%
# class FilteredImageFolder(datasets.ImageFolder):
#     def __init__(self, root, valid_classes, transform=None):
#         super().__init__(root, transform=transform)

#         # Filter samples to keep only the valid classes
#         self.samples = [s for s in self.samples if self.classes[s[1]] in valid_classes]

#         # Create a mapping from old class indices to new continuous indices
#         old_class_indices = sorted({s[1] for s in self.samples})  # Unique class indices after filtering
#         new_index_mapping = {old_index: new_index for new_index, old_index in enumerate(old_class_indices)}
        
#         # Debug: Print the mapping to see how the classes are being remapped
#         print(f"Old to New Class Index Mapping: {new_index_mapping}")

#         # Remap class indices in self.samples to be continuous
#         self.samples = [(s[0], new_index_mapping[s[1]]) for s in self.samples]

#         # Update the targets to reflect the new indices
#         self.targets = [s[1] for s in self.samples]

#         # Update self.class_to_idx and self.classes to match the new mapping
#         self.class_to_idx = {self.classes[old_index]: new_index for old_index, new_index in new_index_mapping.items()}
#         self.classes = [self.classes[old_index] for old_index in old_class_indices]

#         # Debugging print statement to show the number of samples and updated class indices
#         print(f"Number of samples after filtering and remapping: {len(self.samples)}")
#         print(f"New class-to-index mapping: {self.class_to_idx}")

#%% Update FilteredImageFolder to select a max of 5000 images per valid class
#%% Update FilteredImageFolder to use a random seed
class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, valid_classes, max_images_per_class=5000, transform=None, seed=None):
        super().__init__(root, transform=transform)
        
        # Set the random seed for reproducibility if provided
        if seed is not None:
            random.seed(seed)
        
        # Filter samples to keep only the valid classes and limit each to 5000 images
        class_samples = {cls: [] for cls in valid_classes}
        for s in self.samples:
            class_name = self.classes[s[1]]
            if class_name in valid_classes:
                class_samples[class_name].append(s)
        
        # Select up to 5000 images per class
        self.samples = []
        for cls in valid_classes:
            selected_samples = random.sample(class_samples[cls], min(len(class_samples[cls]), max_images_per_class))
            self.samples.extend(selected_samples)
        
        # Create a mapping from old class indices to new continuous indices
        old_class_indices = sorted({s[1] for s in self.samples})
        new_index_mapping = {old_index: new_index for new_index, old_index in enumerate(old_class_indices)}
        
        # Remap class indices in self.samples to be continuous
        self.samples = [(s[0], new_index_mapping[s[1]]) for s in self.samples]
        self.targets = [s[1] for s in self.samples]
        
        self.class_to_idx = {self.classes[old_index]: new_index for old_index, new_index in new_index_mapping.items()}
        self.classes = [self.classes[old_index] for old_index in old_class_indices]
        
        print(f"Number of samples after filtering and limiting to 5000 per class: {len(self.samples)}")
        print(f"New class-to-index mapping: {self.class_to_idx}")

class GaussianNoise(object):
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
    
def build_dataloader(root_dir, split_ratio=(0.70, 0.15), batch_size=32, num_workers=4, seed=42):
    train_transform = transforms.Compose([
        transforms.RandomApply([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1))], p=1.0),  # Rescale & Aspect Ratio
        transforms.RandomRotation(degrees=(0, 360)),  # Random Rotation between 0 and 360 degrees
        transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.3, 1.7), saturation=(0.7, 1.3), hue=(-0.1, 0.1)),  # Brightness, Contrast, Saturation, Hue
        transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),  # Random Sharpness
        transforms.ToTensor(),  # Convert PIL image to Tensor
        transforms.RandomApply([GaussianNoise(mean=0, std=0.1)], p=0.5),  # Apply Gaussian noise with 50% probability
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to a fixed size for validation and test
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    valid_classes,class_count = get_valid_classes(root_dir, min_images=50)
    # Load the dataset with only valid classes
    valid_classes, class_count = get_valid_classes(root_dir, min_images=50)

    # Load the dataset with only valid classes for splitting
    full_dataset = FilteredImageFolder(root=root_dir, valid_classes=valid_classes, seed=seed)

    # Split the dataset into train, validation, and test sets
    train_size = int(split_ratio[0] * len(full_dataset))
    val_size = int(split_ratio[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Now apply the correct transformations to each subset
    train_dataset.dataset.transform = train_transform  # Apply train augmentations to training data
    val_dataset.dataset.transform = val_test_transform  # No augmentations for validation
    test_dataset.dataset.transform = val_test_transform  # No augmentations for testing

    class_count_new = {}
    for key,value in full_dataset.class_to_idx.items():
        if key in class_count:
            class_count_new[value]=class_count[key]
    # Step 5: Create DataLoaders for each subset
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Step 6: Display a summary of the data split
    print(f"Number of training images: {len(train_dataset)}")
    print(f"Number of validation images: {len(val_dataset)}")
    print(f"Number of test images: {len(test_dataset)}")
    return train_loader, val_loader, test_loader,valid_classes,class_count_new 

if __name__ == '__main__':
    root_dir = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/eardrumDs_kaggle'
    train_loader, val_loader, test_loader, valid_classes, dataset = build_dataloader(root_dir)
# %%
