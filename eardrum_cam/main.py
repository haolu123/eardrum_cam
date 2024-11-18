#%%
from dataset import build_dataloader
from models import models
from train_func import train_model, evaluate_model
from utils import get_valid_classes
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.transforms import v2
#%%
root_dir = '/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/project_inherit/Data/2019_2021/frames_by_label'
split_ratio=(0.70, 0.15)
batch_size=80
num_workers=1
lr = 2.5e-05
weight_decay=0.0017

num_epochs = 20
patience=5
tolerence=0.05
momentum=0.9

scheduler_flag = False
factor_schedule=0.1
patience_schedule=3

weight_loss_flag = False
mixup_flag = False


train_loader, val_loader, test_loader,valid_classes, class_counts  = build_dataloader(root_dir, split_ratio=split_ratio, batch_size=batch_size, num_workers=num_workers)

if mixup_flag:
    NUM_CLASSES = len(valid_classes)
    cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    mixup = v2.MixUp(num_classes=NUM_CLASSES)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
else:
    cutmix_or_mixup = None

if weight_loss_flag:
    total_samples = sum(class_counts.values())
    # Step 2: Calculate class weights: total_samples / (number of samples in each class)
    class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
    weights = [class_weights[cls] for cls in sorted(class_counts.keys())]
    weights_tensor = torch.tensor(weights, dtype=torch.float).to('cuda')
else:
    weights_tensor = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model: ResNet50 (adjusted for number of classes)
num_classes = len(valid_classes)  # Assuming valid_classes are defined as in the previous example
model_name = 'resnet34'
model = models.resnet34(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define the criterion, optimizer, and scheduler
if mixup_flag:
    criterion = nn.BCEWithLogitsLoss(weight=weights_tensor)
else:
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
# Define the scheduler (e.g., ReduceLROnPlateau)
if scheduler_flag:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor_schedule, patience=patience_schedule, verbose=True)
else:
    scheduler = None

param_str = f'{model_name}_bs_{batch_size}_lr_{lr}_epoch_{num_epochs}_wd_{weight_decay}_wlf_{weight_loss_flag}'
# Train the model
trained_model, train_loss, val_loss, best_acc = train_model(model, \
                                                train_loader, val_loader,\
                                                criterion, optimizer,\
                                                num_epochs, device=device, \
                                                patience=patience, tolerence=tolerence, \
                                                scheduler=scheduler, cutmix_or_mixup=cutmix_or_mixup,\
                                                NUM_CLASSES = num_classes, param_str = param_str
                                                )

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
torch.save(trained_model.state_dict(), 'best_resnet34_eardrum.pth')

# %%
