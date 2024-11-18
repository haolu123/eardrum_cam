import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import argparse
import os

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2
import datasets
import models
import utils
import numpy as np
from torchvision import transforms
from mmcv.runner import load_checkpoint
import datasets
import models
import utils
import pdb
import re
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


def tensor2PIL(tensor):
    """
    Convert a tensor to a PIL image.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        PIL.Image: PIL image.
    """
    img = transforms.ToPILImage()(tensor)
    return img
# import numpy as np
# from scipy.ndimage import label
# from sklearn.metrics import jaccard_score


import numpy as np
from scipy.ndimage import label
from sklearn.metrics import jaccard_score
import torch
from torch import Tensor

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import jaccard_score
import torch
from torch import Tensor

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import jaccard_score
import torch
from torch import Tensor
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

def dice_multi(input: Tensor, targs: Tensor, iou: bool = False, eps: float = 1e-8) -> Tensor:
    n = targs.shape[0]
    targs = targs.squeeze(1)
    input = input.to(torch.int)  # Convert boolean tensor to integer tensor
    input = input.view(n, -1)  # Flatten the input tensor
    targs = targs.view(n, -1)  # Flatten the target tensor
    targs1 = (targs > 0).float()
    input1 = (input > 0).float()
    ss = (input == targs).float()
    intersect = (ss * targs1).sum(dim=1).float()
    union = (input1 + targs1).sum(dim=1).float()
    
    if not iou:
        dice_coefficient = 2. * intersect / (union + eps)
    else:
        dice_coefficient = intersect / (union - intersect + eps)
    
    dice_coefficient[union == 0.] = 1.
    return dice_coefficient.mean()

def calculate_metrics(true_mask, pred_mask, iou_threshold=0.1):
    threshold = 0.5  # Set your desired threshold value

    true_mask = true_mask > threshold
    pred_mask = pred_mask > threshold

    true_labels, _ = label(true_mask)
    pred_labels, _ = label(pred_mask)

    true_positives = 0
    false_negatives = 0
    visited = set()

    for j in range(1, np.max(true_labels) + 1):
        true_region = true_labels == j
        match_found = False

        for k in range(1, np.max(pred_labels) + 1):
            if k in visited:
                continue

            pred_region = pred_labels == k

            true_region_flat = true_region.flatten()
            pred_region_flat = pred_region.flatten()

            iou = jaccard_score(true_region_flat, pred_region_flat)

            if iou > iou_threshold:
                true_positives += 1
                match_found = True
                visited.add(k)
                break

        if not match_found:
            false_negatives += 1

    dice_coefficient = dice_multi(torch.tensor(pred_mask), torch.tensor(true_mask))

    return dice_coefficient.item()


def draw_contours(image, mask, color):
    mask = np.squeeze(mask)  # Remove singleton dimensions if present

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image  # Return the original image if no contours are found

    # Convert image to RGB format if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Draw the contours on the image
    cv2.drawContours(image, contours, -1, color, 4)

    return image


def convert_tensor_to_image(tensor, data_norm):
    tensor = tensor.cpu().numpy()  # Move tensor to CPU and convert to numpy array
    tensor = np.transpose(tensor, (1, 2, 0))  # Transpose dimensions
    tensor = tensor * data_norm['inp']['div'][0] + data_norm['inp']['sub'][0]  # Apply inverse normalization
    tensor = np.clip(tensor, 0, 1)  # Clip values between 0 and 1
    tensor = tensor * 255  # Scale the values to the 0-255 range
    tensor = tensor.astype(np.uint8)  # Convert to unsigned integer
    return tensor


def eval_test_wo_gt(loader, model, img_size, data_norm=None, save_dir=None, org_dir = None):
    transform_gray = transforms.Grayscale()
    tensor2pil = transforms.ToPILImage()
    pil2tensor = transforms.ToTensor()
    
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]}
        }
    pred_save_dir = os.path.join(save_dir, 'predicted_masks')
    os.makedirs(pred_save_dir, exist_ok=True)
    
    
    pbar = tqdm(loader, leave=False, desc='val')
    cnt = 0
    width, height = img_size
    cumulative_mask = None
    for batch in pbar:
        with torch.no_grad():
            #for i, (batch, fl) in enumerate(loader):
            for k, v in batch.items():
                batch[k]= v.to(device)

            inp = batch['inp']
            pred = torch.sigmoid(model.infer(inp))
            
            for p in range(pred.shape[0]):
                original_image = convert_tensor_to_image(batch['inp'][p], data_norm)  # Convert to numpy array
                mask_pred = (pred[p] > 0.5).cpu().numpy().astype(np.uint8)
                
                num_org = (pil2tensor(transform_gray(tensor2pil(original_image)))>10).sum()
                num_mask = (mask_pred>0).sum()
                ratio_msk_org = num_mask/num_org 
                #pdb.set_trace()
                if ratio_msk_org > 0.35:
                    original_image_with_contours = original_image.copy()
                    file_path = loader.dataset.dataset.dataset_1.files[cnt]

                    name = re.sub('./../data/All_video_frames/AOM/AM355L/','',file_path)
                    img = Image.open(file_path)
                    img_size = img.size
                    pred_mask_path = os.path.join(pred_save_dir, f'predicted_mask_{name}')
                    # cv2.imwrite(pred_mask_path, mask_pred * 255) 
                    pred_mask_tensor = torch.tensor(mask_pred * 255, dtype=torch.uint8)
                    pred_mask_pil = tensor2PIL(pred_mask_tensor)
                    pred_mask_pil = pred_mask_pil.resize(img_size, resample=Image.NEAREST)
                    pred_mask_np = np.array(pred_mask_pil)
                    cv2.imwrite(pred_mask_path, pred_mask_np)

                    img_np = np.array(img)
                    mask_binary = pred_mask_np // 255
                    def convex_hull_image(data):
                        region = np.argwhere(data)
                        hull = ConvexHull(region)
                        verts = [(region[v,0], region[v,1]) for v in hull.vertices]
                        img = Image.new('L', data.shape, 0)
                        ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
                        mask = np.array(img)

                        return mask.T
                    try:
                        hull_mask_np = convex_hull_image(mask_binary)
                    except:
                        try: 
                            if hull_mask_np is not None:
                                hull_mask_np = hull_mask_np
                        except:
                            hull_mask_np = mask_binary
                    blurred_mask = cv2.GaussianBlur(hull_mask_np.astype(np.float32), (21, 21), 11)
                    if cumulative_mask is None:
                        cumulative_mask = blurred_mask
                    else:
                        cumulative_mask = 0.15 * blurred_mask + 0.75 * cumulative_mask
                    stable_mask = (cumulative_mask > 0.5).astype(np.uint8)

                    masked_img_np = img_np * stable_mask[:,:,None]
                    masked_img_np = masked_img_np.astype(np.uint8)
                    masked_img_pil = Image.fromarray(masked_img_np)
                    masked_img_pil.save(os.path.join(save_dir, f'masked_image_{name}'))
                    #pdb.set_trace()
                    # cv2.imwrite(pred_mask_path, pred_mask_pil) 
            # Multiply by 255 to get pixel values in the range [0, 255]

                    # original_image_with_contours = draw_contours(original_image_with_contours, mask_pred, (0, 0, 255))
                    # # Save the modified image
                    # save_path = os.path.join(save_dir, f'image_{cnt}-{p}-{name}.png')
                    # cv2.imwrite(save_path, original_image_with_contours)
               
        cnt += 1
        #average_dice = sum(dice_coefficients) / len(dice_coefficients)

    return 1 #val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), val_iou.item(), average_dice
#---------------------------------------------------------------------------------------------------------------
def numerical_sort(value):
    parts = re.split(r'(\d+)', value)
    return [int(part) if part.isdigit() else part for part in parts]

def create_video_from_images(image_folder, output_video_path, fps=30):
    # Get all the image file names in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    
    # Sort images numerically
    images = sorted(images, key=numerical_sort)

    # Get the size of the images
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs as well
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']

    model = models.make(config['model']).to(device)
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)

    dataset_root_path = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/data/All_video_frames"
    class_labels = sorted(['Normal', 'AOM', 'Effusion', 'Retraction', 'Perforation', 'Tube', 'Tympanosclerosis'])
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Iterate over each class directory
    for class_label in class_labels:
        class_dir = os.path.join(dataset_root_path, class_label)
        if not os.path.exists(class_dir):
            continue
        # Iterate over each subdirectory in the class directory
        for subdir in os.listdir(class_dir):
            subdir_path = os.path.join(class_dir, subdir)
            if os.path.isdir(subdir_path):
                # Check if the subdirectory contains .png files
                png_files = [file for file in os.listdir(subdir_path) if file.endswith('.png')]
                if png_files:
                    first_png_file = png_files[0]
                    image_path = os.path.join(subdir_path, first_png_file)
                    with Image.open(image_path) as img:
                        size_org = img.size

                    spec['dataset']['root_path'] = subdir_path

                    dataset = datasets.make(spec['dataset'])
                    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
                    #dataset = datasets.make(spec['wrapper'], args=spec.args)
                    
                    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                                        num_workers=8)
                    #pdb.set_trace()
    
    
                    save_dir = os.path.join("/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/data/All_video_frames_fm_mask",class_label,subdir)
                    os.makedirs(save_dir, exist_ok=True)
                    # os.makedirs(pred_save_dir, exist_ok=True)
                    print(f"Processing {subdir_path}")
                    eval_test_wo_gt(loader, model, size_org, data_norm=config.get('data_norm'), save_dir=save_dir, org_dir = LH)
                    image_folder = save_dir
                    output_video_path = os.path.join("/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project4_ear/data/All_vide_fm_mask",class_label, f'{subdir}.mp4')
                    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                    create_video_from_images(image_folder, output_video_path, fps=30)