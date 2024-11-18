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







# import numpy as np
# from scipy.ndimage import label
# from sklearn.metrics import jaccard_score
# def calculate_dice(mask1, mask2):
#     intersection = np.logical_and(mask1, mask2)
#     dice = 2 * np.sum(intersection) / (np.sum(mask1) + np.sum(mask2))
#     return dice
# def calculate_metrics(true_mask, pred_mask, iou_threshold=0.1):
#     threshold = 0.5 # Set your desired threshold value

#     true_mask = true_mask > threshold
#     pred_mask = pred_mask > threshold

#     true_labels, _ = label(true_mask)
#     pred_labels, _ = label(pred_mask)

#     true_positives = 0
#     false_negatives = 0
#     visited = set()

#     for j in range(1, np.max(true_labels) + 1):
#         true_region = true_labels == j
#         match_found = False

#         for k in range(1, np.max(pred_labels) + 1):
#             if k in visited:
#                 continue

#             pred_region = pred_labels == k

#             true_region_flat = true_region.flatten()
#             pred_region_flat = pred_region.flatten()

#             iou = jaccard_score(true_region_flat, pred_region_flat)

#             if iou > iou_threshold:
#                 true_positives += 1
#                 match_found = True
#                 visited.add(k)
#                 break

#         if not match_found:
#             false_negatives += 1

#     dice_coefficient = true_positives / (true_positives + false_negatives)

#     return dice_coefficient
# def calculate_metrics(true_mask, pred_mask, iou_threshold=0.1):
#     threshold = 0.5  # Set your desired threshold value
#     # print(np.max(true_mask))
#     # print(np.max(pred_mask))
#     true_mask = true_mask > threshold
#     pred_mask = pred_mask > threshold

#     true_labels, _ = label(true_mask)
#     pred_labels, _ = label(pred_mask)

#     true_positives = 0
#     false_positives = 0
#     false_negatives = 0
#     visited = set()

#     for j in range(1, np.max(true_labels) + 1):
#         true_region = true_labels == j
#         match_found = False

#         for k in range(1, np.max(pred_labels) + 1):
#             if k in visited:
#                 continue

#             pred_region = pred_labels == k

#             true_region_flat = true_region.flatten()
#             pred_region_flat = pred_region.flatten()

#             iou = jaccard_score(true_region_flat, pred_region_flat)

#             if iou > iou_threshold:
#                 true_positives += 1
#                 match_found = True
#                 visited.add(k)
#                 break

#         if not match_found:
#             false_negatives += 1

#     # false_positives = len(np.unique(pred_labels)) - len(visited)
#     # print(true_positives)
#     # print(false_positives)
#     dice_coefficient = true_positives /  true_positives + false_negatives
    
#     return dice_coefficient






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
# import cv2
# import numpy as np
# def calculate_dice_with_location(mask_pred, mask_gt):
#     intersection = np.logical_and(mask_pred, mask_gt)
#     dice_with_location = 2 * np.sum(intersection) / (np.sum(mask_pred) + np.sum(mask_gt))
#     return dice_with_location


def convert_tensor_to_image(tensor, data_norm):
    tensor = tensor.cpu().numpy()  # Move tensor to CPU and convert to numpy array
    tensor = np.transpose(tensor, (1, 2, 0))  # Transpose dimensions
    tensor = tensor * data_norm['inp']['div'][0] + data_norm['inp']['sub'][0]  # Apply inverse normalization
    tensor = np.clip(tensor, 0, 1)  # Clip values between 0 and 1
    tensor = tensor * 255  # Scale the values to the 0-255 range
    tensor = tensor.astype(np.uint8)  # Convert to unsigned integer
    return tensor


def eval_test_wo_gt(loader, model, data_norm=None, save_dir=None):
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
                    name = loader.dataset.dataset.dataset_1.files[cnt];
                
                    name = re.sub('./../data/All_video_frames/AOM/AM355L/','',name)
                
                    pred_mask_path = os.path.join(pred_save_dir, f'predicted_mask_{cnt}-{p}-{name}.png')
                    # cv2.imwrite(pred_mask_path, mask_pred * 255) 
                    pred_mask_tensor = torch.tensor(mask_pred * 255, dtype=torch.uint8)
                    pred_mask_pil = tensor2PIL(pred_mask_tensor)
                    pred_mask_np = np.array(pred_mask_pil)
                    print(f"pred_mask_path:{pred_mask_path}")
                    cv2.imwrite(pred_mask_path, pred_mask_np)
                    #pdb.set_trace()
                    # cv2.imwrite(pred_mask_path, pred_mask_pil) 
            # Multiply by 255 to get pixel values in the range [0, 255]

                    original_image_with_contours = draw_contours(original_image_with_contours, mask_pred, (0, 0, 255))
                    # Save the modified image
                    save_path = os.path.join(save_dir, f'image_{cnt}-{p}-{name}.png')
                    cv2.imwrite(save_path, original_image_with_contours)
               
        cnt += 1
        #average_dice = sum(dice_coefficients) / len(dice_coefficients)

    return 1 #val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), val_iou.item(), average_dice
#---------------------------------------------------------------------------------------------------------------
def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False, save_dir=None):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()
    val_iou = utils.Averager()
    val_dice = utils.Averager()

    # os.makedirs(save_dir, exist_ok=True)  # Create the save directory if it doesn't exist
     # Create a separate directory for original images
    # orig_save_dir = os.path.join(save_dir, 'original_images')
    # os.makedirs(orig_save_dir, exist_ok=True)
    pred_save_dir = os.path.join(save_dir, 'predicted_masks')
    os.makedirs(pred_save_dir, exist_ok=True)
    
    dice_coefficients = [] 
    pbar = tqdm(loader, leave=False, desc='val')
    cnt = 0
    for batch in pbar:
        with torch.no_grad():
            for k, v in batch.items():
                batch[k] = v.to(device)

            inp = batch['inp']
            pred = torch.sigmoid(model.infer(inp))

            result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
            val_metric1.add(result1.item(), inp.shape[0])
            val_metric2.add(result2.item(), inp.shape[0])
            val_metric3.add(result3.item(), inp.shape[0])
            val_metric4.add(result4.item(), inp.shape[0])

            if verbose:
                pbar.set_description('val {} {:.4f}'.format(metric1, val_metric1.item()))
                pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
                pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
                pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))
            for p in range(pred.shape[0]):
                original_image = convert_tensor_to_image(batch['inp'][p], data_norm)  # Convert to numpy array
                mask_pred = (pred[p] > 0.5).cpu().numpy().astype(np.uint8)
                mask_gt = batch['gt'][p].cpu().numpy().astype(np.uint8)
                #print(type(mask_pred))
                #print(type(mask_gt))
            

                iou = calculate_iou(mask_pred, mask_gt)
                #dice = calculate_dice(mask_pred, mask_gt)
                val_iou.add(iou, 1)
               # val_dice.add(dice, 1)
                dice= calculate_metrics(mask_gt, mask_pred)
                # val_dice.add(dice_with_location, 1)
                dice_coefficients.append(dice) 
                # import random
                original_image_with_contours = original_image.copy()
                # print(original_image_with_contours)  # Create a copy to draw contours on
                # original_image_with_contours = draw_contours(original_image_with_contours, mask_gt, (0, 255, 0), thickness=2)
                original_image_with_contours = draw_contours(original_image_with_contours, mask_gt, (0, 255, 0))
            # Save the predicted mask directly as a binary image
                pred_mask_path = os.path.join(pred_save_dir, f'predicted_mask_{cnt}-{p}.png')
                # cv2.imwrite(pred_mask_path, mask_pred * 255) 
                pred_mask_tensor = torch.tensor(mask_pred * 255, dtype=torch.uint8)
                pred_mask_pil = tensor2PIL(pred_mask_tensor)
                pred_mask_np = np.array(pred_mask_pil)
                cv2.imwrite(pred_mask_path, pred_mask_np)

                # cv2.imwrite(pred_mask_path, pred_mask_pil) 
 # Multiply by 255 to get pixel values in the range [0, 255]

                original_image_with_contours = draw_contours(original_image_with_contours, mask_pred, (0, 0, 255))
                # pred_mask_path = os.path.join(pred_save_dir, f'predicted_mask_{cnt}-{p}.png')
                # cv2.imwrite(pred_mask_path, mask_pred * 255)
                # Save the modified image
                save_path = os.path.join(save_dir, f'image_{cnt}-{p}.png')
                cv2.imwrite(save_path, original_image_with_contours)
               
        cnt += 1
        average_dice = sum(dice_coefficients) / len(dice_coefficients)

    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(), val_iou.item(), average_dice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    #dataset = datasets.make(spec['wrapper'], args=spec.args)
    
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)
    #pdb.set_trace()
    
    model = models.make(config['model']).to(device)
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=True)
    save_dir = './contours/'
    # os.makedirs(pred_save_dir, exist_ok=True)
    eval_test_wo_gt(loader, model, data_norm=config.get('data_norm'), save_dir=save_dir)
    # metric1, metric2, metric3, metric4, iou, dice = eval_psnr(loader, model,
                                                               # data_norm=config.get('data_norm'),
                                                               # eval_type=config.get('eval_type'),
                                                               # eval_bsize=config.get('eval_bsize'),
                                                               # verbose=True,
                                                               # save_dir=save_dir)
    # print('metric1: {:.4f}'.format(metric1))
    # print('metric2: {:.4f}'.format(metric2))
    # print('metric3: {:.4f}'.format(metric3))
    # print('metric4: {:.4f}'.format(metric4))
    # print(iou)
    # print(dice)
