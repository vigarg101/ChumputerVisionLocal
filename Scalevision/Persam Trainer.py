import concurrent.futures
import optuna
from typing import ByteString, Callable, Optional, Tuple, Union, Any, Dict, List
from functools import partial
import cv2
import numpy as np
from pathlib import Path
from per_segment_anything import SamPredictor, sam_model_registry
from torch.nn import functional as F
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm


tqdm.pandas()


class Mask_Weights(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


def grab_filenames(directory) -> tuple[list[str], list[str]]:
    directory = Path(directory)
    image_filenames = []
    mask_filenames = []

    for ext in ["*.jpg", "*.png"]:
        for file in directory.glob(ext):
            mask_file = file.with_stem(f"{file.stem}_mask_bw").with_suffix(".png")
            if mask_file.exists():
                image_filenames.append(str(file))
                mask_filenames.append(str(mask_file))

    return image_filenames, mask_filenames


def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()

    return topk_xy, topk_label


def calculate_dice_loss(inputs, targets, num_masks=1):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def calculate_sigmoid_focal_loss(inputs, targets, num_masks=1, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t)**gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks


def calculate_accuracy(pred, target):
    pred = torch.sigmoid(pred)  # apply sigmoid to the prediction logits
    pred = (pred > 0.5).float()  # binarize the predictions using a threshold of 0.5
    correct = (pred == target).float()  # compute element-wise equality and convert to float
    accuracy = correct.sum() / correct.numel()  # calculate the mean accuracy
    return accuracy.item()


def compute_prior(ref_images, ref_masks):
    print("======> Obtain Self Location Prior")

    prior_data = {}

    for i, (image, mask) in enumerate(zip(ref_images, ref_masks)):
        # Image features encoding
        mask = predictor.set_image(image, mask)
        ref_feat = predictor.features.squeeze().permute(1, 2, 0)

        mask = F.interpolate(mask, size=ref_feat.shape[:2], mode="bilinear")
        mask = mask.squeeze()[0]

        # Target feature extraction
        target_feat = ref_feat[mask > 0]
        target_feat_mean = target_feat.mean(0)
        target_feat_max = torch.max(target_feat, dim=0)[0]
        target_feat = (target_feat_max / 2 + target_feat_mean / 2).unsqueeze(0)

        # Cosine similarity
        h, w, C = ref_feat.shape
        target_feat = target_feat / target_feat.norm(dim=-1, keepdim=True)
        ref_feat = ref_feat / ref_feat.norm(dim=-1, keepdim=True)
        ref_feat = ref_feat.permute(2, 0, 1).reshape(C, h * w)
        sim = target_feat @ ref_feat

        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(sim,
                                                input_size=predictor.input_size,
                                                original_size=predictor.original_size).squeeze()

        # Positive location prior
        topk_xy, topk_label = point_selection(sim, topk=1)

        # Store the computed values in the dictionary
        prior_data[i] = (ref_feat, target_feat, sim, topk_xy, topk_label)

    return prior_data


def log_trial(trial_idx, train_idx, train_epoch, txt_summary):
    with open(f'logs/trial_{trial_idx}_log.txt', 'a+') as f:
        f.write(f'Train Epoch: {train_idx} / {train_epoch}\n')
        f.write(f'{txt_summary}\n')


def objective(trial, *args, **kwargs):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    train_epoch = trial.suggest_int('train_epoch', 100, 5000)
    eps = trial.suggest_float('eps', 1e-8, 1e-2, log=True)
    log_epoch = 50

    best_val_loss = float('inf')

    print('======> Start Training')
    mask_weights = Mask_Weights().cuda()
    mask_weights.train()

    optimizer = torch.optim.AdamW(mask_weights.parameters(), lr=lr, eps=eps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=train_epoch)

    for train_idx in range(train_epoch):

        total_dice_loss = 0.0
        total_focal_loss = 0.0
        total_dice_loss_val = 0.0
        total_focal_loss_val = 0.0
        total_accuracy = 0.0
        total_accuracy_val = 0.0
        total_val_loss = 0.0
        total_loss = 0.0

        for i, (ref_image, gt_mask) in enumerate(zip(ref_images_train, gt_masks_train)):
            _, _, _, topk_xy, topk_label = prior_data_train[i]
            # Run the decoder
            masks, scores, logits, logits_high = predictor.predict(point_coords=topk_xy,
                                                                    point_labels=topk_label,
                                                                    multimask_output=True)
            logits_high = logits_high.flatten(1)

            # Weighted sum three-scale masks
            weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
            logits_high = logits_high * weights
            logits_high = logits_high.sum(0).unsqueeze(0)

            dice_loss = calculate_dice_loss(logits_high, gt_mask)
            focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
            loss = dice_loss + focal_loss

            total_loss += loss
            total_dice_loss += dice_loss.item()
            total_focal_loss += focal_loss.item()

            total_accuracy += calculate_accuracy(logits_high, gt_mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation loop

        mask_weights.eval()
        with torch.no_grad():
            # After a training step, perform a validation step
            for i, (ref_image, gt_mask) in enumerate(zip(ref_images_val, gt_masks_val)):
                _, _, _, topk_xy, topk_label = prior_data_val[i]
                # Run the decoder
                masks, scores, logits, logits_high = predictor.predict(point_coords=topk_xy,
                                                                        point_labels=topk_label,
                                                                        multimask_output=True)
                logits_high = logits_high.flatten(1)

                # Weighted sum three-scale masks
                weights = torch.cat((1 - mask_weights.weights.sum(0).unsqueeze(0), mask_weights.weights), dim=0)
                logits_high = logits_high * weights
                logits_high = logits_high.sum(0).unsqueeze(0)

                dice_loss = calculate_dice_loss(logits_high, gt_mask)
                focal_loss = calculate_sigmoid_focal_loss(logits_high, gt_mask)
                loss = dice_loss + focal_loss

                total_dice_loss_val += dice_loss.item()
                total_focal_loss_val += focal_loss.item()
                total_val_loss += loss.item()
                total_accuracy_val += calculate_accuracy(logits_high, gt_mask)

        mask_weights.train()
        scheduler.step()

        avg_loss = total_loss / len(gt_masks_train)
        avg_val_loss = total_val_loss / len(gt_masks_val)
        avg_dice_loss = total_dice_loss / len(gt_masks_train)
        avg_dice_loss_val = total_dice_loss_val / len(gt_masks_val)
        avg_focal_loss = total_focal_loss / len(gt_masks_train)
        avg_focal_loss_val = total_focal_loss_val / len(gt_masks_val)
        avg_accuracy = total_accuracy / len(gt_masks_train)
        avg_accuracy_val = total_accuracy_val / len(gt_masks_val)

        target_feats = []
        for key in prior_data_train:
            _, target_feat, _, _, _ = prior_data_train[key]
            target_feats.append(target_feat)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # save the best model and associated data
            torch.save(mask_weights.state_dict(), f'crop_vision/mask_weights_{trial.number}.pth')
            torch.save(target_feats, f'crop_vision/target_feats_{trial.number}.pth')
            torch.save(weights, f'crop_vision/weights_{trial.number}.pth')

        if train_idx % log_epoch == 0:
            current_lr = scheduler.get_last_lr()[0]
            txt_loss = f'Loss(t:{avg_loss:.4f}, v:{avg_val_loss:.4f})'
            txt_dice = f'Dice(t:{avg_dice_loss:.4f}, v:{avg_dice_loss_val:.4f})'
            txt_focal = f'Focal(t:{avg_focal_loss:.4f}, v:{avg_focal_loss_val:.4f})'
            txt_accuracy = f'Acc(t:{avg_accuracy:.4f}, v:{avg_accuracy_val:.4f})'
            txt_summary = f'LR: {current_lr:.6f},', txt_loss, txt_dice, txt_focal, txt_accuracy

            log_trial(
                trial_idx=trial.number,
                train_idx=train_idx,
                train_epoch=train_epoch,
                txt_summary=txt_summary,
            )

        # Report intermediate objective value.
        trial.report(avg_val_loss, step=train_idx)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Print trial log file
    filename = f'logs/trial_{trial.number}_log.txt'
    with open(filename, 'r') as f:
        print(f.read())

    return best_val_loss


if __name__ == "__main__":
    reference_dir = Path('mask_and_images')
    # initialize lists for prompt images and targets
    reference_image_files, reference_mask_files = grab_filenames(reference_dir)
    # Grab the first 10 images and masks for testing purporses
    reference_image_files = reference_image_files[:10]
    reference_mask_files = reference_mask_files[:10]

    print(f'{len(reference_image_files)} images and {len(reference_mask_files)} masks found.')

    ref_images = []
    ref_masks = []
    size = 1024
    for ref_image_file, ref_mask_file in zip(reference_image_files, reference_mask_files):
        ref_image = cv2.imread(ref_image_file)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        ref_image = cv2.resize(ref_image, (size, size))
        ref_images.append(ref_image)

        ref_mask = cv2.imread(ref_mask_file)
        ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
        ref_mask = cv2.resize(ref_mask, (size, size))
        ref_masks.append(ref_mask)

    # Load images and masks
    gt_masks = []
    for ref_mask in ref_masks:
        gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0
        gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()
        gt_masks.append(gt_mask)

    # Split the data into train and validation sets
    train_ratio = 0.75
    ref_images_train, ref_images_val, ref_masks_train, ref_masks_val, gt_masks_train, gt_masks_val = train_test_split(
        ref_images, ref_masks, gt_masks, train_size=train_ratio, random_state=42)

    arguments = {
        "ref_images_train": ref_images_train,
        "ref_images_val": ref_images_val,
        "ref_masks_train": ref_masks_train,
        "ref_masks_val": ref_masks_val,
        "gt_masks_train": gt_masks_train,
        "gt_masks_val": gt_masks_val
    }

    print(f'Train set: {len(ref_images_train)} images and masks.')
    print(f'Validation set: {len(ref_images_val)} images and masks.')

    print("======> Load SAM")
    sam_type, sam_ckpt = 'vit_h', './sam_vit_h_4b8939.pth'
    sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    for name, param in sam.named_parameters():
        param.requires_grad = False
    predictor = SamPredictor(sam)

    model_path = Path('crop_vision')

    model_path.mkdir(parents=True, exist_ok=True)

    prior_data_train = compute_prior(ref_images_train, ref_masks_train)
    prior_data_val = compute_prior(ref_images_val, ref_masks_val)

    arguments['prior_data_train'] = prior_data_train
    arguments['prior_data_val'] = prior_data_val

    study = optuna.create_study(direction='minimize')
    study.optimize(partial(objective, **arguments), n_trials=100, n_jobs=-1)

    print('Best trial:')
    trial_ = study.best_trial
    print(f'  Value: {trial_.value}')
    print('  Params: ')
    for key, value in trial_.params.items():
        print(f'    {key}: {value}')
