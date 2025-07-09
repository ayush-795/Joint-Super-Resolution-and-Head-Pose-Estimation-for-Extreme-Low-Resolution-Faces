import os
import zipfile
import math
import numpy as np
import cv2
from itertools import chain
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

# Additional imports for evaluation metrics:
from pytorch_msssim import ssim  # pip install pytorch-msssim
import lpips  # pip install lpips

#############################################
# 0. Unzip Datasets (if not already extracted)
#############################################
hr_extract_dir = r"C:\Users\welcome\Desktop\sop\CelebA_128x128_HR"
lr_extract_dir = r"C:\Users\welcome\Desktop\sop\CelebA_8x8_LR"

#############################################
# 1. Define Bottleneck Block for Hopenet
#############################################
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

#############################################
# 2. Hopenet Definition (with backbone method)
#############################################
class Hopenet(nn.Module):
    # Hopenet with 3 output layers for yaw, pitch and roll.
    def __init__(self, block, layers, num_bins):
        self.inplanes = 64
        super(Hopenet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_yaw = nn.Linear(512 * block.expansion, num_bins)
        self.fc_pitch = nn.Linear(512 * block.expansion, num_bins)
        self.fc_roll = nn.Linear(512 * block.expansion, num_bins)

        # Vestigial layer from previous experiments
        self.fc_finetune = nn.Linear(512 * block.expansion + 3, 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def backbone(self, x):
        """Return features from the penultimate layer."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        pre_yaw = self.fc_yaw(x)
        pre_pitch = self.fc_pitch(x)
        pre_roll = self.fc_roll(x)
        return pre_yaw, pre_pitch, pre_roll

#############################################
# 3. Dataset Definition (Pre-degraded LR and HR images)
#############################################
class FaceSRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, img_size=128):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_files = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir)
                                if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        self.lr_files = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir)
                                if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # Normalize to [-1, 1]
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.hr_files)

    def __getitem__(self, idx):
        hr_img = cv2.imread(self.hr_files[idx])
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2RGB)
        lr_img = cv2.imread(self.lr_files[idx])
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
        return {
            'lr': self.transform(lr_img),
            'hr': self.transform(hr_img)
        }

#############################################
# 4. Generator (MSGAN) with Corrected Upsampling
#############################################
class RRDB(nn.Module):
    def __init__(self, channels):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.lrelu(out)
        out = self.conv2(out)
        return x + out * 0.2

class MSGenerator(nn.Module):
    def __init__(self):
        super(MSGenerator, self).__init__()
        self.initial = nn.Conv2d(3, 64, 3, 1, 1)
        self.rrdb = nn.Sequential(*[RRDB(64) for _ in range(16)])

        # Upsampling stages
        self.up1 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2)
        )

        # Add RGB conversion layers for intermediate scales
        self.rgb_s1 = nn.Conv2d(64, 3, 3, 1, 1)
        self.rgb_s2 = nn.Conv2d(64, 3, 3, 1, 1)

        self.final = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        feats = self.rrdb(x)
        s1_feat = self.up1(feats)      # Feature map at scale 1
        s1 = self.rgb_s1(s1_feat)        # Convert to RGB (3 channels)

        s2_feat = self.up2(s1_feat)      # Feature map at scale 2
        s2 = self.rgb_s2(s2_feat)        # Convert to RGB (3 channels)

        s3 = self.up3(s2_feat)           # Feature map for final output
        out = self.final(s3)             # Final RGB image output
        return [s1, s2, out]

#############################################
# 5. Multi-Scale Discriminator with Spectral Normalization
#############################################
class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
                nn.LeakyReLU(0.2),
                nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.utils.spectral_norm(nn.Conv2d(256, 1, 4, 1, 1))
            ) for _ in range(3)  # Three stages for the multi-scale approach
        ])

    def forward(self, x, stage):
        return self.discriminators[stage](x)

#############################################
# 6. Loss Functions including Head Pose Loss
#############################################
class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.λ_adv = 0.01
        self.λ_l1 = 1.0
        self.λ_lpips = 0.001
        self.λ_ip = 0.1
        self.λ_sim = 0.2
        self.λ_norm = 0.01
        self.λ_est = 1.0

        # VGG-based perceptual loss (using a partial VGG16)
        self.vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True).features[:16].to(device)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, sr_list, hr):
        losses = {}
        # Reconstruction loss (L1) on final output
        losses['l1'] = F.l1_loss(sr_list[-1], hr)
        # Perceptual loss using VGG features
        vgg_sr = self.vgg(sr_list[-1])
        vgg_hr = self.vgg(hr)
        losses['lpips'] = F.l1_loss(vgg_sr, vgg_hr)
        # Identity preservation loss via cosine similarity (FaceNet)
        ip_sr = facenet(sr_list[-1])
        ip_hr = facenet(hr)
        losses['ip'] = 1 - F.cosine_similarity(ip_sr, ip_hr).mean()
        # Pose-aware representation alignment losses using Hopenet's backbone features
        sr_feats = hopenet.backbone(sr_list[-1])
        hr_feats = hopenet.backbone(hr)
        losses['sim'] = 1 - F.cosine_similarity(F.normalize(sr_feats, dim=1),
                                                F.normalize(hr_feats, dim=1)).mean()
        losses['norm'] = F.mse_loss(sr_feats.norm(dim=1), hr_feats.norm(dim=1))
        # Head Pose Estimation loss:
        with torch.no_grad():
            hr_angles = hopenet(hr)
        sr_angles = hopenet(sr_list[-1])
        hr_angles_cat = torch.cat(hr_angles, dim=1)
        sr_angles_cat = torch.cat(sr_angles, dim=1)
        losses['hpe'] = F.mse_loss(sr_angles_cat, hr_angles_cat)

        total_loss = (self.λ_l1 * losses['l1'] +
                      self.λ_lpips * losses['lpips'] +
                      self.λ_ip * losses['ip'] +
                      self.λ_sim * losses['sim'] +
                      self.λ_norm * losses['norm'] +
                      self.λ_est * losses['hpe'])
        return total_loss, losses

#############################################
# 7. Instantiate Models and Optimizers
#############################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Face recognition model for identity loss
from facenet_pytorch import InceptionResnetV1
facenet = InceptionResnetV1(pretrained='vggface2').to(device).eval()

# Instantiate Hopenet with Bottleneck, layers similar to ResNet-50 ([3, 4, 6, 3]), and 66 bins.
num_bins = 66
hopenet = Hopenet(Bottleneck, [3, 4, 6, 3], num_bins).to(device)
# === CHANGE THIS PATH to point to your local hopenet weights file ===
hopenet_weights_path = r"C:\Users\welcome\Desktop\sop\hopenet_alpha2.pkl"
hopenet.load_state_dict(torch.load(hopenet_weights_path, map_location=device))
hopenet.eval()

# Freeze Hopenet and FaceNet parameters
for param in chain(facenet.parameters(), hopenet.parameters()):
    param.requires_grad = False

generator = MSGenerator().to(device)
discriminator = MultiScaleDiscriminator().to(device)
joint_loss = JointLoss()

g_optim = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.9, 0.999))
d_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))

#############################################
# 8. Dataset and DataLoader Setup (limit to 1500 images)
#############################################
# === Set the directories to the extracted folders ===
hr_dir = hr_extract_dir
lr_dir = lr_extract_dir
full_dataset = FaceSRDataset(hr_dir, lr_dir)

# Use only the first 1500 images (if available)
num_images = 200000
if len(full_dataset) > num_images:
    indices = list(range(num_images))
    dataset = Subset(full_dataset, indices)
else:
    dataset = full_dataset

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

#############################################
# 9. Helper Functions for Evaluation Metrics
#############################################
def unnormalize(img):
    """
    Convert an image from [-1, 1] to [0, 1]
    """
    return (img + 1) / 2

def compute_psnr(sr, hr):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio) given two images (assumed in [0,1]).
    """
    mse = F.mse_loss(sr, hr)
    psnr = 10 * torch.log10(1 / mse)
    return psnr

def get_angles_from_output(logits, idx_tensor, a, b):
    """
    Convert the output logits from Hopenet to an expected angle.
    The predicted angle = a + (b - a) / (num_bins - 1) * sum_i i * p_i,
    where p_i is the softmax probability for bin i.
    """
    prob = F.softmax(logits, dim=1)
    expected = torch.sum(prob * idx_tensor, dim=1)
    angle = a + (b - a) / (len(idx_tensor) - 1) * expected
    return angle

# Create an index tensor for angle conversion
idx_tensor = torch.arange(num_bins).float().to(device)

# Instantiate the LPIPS metric (LPIPS expects images in [-1, 1])
lpips_fn = lpips.LPIPS(net='alex').to(device)

#############################################
# 10. Training Loop with tqdm Progress Bar, Evaluation Metrics, and Fail-safe Checkpointing
#############################################
checkpoint_dir = "./checkpoints"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# File to save the latest checkpoint (for resuming mid-training)
checkpoint_latest_path = os.path.join(checkpoint_dir, "checkpoint_latest.pth")

# Initialize training history dictionary to log metrics per batch
training_history = {"epoch": [], "batch": [], "g_loss": [], "d_loss": []}

# Variables to resume training if checkpoint exists
start_epoch = 0
start_batch = 0
num_epochs = 100  # Change as desired

# Load checkpoint if it exists
if os.path.exists(checkpoint_latest_path):
    checkpoint = torch.load(checkpoint_latest_path, map_location=device)
    start_epoch = checkpoint.get("epoch", 0)
    start_batch = checkpoint.get("batch", 0)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
    g_optim.load_state_dict(checkpoint["g_optim_state_dict"])
    d_optim.load_state_dict(checkpoint["d_optim_state_dict"])
    training_history = checkpoint.get("training_history", training_history)
    print(f"Resuming training from epoch {start_epoch+1}, batch {start_batch+1}")

checkpoint_interval = 100  # Save checkpoint every 100 batches

try:
    for epoch in range(start_epoch, num_epochs):
        generator.train()
        # Reset batch counter at the beginning of a new epoch
        batch_counter = 0
        epoch_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        for batch_idx, batch in enumerate(epoch_bar):
            # If resuming in the middle of an epoch, skip already processed batches
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            # --- Train Generator ---
            g_optim.zero_grad()
            sr_outputs = generator(lr)  # List of outputs from each scale
            g_total_loss, g_losses = joint_loss(sr_outputs, hr)

            # Adversarial loss computed at each stage (using multi-scale discriminators)
            adv_loss = 0
            for i in range(3):
                scale = 0.5 ** (3 - 1 - i)
                real_imgs = F.interpolate(hr, scale_factor=scale, mode='bicubic', align_corners=False)
                pred_fake = discriminator(sr_outputs[i].detach(), i)
                adv_loss += F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
            g_total_loss += joint_loss.λ_adv * adv_loss

            g_total_loss.backward()
            g_optim.step()

            # --- Train Discriminator ---
            d_optim.zero_grad()
            d_loss = 0
            for i in range(3):
                scale = 0.5 ** (3 - 1 - i)
                real_imgs = F.interpolate(hr, scale_factor=scale, mode='bicubic', align_corners=False)
                pred_real = discriminator(real_imgs, i)
                loss_real = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
                pred_fake = discriminator(sr_outputs[i].detach(), i)
                loss_fake = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
                d_loss += 0.5 * (loss_real + loss_fake)
            d_loss.backward()
            d_optim.step()

            epoch_bar.set_postfix({
                "G Loss": g_total_loss.item(),
                "D Loss": d_loss.item()
            })

            # Log training metrics for this batch
            training_history["epoch"].append(epoch)
            training_history["batch"].append(batch_idx)
            training_history["g_loss"].append(g_total_loss.item())
            training_history["d_loss"].append(d_loss.item())

            batch_counter += 1

            # Save a checkpoint every checkpoint_interval batches
            if batch_counter % checkpoint_interval == 0:
                checkpoint_state = {
                    "epoch": epoch,
                    "batch": batch_idx,
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "g_optim_state_dict": g_optim.state_dict(),
                    "d_optim_state_dict": d_optim.state_dict(),
                    "training_history": training_history,
                }
                torch.save(checkpoint_state, checkpoint_latest_path)
                print(f"Checkpoint saved at epoch {epoch+1}, batch {batch_idx+1}")

        print(f"Epoch {epoch+1}/{num_epochs} | G Loss: {g_total_loss.item():.3f} | D Loss: {d_loss.item():.3f}")

        # ------------------------------
        # Evaluation Phase: Compute Metrics on the current epoch
        # ------------------------------
        generator.eval()
        with torch.no_grad():
            total_psnr = 0
            total_ssim = 0
            total_lpips = 0
            total_hpe_mae_yaw = 0
            total_hpe_mae_pitch = 0
            total_hpe_mae_roll = 0
            count = 0

            for batch in dataloader:
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                sr_list = generator(lr)
                sr = sr_list[-1]  # Use the final SR output

                # Unnormalize images for PSNR and SSIM (convert from [-1,1] to [0,1])
                sr_unnorm = unnormalize(sr)
                hr_unnorm = unnormalize(hr)

                psnr_val = compute_psnr(sr_unnorm, hr_unnorm)
                ssim_val = ssim(sr_unnorm, hr_unnorm, data_range=1.0, size_average=True)

                total_psnr += psnr_val.item() * lr.size(0)
                total_ssim += ssim_val.item() * lr.size(0)

                # LPIPS metric (LPIPS expects images in [-1, 1])
                lpips_val = lpips_fn(sr, hr).mean()  # take mean over batch
                total_lpips += lpips_val.item() * lr.size(0)

                # Compute head pose predictions using Hopenet.
                hr_yaw, hr_pitch, hr_roll = hopenet(hr)
                sr_yaw, sr_pitch, sr_roll = hopenet(sr)

                # Convert logits to angles using the expected value formulation.
                # Assume the angle range is [-99, 99]
                yaw_gt = get_angles_from_output(hr_yaw, idx_tensor, a=-99, b=99)
                pitch_gt = get_angles_from_output(hr_pitch, idx_tensor, a=-99, b=99)
                roll_gt = get_angles_from_output(hr_roll, idx_tensor, a=-99, b=99)

                yaw_pred = get_angles_from_output(sr_yaw, idx_tensor, a=-99, b=99)
                pitch_pred = get_angles_from_output(sr_pitch, idx_tensor, a=-99, b=99)
                roll_pred = get_angles_from_output(sr_roll, idx_tensor, a=-99, b=99)

                mae_yaw = torch.mean(torch.abs(yaw_pred - yaw_gt))
                mae_pitch = torch.mean(torch.abs(pitch_pred - pitch_gt))
                mae_roll = torch.mean(torch.abs(roll_pred - roll_gt))

                total_hpe_mae_yaw += mae_yaw.item() * lr.size(0)
                total_hpe_mae_pitch += mae_pitch.item() * lr.size(0)
                total_hpe_mae_roll += mae_roll.item() * lr.size(0)

                count += lr.size(0)

            avg_psnr = total_psnr / count
            avg_ssim = total_ssim / count
            avg_lpips = total_lpips / count
            avg_hpe_yaw = total_hpe_mae_yaw / count
            avg_hpe_pitch = total_hpe_mae_pitch / count
            avg_hpe_roll = total_hpe_mae_roll / count

            print(f"--- Evaluation Metrics after Epoch {epoch+1} ---")
            print(f"PSNR: {avg_psnr:.2f} dB")
            print(f"SSIM: {avg_ssim:.4f}")
            print(f"LPIPS: {avg_lpips:.4f}")
            print(f"HPE MAE (Yaw): {avg_hpe_yaw:.2f}°, (Pitch): {avg_hpe_pitch:.2f}°, (Roll): {avg_hpe_roll:.2f}°")
        generator.train()

        # Save an epoch-level checkpoint
        checkpoint_state = {
            "epoch": epoch,
            "batch": 0,  # Reset batch counter for new epoch
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optim_state_dict": g_optim.state_dict(),
            "d_optim_state_dict": d_optim.state_dict(),
            "training_history": training_history,
        }
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save(checkpoint_state, epoch_checkpoint_path)
        torch.save(checkpoint_state, checkpoint_latest_path)  # Update latest checkpoint as well
        print(f"Checkpoint saved for epoch {epoch+1} at {epoch_checkpoint_path}")

        # Reset start_batch after resuming once
        start_batch = 0

except KeyboardInterrupt:
    # In case of manual interruption, save a checkpoint and exit gracefully
    print("Training interrupted. Saving checkpoint...")
    checkpoint_state = {
        "epoch": epoch,
        "batch": batch_idx,
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "g_optim_state_dict": g_optim.state_dict(),
        "d_optim_state_dict": d_optim.state_dict(),
        "training_history": training_history,
    }
    torch.save(checkpoint_state, checkpoint_latest_path)
    print(f"Checkpoint saved at epoch {epoch+1}, batch {batch_idx+1}. Exiting training.")
    exit(0)