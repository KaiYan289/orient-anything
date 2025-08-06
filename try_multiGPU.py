import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
from transformers import AutoImageProcessor
from huggingface_hub import hf_hub_download
from tqdm import tqdm
import torch.multiprocessing as mp
# Assuming these modules are in your project
from paths import *
from vision_tower import DINOv2_MLP
from utils import *
from inference import * # Assuming get_3angle_batched exists here

ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything", filename="croplargeEX2/dino_weight.pt", repo_type="model", cache_dir='./', resume_download=True)

# 1. Create a custom Dataset
class ImageDataset(Dataset):
    def __init__(self, tasks, processor):
        self.tasks = tasks # List of (image_path, output_path)
        self.processor = processor

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        image_path, output_path = self.tasks[idx]
        image = Image.open(image_path).convert("RGB")
        processed_image = self.processor(image, return_tensors="pt")['pixel_values'].squeeze(0)
        return processed_image, output_path

def setup(rank, world_size):
    """Initializes the distributed process group."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    print(f"Rank {rank} attempting to initialize process group.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Rank {rank} initialize complete!")

def cleanup():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def run_inference(rank, world_size, tasks, ckpt_path):
    """Main function for each DDP process."""
    setup(rank, world_size)
    print("starting inference...")
    # --- Model and Processor Setup ---
    device = f'cuda:{rank}'
    val_preprocess = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir='./')
    print("starting dino...")
    model = DINOv2_MLP(
        dino_mode='large', in_dim=1024, out_dim=360+180+180+2, 
        evaluate=True, mask_dino=False, frozen_back=False
    ).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    
    # Wrap the model with DDP
    ddp_model = DDP(model, device_ids=[rank])
    ddp_model.eval()
    print("ddp ready...")
    # --- Data Loading Setup ---
    dataset = ImageDataset(tasks, val_preprocess)
    # DistributedSampler ensures each GPU gets a unique subset of data
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

    # Progress bar only on the main process
    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc="Batch Inference", ncols=100)

    # --- Inference Loop ---
    with torch.no_grad():
        for batch, output_paths in dataloader:
            batch = batch.to(device)
            # Write results from this batch
            for i in range(batch.size(0)):
                single_image_tensor = batch[i].unsqueeze(0) # Add batch dim back: [1, C, H, W]
                single_output_path = output_paths[i]
                # origin_image = Image.open(image_path).convert('RGB')
                angles = get_3angle_dataloader(single_image_tensor, ddp_model)
                azimuth, polar, rotation, confidence = angles
                print(i, single_output_path, f"azimuth: {azimuth} polar: {polar} rotation: {rotation} confidence: {confidence}\n")
                with open(single_output_path, "w") as f:
                    f.write(f"azimuth: {azimuth} polar: {polar} rotation: {rotation} confidence: {confidence}\n")
            
            if rank == 0:
                pbar.update(1)
    
    if rank == 0:
        pbar.close()

    cleanup()

if __name__ == "__main__":
    # --- Collect Tasks ---
    # (Same code as before to collect all image and output paths into a list called `all_tasks`)

    # --- Launch with torchrun ---
    world_size = torch.cuda.device_count()
    all_tasks = []
    for folder in os.listdir('/root/VQASynth/vqasynth/patches_with_boxes/'):
        os.makedirs("/root/VQASynth/vqasynth/orientation_by_oa/" + folder, exist_ok=True)

        for patch_name in os.listdir("/root/VQASynth/vqasynth/patches_with_boxes/" + folder):
            if patch_name.find(".png") == -1 and patch_name.find(".jpg") == -1: continue
            image_path = '/root/VQASynth/vqasynth/patches_with_boxes/' + folder + "/" + patch_name    
            all_tasks.append((image_path, "/root/VQASynth/vqasynth/orientation_by_oa/" + folder + "/" + patch_name.replace(".jpg", ".txt").replace(".png", ".txt")))
    print(len(all_tasks), all_tasks[0])

    mp.spawn(run_inference,
             args=(world_size, all_tasks, ckpt_path),
             nprocs=world_size,
             join=True)