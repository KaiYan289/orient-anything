from paths import *
from vision_tower import DINOv2_MLP
from transformers import AutoImageProcessor
import torch
from PIL import Image
import os
import torch.nn.functional as F
from utils import *
from inference import *
import time
cnt = 0
from huggingface_hub import hf_hub_download
ckpt_path = hf_hub_download(repo_id="Viglong/Orient-Anything", filename="croplargeEX2/dino_weight.pt", repo_type="model", cache_dir='./', resume_download=True)
print(ckpt_path)

save_path = './'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dino = DINOv2_MLP(
                    dino_mode   = 'large',
                    in_dim      = 1024,
                    out_dim     = 360+180+180+2,
                    evaluate    = True,
                    mask_dino   = False,
                    frozen_back = False
                )

dino.eval()
print('model create')
dino.load_state_dict(torch.load(ckpt_path, map_location='cuda:0'))
dino = dino.to(device)
print('weight loaded')
val_preprocess   = AutoImageProcessor.from_pretrained(DINO_LARGE, cache_dir='./')
t0 = time.time()
for folder in os.listdir('/root/VQASynth/vqasynth/patches_with_boxes/'):
    os.makedirs("/root/VQASynth/vqasynth/orientation_by_oa/" + folder, exist_ok=True)

    for patch_name in os.listdir("/root/VQASynth/vqasynth/patches_with_boxes/" + folder):
        if patch_name.find(".png") == -1 and patch_name.find(".jpg") == -1: continue
        image_path = '/root/VQASynth/vqasynth/patches_with_boxes/' + folder + "/" + patch_name    
        # image_path = '/root/VQASynth/vqasynth/patches_with_boxes/000000000136/0.png'
        
        # print("image_path:", image_path)
        
        origin_image = Image.open(image_path).convert('RGB')
        angles = get_3angle(origin_image, dino, val_preprocess, device)
        azimuth     = float(angles[0])
        polar       = float(angles[1])
        rotation    = float(angles[2])
        confidence  = float(angles[3])

        with open("/root/VQASynth/vqasynth/orientation_by_oa/" + folder + "/" + patch_name.replace(".jpg", ".txt").replace(".png", ".txt"), "w") as f:
            f.write(f"azimuth: {azimuth} polar: {polar} rotation: {rotation} confidence: {confidence}\n")
        cnt += 1
        if cnt % 100 == 0:
            print(cnt, time.time() - t0)
        # print('azimuth:', azimuth, 'polar:', polar, 'rotation:', rotation, 'confidence:', confidence)