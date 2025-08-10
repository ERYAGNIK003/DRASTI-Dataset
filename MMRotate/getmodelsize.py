import torch
import io
import sys

# Load your original .pth file
pth_file = './work_dirs/s2anet_r50_fpn_1x_dota_le135/latest.pth'  # <-- change this to your file path
checkpoint = torch.load(pth_file, map_location='cpu')

# Extract only the model weights (state_dict)
if isinstance(checkpoint, dict):
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
else:
    raise ValueError("Unrecognized format")

# Serialize to in-memory buffer
buffer = io.BytesIO()
torch.save(state_dict, buffer)

# Get size in bytes → convert to MB
buffer_size_mb = sys.getsizeof(buffer.getvalue()) / (1024 * 1024)
print(f" S2ANET Model size (in-memory, weights only): {buffer_size_mb:.2f} MB")



# Load your original .pth file
pth_file = './work_dirs/rotated_fcos_csl_gaussian_r50_fpn_1x_dota_le90/latest.pth'  # <-- change this to your file path
checkpoint = torch.load(pth_file, map_location='cpu')

# Extract only the model weights (state_dict)
if isinstance(checkpoint, dict):
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
else:
    raise ValueError("Unrecognized format")

# Serialize to in-memory buffer
buffer = io.BytesIO()
torch.save(state_dict, buffer)

# Get size in bytes → convert to MB
buffer_size_mb = sys.getsizeof(buffer.getvalue()) / (1024 * 1024)
print(f"Rotated FCOS CSL Gaussian Model size : {buffer_size_mb:.2f} MB")


