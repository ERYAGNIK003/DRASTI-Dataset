# Benchmarking YOLOv8 Models on DRASTI Dataset using Stepwell HPC (High Performance Computing) facility

This repository provides the complete workflow to benchmark YOLOv8 models on the **DRASTI** dataset using **HPC clusters with Slurm job scheduler**. It includes environment setup, data preparation, model training, testing, and benchmarking steps.

---

## 📦 Environment Setup (HPC Modules)

Before starting, ensure the appropriate modules are loaded and the Python virtual environment is created:

```bash
module load Python/3.10.15
module load cuda/latest       # Typically uses torch-2.7.1+cu126
module list                   # Verify Python and CUDA are loaded

python3.10 -m venv yolov8_env
source ~/yolov8_env/bin/activate
pip install --upgrade pip
pip install ultralytics       # Version used: 8.3.159

mkdir NewFolder
cd ~/NewFolder
```

---

## 📁 Step 1: Dataset Preparation

1. **Download** and unzip the `DRASTI` dataset and `DRASTI.yaml` inside `NewFolder`.

2. Update the dataset path in the `DRASTI.yaml` file.

3. **Original Directory Structure**:
```
DRASTI/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

4. **Reorganize** the dataset to match YOLO-compatible structure:
   - Rename original `labels/` folders by appending `_original`.
   - Final structure:
```
DRASTI.yaml
DRASTI/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train_original/
    ├── val_original/
    └── test_original/
```

5. **Annotation Conversion for YOLO OBB**

   - Navigate to the converter.py script:
     ```
     yolov8_env/lib/python3.10/site-packages/ultralytics/data/converter.py
     ```

   - Edit function `convert_dota_to_yolo_obb()`:
     - Around **line 458**, update `class_mapping`:
       ```python
       class_mapping = {
           "Auto3WCargo": 0,
           "AutoRicksaw": 1,
           "Bus": 2,
           "Container": 3,
           "Mixer": 4,
           "MotorCycle": 5,
           "PickUp": 6,
           "SUV": 7,
           "Sedan": 8,
           "Tanker": 9,
           "Tipper": 10,
           "Trailer": 11,
           "Truck": 12,
           "Van": 13,
       }
       ```

     - Around **line 499**, where it says `for split in {"train", "val"}:`, update:
       ```python
       for split in {"train", "val", "test"}:
       ```

     - Around **line 508**, in place of `.png` change:
       ```python
       image_ext = ".jpg"
       ```

6. Run the conversion:
- This will create the necessary YOLO OBB format annotations in the `DRASTI/labels/` directory with names `train, val, test`.
```python
from ultralytics.data.converter import convert_dota_to_yolo_obb
convert_dota_to_yolo_obb("../DRASTI") # Adjust the path to your DRASTI dataset
```

7. The final structure becomes:

```
DRASTI/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── train_original/
    ├── val/
    ├── val_original/
    ├── test/
    └── test_original/
```

---

## 🏋️ Step 2: Training the YOLOv8 OBB Model

1. Download the training script `DRASTI_Train_YOLOv8n_obb.py` and place it in your working directory.

2. Configure:
   - Path to `DRASTI.yaml`
   - Output directory for results (`project` path in script)
   - Choose desired model YAML file (e.g., `yolov8n-obb.yaml`)

3. Prepare the training bash script `DRASTI_Train_YOLOv8n_obb.sh`:
```bash
#!/bin/bash
#SBATCH --job-name=DRASTI_Train_YOLOv8n_obb
#SBATCH --output=logs/output_%j.log
#SBATCH --error=logs/error_%j.log
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --partition=gpu

module load Python/3.10.15

# Activate your virtual environment
source ~/yolov8s_env/bin/activate

# Debug GPU availability before training
echo "Checking GPU availability..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count()); print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Run the training
python DRASTI_Train_YOLOv8n_obb.py
```

4. Launch training:
```bash
sbatch DRASTI_Train_YOLOv8n_obb.sh
```

5. Monitor progress:
- Check the job status:
```bash
squeue
```

6. Training logs:
   - Check the Training files like `result.csv` by navigating to the `project` directory specified in the script.
   - Logs are saved in `logs/` directory:
     - `output.log` for standard output.
     - `error.log` for errors.
   - Model weights and metrics are saved in the `project` directory specified in the training script.
   - Make a plot of the training result like mAP, loss, from `results.csv` file in sheets or excel.

---

## ✅ Testing

1. Download and configure `DRASTI_Test_YOLOv8n_obb.py`:
   - Set path to `DRASTI.yaml`
   - Set path to trained model and output `project` folder

2. Create a bash file `DRASTI_Test_YOLOv8n_obb.sh` similar to the training script and execute:
```bash
sbatch DRASTI_Test_YOLOv8n_obb.sh
```

3. Testing outputs:
   - Metrics (mAP, Precision, Recall)
   - Confusion matrix
   - Saved in the `project` folder

---

## ⚡ Benchmarking (parameters like mAP50-95,  Parameters(in M), GFLOPs, Model Size, Inference, FPS, Input Size)

1. Download `DRASTI_Benchmark_YOLO.py`.

2. Update:
   - Paths to trained model `.pt` files
   - Path to `DRASTI.yaml`

3. Internally, the script uses a temporary copy of the `DRASTI.yaml` file and points to the test set for evaluation.

4. Run:
```bash
python DRASTI_Benchmark_YOLO.py
```

---

## 🎥 Inference on Video

1. Download `DRASTI_Inference_YOLO.py`.

2. Configure the script:
   - Model path
   - Input video path
   - Output video path for annotated video results

3. Run:
```bash
python DRASTI_Inference_YOLO.py
```

---

## 📂 Directory Summary

```
NewFolder/
├── DRASTI/
│   ├── images/
│   └── labels/
├── DRASTI.yaml
├── DRASTI_Train_YOLOv8n_obb.py
├── DRASTI_Test_YOLOv8n_obb.py
├── DRASTI_Benchmark_YOLO.py
├── DRASTI_Inference_YOLO.py
├── DRASTI_Train_YOLOv8n_obb.sh
├── DRASTI_Test_YOLOv8n_obb.sh
├── logs/
└── runs/
```

---

## 📌 Notes

- Ensure `ultralytics` version is consistent across training, testing, and benchmarking.
- Adapt SLURM job scripts according to your HPC scheduler and resource availability.
- Always activate the virtual environment before running scripts.

---

## 📜 Citation

If you use this pipeline or dataset in your research, please cite:

- The DRASTI Dataset paper
```bibtex
```
- Ultralytics YOLOv8 framework

---

## 🔗 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- DRASTI Dataset Authors
- Stepwell HPC Support Team
