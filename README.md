# FluidLensing

This repository contains a deep learning solution for segmentation of underwater coral reef imagery. It implements an Attentive ResUNet with Channel-Matching Residual Blocks for accurate semantic segmentation of different coral reef habitats.


## Installation

Clone this repository and install the required dependencies:

```bash
git clone https://github.com/acesumiami/FluidLensing.git
cd FluidLensing
pip install -r requirements.txt
```

## Dataset Structure

For optimal performance, organize your data like this:

```
data/
├── train_images/
│   ├── image001.png
│   ├── image002.png
│   └── ...
├── train_labels/
│   ├── image001.png
│   ├── image002.png
│   └── ...
├── val_images/
└── val_labels/
```

Labels should be RGB images where each class is represented by a specific color:

| Class | Name | Hex Color |
|-------|------|-----------|
| 0 | Branching Coral | #93e7a9 |
| 1 | Massive Coral | #4366d4 |
| 2 | Coral Fore Reef | #08e8f6 |
| 3 | Reef Crest - Coralline Algal Ridge | #f2bc88 |
| 4 | Algae | #5dbd3f |
| 5 | Sand / Rubble | #b0447c |
| 6 | No Class | #9c7261 |

## Usage

### Training

To train the segmentation model:

```bash
python train.py --train_img_dir path/to/train_images --train_lbl_dir path/to/train_labels \
                --val_img_dir path/to/val_images --val_lbl_dir path/to/val_labels \
                --epochs 50 --batch_size 4 --image_size 512 \
                --output_dir path/to/output
```

Options:
- `--train_img_dir`: Path to training images directory
- `--train_lbl_dir`: Path to training labels directory
- `--val_img_dir`: Path to validation images directory
- `--val_lbl_dir`: Path to validation labels directory
- `--image_size`: Image size for training (default: 512)
- `--batch_size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 50)
- `--initial_lr`: Initial learning rate (default: 5e-5)
- `--output_dir`: Directory for model and results (default: './output')

### Prediction

To run predictions on large GeoTIFF images:

```bash
python predict.py --model_path path/to/model.keras --input_tif path/to/image.tif \
                  --patch_size 512 --overlap 256 --output_dir path/to/output
```

Options:
- `--model_path`: Path to trained model file
- `--input_tif`: Path to input GeoTIFF image
- `--patch_size`: Size of patches for prediction (default: 512)
- `--overlap`: Overlap between patches in pixels (default: 256)
- `--output_dir`: Directory to save results (default: './predictions')

## Model Architecture

The model uses an Attentive ResUNet architecture with:
- EfficientNetB3 backbone pre-trained on ImageNet
- Channel-matching residual blocks for better gradient flow
- Attention gates for improved feature selection
- Class-weighted loss function to handle imbalanced data
- Advanced data augmentation for improved generalization

## Output Files

The prediction script generates:
- Visual comparison of original image and segmentation prediction
- Color-coded segmentation mask
- Overlay visualization of prediction on original image
- Class distribution chart
- Segmentation legend

