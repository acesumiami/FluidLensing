import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from osgeo import gdal
import cv2
import time
from matplotlib.patches import Patch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Run prediction on GeoTIFF imagery using a trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--input_tif', type=str, required=True, help='Path to the input GeoTIFF file')
    parser.add_argument('--output_dir', type=str, default='./predictions', help='Directory to save output files')
    parser.add_argument('--patch_size', type=int, default=512, help='Patch size for prediction (must match model input)')
    parser.add_argument('--overlap', type=int, default=256, help='Overlap between patches in pixels')
    
    return parser.parse_args()


def define_classes():
    classes = {
        0: {"name": "Branching Coral", "hex": "#93e7a9", "rgb": (147, 231, 169)},
        1: {"name": "Massive Coral", "hex": "#4366d4", "rgb": (67, 102, 212)},
        2: {"name": "Coral Fore Reef", "hex": "#08e8f6", "rgb": (8, 232, 246)},
        3: {"name": "Reef Crest - Coralline Algal Ridge", "hex": "#f2bc88", "rgb": (242, 188, 136)},
        4: {"name": "Algae", "hex": "#5dbd3f", "rgb": (93, 189, 63)},
        5: {"name": "Sand / Rubble", "hex": "#b0447c", "rgb": (176, 68, 124)},
        6: {"name": "No Class", "hex": "#9c7261", "rgb": (156, 114, 97)}
    }
    
    num_classes = len(classes)
    color_map = {i: classes[i]["rgb"] for i in range(num_classes)}
    
    return classes, num_classes, color_map


def predict_tif(model, tif_path, patch_size, stride, output_dir):
    print(f"Processing {os.path.basename(tif_path)}...")
    start_time = time.time()

    ds = gdal.Open(tif_path)
    if ds is None:
        print(f"Error: Could not open {tif_path}")
        return

    width = ds.RasterXSize
    height = ds.RasterYSize
    bands = ds.RasterCount

    print(f"Image dimensions: {width}x{height}, {bands} bands")

    if bands >= 3:
        r_band = ds.GetRasterBand(1).ReadAsArray()
        g_band = ds.GetRasterBand(2).ReadAsArray()
        b_band = ds.GetRasterBand(3).ReadAsArray()
        image = np.dstack((r_band, g_band, b_band))
    elif bands == 1:
        image = ds.GetRasterBand(1).ReadAsArray()
        image = np.stack([image] * 3, axis=-1)
    else:
        print(f"Unsupported number of bands: {bands}")
        return

    display_img = image.copy()

    classes, num_classes, color_map = define_classes()
    
    prediction_accumulator = np.zeros((height, width, num_classes), dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)

    y_grid, x_grid = np.mgrid[0:patch_size, 0:patch_size]
    center_y, center_x = patch_size // 2, patch_size // 2
    dist_from_center = np.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
    max_dist = np.sqrt(2) * (patch_size // 2)
    patch_weight = np.clip(1.0 - dist_from_center / max_dist, 0.1, 1.0)

    patches_processed = 0
    total_patches = max(1, (height - patch_size + stride) // stride) * max(1, (width - patch_size + stride) // stride)

    print(f"Total patches to process: {total_patches}")

    for y in range(0, max(1, height - patch_size + 1), stride):
        for x in range(0, max(1, width - patch_size + 1), stride):
            if y + patch_size > height or x + patch_size > width:
                padded_patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)

                valid_h = min(patch_size, height - y)
                valid_w = min(patch_size, width - x)

                padded_patch[:valid_h, :valid_w, :] = image[y:y+valid_h, x:x+valid_w, :]

                if valid_h < patch_size:
                    reflection_range = min(valid_h, patch_size - valid_h)
                    padded_patch[valid_h:valid_h+reflection_range, :valid_w, :] = np.flip(
                        padded_patch[valid_h-reflection_range:valid_h, :valid_w, :], axis=0)

                if valid_w < patch_size:
                    reflection_range = min(valid_w, patch_size - valid_w)
                    padded_patch[:valid_h, valid_w:valid_w+reflection_range, :] = np.flip(
                        padded_patch[:valid_h, valid_w-reflection_range:valid_w, :], axis=1)

                patch = padded_patch
                actual_h, actual_w = valid_h, valid_w
            else:
                patch = image[y:y+patch_size, x:x+patch_size].copy()
                actual_h, actual_w = patch_size, patch_size

            white_mask = np.all(patch > 250, axis=-1)
            white_percentage = np.mean(white_mask)
            if white_percentage > 0.25:
                patches_processed += 1
                if patches_processed % 10 == 0:
                    print(f"Processed {patches_processed}/{total_patches} patches ({patches_processed*100//total_patches}%)")
                continue

            try:
                patch_input = patch.astype(np.float32) / 255.0
                patch_input = tf.image.per_image_standardization(patch_input).numpy()
                patch_input = np.expand_dims(patch_input, axis=0)

                patch_pred_probs = model.predict(patch_input, verbose=0)[0]

                valid_y_end = min(y + patch_size, height)
                valid_x_end = min(x + patch_size, width)

                for i in range(num_classes):
                    weighted_pred = patch_pred_probs[:actual_h, :actual_w, i] * patch_weight[:actual_h, :actual_w]
                    prediction_accumulator[y:valid_y_end, x:valid_x_end, i] += weighted_pred

                weight_map[y:valid_y_end, x:valid_x_end] += patch_weight[:actual_h, :actual_w]

                patches_processed += 1
                if patches_processed % 10 == 0:
                    print(f"Processed {patches_processed}/{total_patches} patches ({patches_processed*100//total_patches}%)")

            except Exception as e:
                print(f"Error processing patch at ({x}, {y}): {e}")
                continue

    nonzero_mask = weight_map > 0
    for i in range(num_classes):
        prediction_accumulator[:, :, i][nonzero_mask] /= weight_map[nonzero_mask]

    prediction = np.argmax(prediction_accumulator, axis=2).astype(np.uint8)

    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for class_idx, color in color_map.items():
        mask = (prediction == class_idx)
        rgb_mask[mask] = color

    end_time = time.time()
    print(f"Prediction completed in {end_time - start_time:.2f} seconds")

    class_pixels = [np.sum(prediction == i) for i in range(num_classes)]
    total_pixels = prediction.size
    percentages = [100 * count / total_pixels for count in class_pixels]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.title('Original Image', fontsize=14)

    display_img_normalized = display_img.astype(np.float32)
    for c in range(min(3, display_img_normalized.shape[2])):
        channel = display_img_normalized[:,:,c]
        min_val, max_val = np.percentile(channel, [2, 98])
        channel = np.clip((channel - min_val) / (max_val - min_val + 1e-8), 0, 1)
        display_img_normalized[:,:,c] = channel * 255

    plt.imshow(display_img_normalized.astype(np.uint8))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Segmentation Prediction', fontsize=14)
    plt.imshow(rgb_mask)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'segmentation_{timestamp}.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(20, 10))
    overlay = cv2.addWeighted(display_img_normalized.astype(np.uint8), 0.7, rgb_mask, 0.3, 0)
    plt.imshow(overlay)
    plt.title('Overlay (70% Original, 30% Mask)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'overlay_{timestamp}.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(15, 7))
    class_names = [classes[i]["name"] for i in range(num_classes)]
    colors = [tuple(c/255 for c in color_map[i]) for i in range(num_classes)]

    bars = plt.bar(class_names, percentages, color=colors, edgecolor='k')
    plt.title('Class Distribution', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Percentage (%)')

    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'class_distribution_{timestamp}.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(12, 2))
    plt.axis('off')

    legend_elements = []
    for i in range(num_classes):
        color = tuple(c/255 for c in color_map[i])
        legend_elements.append(
            Patch(facecolor=color, edgecolor='k', label=classes[i]["name"])
        )

    plt.legend(handles=legend_elements, loc='center', ncol=len(classes),
               frameon=True, fontsize=12, title="Segmentation Classes")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'legend_{timestamp}.png'), dpi=300)
    plt.close()
    
    driver = gdal.GetDriverByName('GTiff')
    outds = driver.Create(os.path.join(output_dir, f'prediction_{timestamp}.tif'), 
                          width, height, 1, gdal.GDT_Byte)
    outds.SetGeoTransform(ds.GetGeoTransform())
    outds.SetProjection(ds.GetProjection())
    outds.GetRasterBand(1).WriteArray(prediction)
    outds = None
    
    outds = driver.Create(os.path.join(output_dir, f'colored_mask_{timestamp}.tif'), 
                         width, height, 3, gdal.GDT_Byte)
    outds.SetGeoTransform(ds.GetGeoTransform())
    outds.SetProjection(ds.GetProjection())
    for i in range(3):
        outds.GetRasterBand(i+1).WriteArray(rgb_mask[:,:,i])
    outds = None
    
    print(f"Results saved to {output_dir}")
    
    return prediction, rgb_mask


def main():
    args = parse_args()
    
    if not os.path.exists(args.input_tif):
        print(f"Error: File not found: {args.input_tif}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found: {args.model_path}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPU is available: {physical_devices}")
        for gpu in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except:
                pass
    else:
        print("No GPU found. Running on CPU (slower)")

    print(f"Loading model from {args.model_path}")
    model = keras.models.load_model(
        args.model_path,
        custom_objects={'loss': None},
        compile=False
    )
    
    print(f"Model loaded successfully. Input shape: {model.input_shape}")
    
    if model.input_shape[1] != args.patch_size or model.input_shape[2] != args.patch_size:
        print(f"Warning: Model expects {model.input_shape[1]}x{model.input_shape[2]} input, but patch_size is {args.patch_size}")
        print(f"Adjusting patch_size to match model requirements: {model.input_shape[1]}")
        args.patch_size = model.input_shape[1]

    print(f"Starting prediction with patch size: {args.patch_size}, stride: {args.overlap}")
    
    prediction, rgb_mask = predict_tif(
        model=model,
        tif_path=args.input_tif,
        patch_size=args.patch_size,
        stride=args.overlap,
        output_dir=args.output_dir
    )
    
    print(f"Prediction visualization completed!")


if __name__ == "__main__":
    main()