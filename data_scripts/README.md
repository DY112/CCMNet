# Dataset Details

This file provides detailed information about the datasets used in the CCMNet project.

## Overview

This document explains the **entire data pipeline** required before training CCMNet:
1. Obtain a base & preprocessed dataset covering all supported cameras (Option A or Option B below).
2. Verify that the preprocessed dataset resides in `../dataset/CCMNet/` with the expected structure.
3. Run the Imaginary Camera Dataset Generation script to create synthetic RAW images that broaden the camera diversity of the training data.

The following sections guide you through these steps in order.

You can prepare the preprocessed dataset for this project in one of the following ways. **Option&nbsp;A** is recommended for most users as it skips all manual download and preprocessing steps.

-  Option A. Use the preprocessed dataset (Recommended)
-  Option B. Prepare from scratch

Regardless of the option you choose, the subsequent steps expect the preprocessed datasets to be located at `../dataset/CCMNet/`.

## Option A. Use the preprocessed dataset (Recommended)
1. Download `CCMNet.tar.gz` from [this link](https://drive.google.com/drive/folders/1E79wLal1x1XSPRYpNE5QLGfOC-SJITaZ?usp=sharing).
2. Move the file to the `../dataset/` directory of your project.
3. Extract the archive:
   ```bash
   cd ../dataset
   tar -xzvf CCMNet.tar.gz
   ```
   This will create the `CCMNet/` directory with the structure described in the *Output Directory Structure* section.

4. Now you can directly proceed to the [Imaginary Camera Dataset Generation](#imaginary-camera-dataset-generation) section to create augmented data before training or evaluation.

## Option B: Generate preprocessed data from scratch

You need to create a `dataset` directory at `../` relative to your project root. The following structure is expected within `../dataset/` for each dataset before running the preprocessing scripts.

> **DNG Files for Preprocessing**
>
> For preprocessing, this repository provides a set of DNG files in the `/dngs` directory. These DNG files should be copied to the appropriate `dng` folders for each dataset (e.g., `../dataset/cube+/dng/`, `../dataset/NUS-8/[Camera]/dng/`, `../dataset/Intel-TAU/[Camera]/`).
>
> The reason for providing these DNG files is that the official datasets do not always include DNG files:
> - The NUS dataset provides RAW files in various proprietary formats (e.g., CR2, ARW, NEF, etc.) rather than DNG.
> - The Intel-TAU dataset only provides PNG images and does not include RAW or DNG files.
> To facilitate preprocessing, all necessary DNG files have been collected and converted using Adobe DNG Converter and web search, and are provided in this repository for your convenience.
>
> **Before running any preprocessing script, please make sure to copy the DNG files from `/dngs` to the corresponding locations in each dataset directory as required by the preprocessing scripts.**

### 1. NUS-8 Dataset

Expected path: `../dataset/NUS-8/`

The preprocessing script `data_scripts/preprocess_NUS_mp.py` expects the following structure for each camera:

```
../dataset/NUS-8/
├── Canon1DsMkIII/              # Example camera directory
│   ├── dng/
│   │   └── *.dng              # At least one DNG file for calibration metadata
│   ├── PNG/
│   │   └── *.PNG              # All raw images in PNG format for this camera
│   └── Canon1DsMkIII_gt.mat   # Ground truth illuminant and MCC coordinates
├── Canon600D/                  # Other cameras follow the same structure
├── NikonD5200/
├── SamsungNX2000/
├── SonyA57/
├── FujifilmXM1/
├── OlympusEPL6/
└── PanasonicGX1/
```

### 2. Cube+ Dataset

Expected path: `../dataset/cube+/`

The preprocessing script `data_scripts/preprocess_Cube_mp.py` expects the following structure for the Canon550D camera:

```
../dataset/cube+/
├── dng/
│   └── *.dng          (At least one DNG file for Canon550D calibration)
├── PNG/
│   └── *.PNG          (All raw images in PNG format)
└── cube+_gt.txt       (Ground truth illuminant data)
```

### 3. Intel-TAU Dataset

Expected path: `../dataset/Intel-TAU/`

The preprocessing script `data_scripts/preprocess_Intel_mp.py` expects the following structure for each camera:

```
../dataset/Intel-TAU/
├── Canon_5DSR/                # Example camera directory
│   ├── *.dng                 # At least one DNG file for calibration
│   ├── field_1_cameras/      # Each scene type has TIFF images and white point data
│   │   ├── *.tiff
│   │   └── *.wp             # White point data for corresponding TIFFs
│   ├── field_3_cameras/
│   │   ├── *.tiff
│   │   └── *.wp
│   ├── lab_printouts/
│   │   ├── *.tiff
│   │   └── *.wp
│   └── lab_realscene/
│       ├── *.tiff
│       └── *.wp
└── Nikon_D810/               # Other cameras follow the same structure
```

Note: The script is currently configured to process `Canon_5DSR` and `Nikon_D810`. The `IMX135-BLCCSC` camera is not included in training/testing as its CCM (Color Correction Matrix) is not provided.

### 4. Gehler-Shi Dataset

Expected path: `../dataset/Gehler_Shi/`

The preprocessing script `data_scripts/preprocess_Gehler_mp.py` expects the following structure:

```
../dataset/Gehler_Shi/
├── Canon1D/
│   ├── dng/
│   │   └── *.dng       (At least one DNG file for calibration)
│   └── png/
│       └── *.png       (All raw images in PNG format for this camera)
├── Canon5D/
│   ├── dng/
│   │   └── *.dng
│   └── png/
│       └── *.png
├── coordinates/
│   └── *_macbeth.txt (Macbeth ColorChecker coordinates for each image)
└── real_illum_568.mat (Ground truth illuminant data for all images)
```

Please download the respective datasets and arrange them according to these structures to ensure the preprocessing scripts work correctly.

### Preprocessing Scripts

After organizing the datasets according to the structure above, you need to run the preprocessing scripts located in the `data_scripts` directory. These scripts are multiprocessing-enabled versions for faster processing:

1. `preprocess_NUS_mp.py`: Processes the NUS-8 dataset
2. `preprocess_Cube_mp.py`: Processes the Cube+ dataset
3. `preprocess_Intel_mp.py`: Processes the Intel-TAU dataset
4. `preprocess_Gehler_mp.py`: Processes the Gehler-Shi dataset

Each script will:
- Extract camera calibration metadata from DNG files
- Process raw images to generate white-balanced and XYZ color space versions
- Create binary masks for color checker regions
- Resize images to 384x256 pixels
- Generate metadata files containing illuminant information and color transformation matrices

## Output Directory Structure

Regardless of whether you prepared the data via **Option A** or **Option B** in the Overview, the preprocessed dataset directory `../dataset/CCMNet/` should have the following structure:

```
../dataset/CCMNet/
├── preprocessed_for_augmentation/
│   ├── calibration_metadata.json
│   ├── Canon1D/                # Example camera directory
│   │   ├── metadata.json      # Per-image metadata
│   │   ├── *_raw.png         # Raw images
│   │   ├── *_wb.png          # White-balanced images
│   │   ├── *_xyz.png         # XYZ color space images
│   │   └── *_mask.png        # Color checker masks
│   └── [other camera folders]/ # Other cameras follow the same structure
└── original_resized/
    ├── Gehler-Shi/            # Example dataset directory
    │   ├── *_sensorname_Canon1D.png
    │   └── *_sensorname_Canon1D_metadata.json
    ├── NUS-8/                 # Other datasets follow the same structure
    ├── cube+/
    └── Intel-TAU/
```

The `preprocessed_for_augmentation` directory contains:
- `calibration_metadata.json`: Camera calibration information for all cameras
- Per-camera subdirectories containing:
  - `metadata.json`: Per-image metadata including ground truth illuminants and color transformation matrices
  - Processed images in various formats (raw, white-balanced, XYZ color space)
  - Binary masks for color checker regions

The `original_resized` directory contains:
- Dataset-specific subdirectories
- Resized raw images with their corresponding metadata files
- Images are named with their original filenames and camera information

## Imaginary Camera Dataset Generation

The `gen_imaginary_raw.py` script generates synthetic RAW images by interpolating between different camera characteristics. This is useful for creating augmented training data with diverse camera responses.

### Basic Usage

```bash
python gen_imaginary_raw.py \
    --datasets_for_interpolation Cube+ Gehler-Shi NUS Intel-TAU \
    --n_target_cams_per_scene 1 \
    --base_data_root ../../dataset/CCMNet/preprocessed_for_augmentation/ \
    --output_root_prefix ../../dataset/CCMNet/augmented_dataset/ \
    --gt_illum_json ./gt_illumination.json \
    --aug_illum_json ./sampled_illumination.json
```

### Key Parameters

- `--target_width`, `--target_height`: Size of output images (default: 384x256)
- `--n_target_cams_per_scene`: Number of synthetic cameras to generate per source scene
- `--datasets_for_interpolation`: List of datasets to use for camera pool (e.g., Cube+, Gehler-Shi, NUS, Intel-TAU)
- `--custom_source_cam_pool`, `--custom_target_cam_pool`: Optional custom camera lists for interpolation
- `--use_ratio_one_for_validation`: Set blending ratio to 0 (100% target camera)
- `--use_ratio_oneorzero_for_ablation`: Randomly choose between 0 or 1 for blending ratio
- `--base_data_root`: Directory containing preprocessed XYZ images
- `--output_root_prefix`: Base directory for augmented dataset output
- `--augmented_dataset_name`: Custom name for output dataset folder (auto-generated if not provided)

### Output Structure

The script creates a new directory under `output_root_prefix` with the following structure:

```
../dataset/CCMNet/augmented_dataset/
└── [dataset_name]/                    # e.g., "CGNI" for Cube+ Gehler-Shi NUS Intel-TAU
    ├── [source_cam]_[scene]_ratio_[blend]_[target_cam].png
    └── [source_cam]_[scene]_ratio_[blend]_[target_cam]_metadata.json
```

Each generated image includes:
- Synthetic RAW image interpolated between source and target cameras
- Metadata file containing:
  - Source and target camera information
  - Illuminant data
  - Color transformation matrices
  - Blending ratio used
  - Original XYZ image reference

### Example Use Cases

1. Data augmentation example for training the model used to test on NUS dataset.
   Using Cube+, Gehler-Shi, Intel-TAU datasets for augmentation:
```bash
python gen_imaginary_raw.py --datasets_for_interpolation Cube+ Gehler-Shi NUS Intel-TAU --n_target_cams_per_scene 3
```

2. Custom camera interpolation:
```bash
python gen_imaginary_raw.py --custom_source_cam_pool Canon1D Canon5D --custom_target_cam_pool Canon_5DSR Nikon_D810
``` 