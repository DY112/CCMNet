# CCMNet: Leveraging Calibrated Color Correction Matrices for Cross-Camera Color Constancy

<!-- ### [Imitating Human Videos via Cross-Embodiment Skill Representations](https://uniskill.github.io)    -->
*[Dongyoung Kim](https://www.dykim.me/home)*<sup>1</sup>, *[Mahmoud Afifi](https://www.mafifi.info/)*<sup>2</sup>, *[Dongyun Kim](https://sites.google.com/view/dongyunkim/home)*<sup>1</sup>, *[Michael S. Brown](https://www.eecs.yorku.ca/~mbrown/)*<sup>2</sup>, and *[Seon Joo Kim](https://www.ciplab.kr/members/professor)*<sup>1</sup>

<sup>1</sup>Yonsei University  &ensp; <sup>2</sup>AI Center - Toronto, Samsung Electronics

[[Paper](https://arxiv.org/abs/2504.07959)] | [[Project page](https://www.dykim.me/projects/ccmnet)]


## Overview

CCMNet is a deep learning-based methodology for maintaining color consistency in images captured by different cameras. This project aims to achieve cross-camera color constancy by leveraging camera-specific Color Correction Matrices (CCM). **The core idea is to learn a Camera Fingerprint Embedding (CFE) that represents the unique color characteristics of each camera.** This CFE is then used to predict an appropriate illumination chromaticity vector for an input image, thereby correcting its colors.

## Dataset

For detailed information about the datasets used in this project, please refer to the [`data_scripts/README.md`](./data_scripts/README.md) file.

### Supported Datasets
- NUS
- Cube+
- Intel-TAU
- Gehler-Shi

> **Note**: IMX135-BLCCSC camera from Intel-TAU dataset was not used in training and testing as its Color Correction Matrix (CCM) was not provided.

## Usage

### Training

The model is trained using a leave-one-out strategy, where one dataset is used for testing, and the remaining datasets are used for training.

#### Training Script

```bash
python train.py \
    -tr [TRAIN_DATASETS] \
    -augd [AUGMENTATION_DATA_PATH] \
    -lr [LEARNING_RATE] \
    -e [EPOCHS] \
    -cfefn [CFE_FEATURE_NUMBER]
```

#### Example

```bash
python train.py \
    -tr NUS-8 cube+ Intel-TAU \
    -augd CIN \
    -lr 5e-4 \
    -e 50 \
    -cfefn 8
```

#### Required Parameters

*   `-tr`, `--train_subsets`: List of training dataset names/subsets (e.g., `NUS-8 cube+ Intel-TAU`). Default: `None`.
*   `-augd`, `--augmentation-dir`: Path(s) to augmented dataset directories. (e.g., `CGN`, `CIN`). Default: `None`. These augmented datasets are generated using the `gen_imaginary_raw.py` script, which creates synthetic RAW images by interpolating between different camera characteristics. For detailed information about the dataset generation process, please refer to the [`data_scripts/README.md`](./data_scripts/README.md) file.
*   `-lr`, `--learning-rate`: Learning rate (default: `5e-4`).
*   `-e`, `--epochs`: Number of training epochs (default: 50).
*   `-cfefn`, `--cfe-feature-num`: Number of channels for the Camera Fingerprint Embedding (CFE) feature (default: 8).

<details>
<summary>Optional Parameters</summary>

*   `-b`, `--batch-size`: Batch size (default: 16).
*   `-ts`, `--test_subsets`: Test/Validation dataset name/subset (e.g., `Gehler-Shi`). Default: `None`.
*   `-opt`, `--optimizer`: Optimizer to use. Choices: `Adam`, `SGD` (default: `Adam`).
*   `-l2r`, `--l2reg`: L2 regularization factor (default: `5e-4`).
*   `-l`, `--load`: Load model from a .pth file (flag, default: disabled).
*   `-ml`, `--model-location`: Path to the model file to load (default: `None`).
*   `-vr`, `--validation-ratio`: Ratio of training data to use for validation if `val_dir_img` is not provided (default: 0.1).
*   `-vf`, `--validation-frequency`: Validate the model every N epochs (default: 1).
*   `-s`, `--input-size`: Size of the input histogram (number of bins) (default: 64).
*   `-lh`, `--load-hist`: Load pre-computed histograms if they exist (flag, default: enabled).
*   `-ibs`, `--increasing-batch-size`: Flag to gradually increase batch size during training (default: disabled).
*   `-gc`, `--grad-clip-value`: Gradient clipping threshold (0 for no clipping) (default: 0).
*   `-slf`, `--smoothness-factor-F`: Smoothness regularization factor for the convolutional filter F (default: 0.15).
*   `-slb`, `--smoothness-factor-B`: Smoothness regularization factor for the bias B (default: 0.02).
*   `-ntrd`, `--training-dir-in`: Root directory for input training images (default: `../../dataset/CCMNet/original_resized/`).
*   `-nvld`, `--validation-dir-in`: Root directory for input validation images. If `None`, validation set is split from training data (default: `None`).
*   `-nagd`, `--augmentation-dir-in`: Root directory for augmentation images specified by `-augd` (default: `../../dataset/CCMNet/augmented_dataset`).
*   `-n`, `--model-name`: Base name for the trained model and wandb logging (a datetime prefix will be added) (default: `CCMNet`).
*   `-g`, `--gpu`: GPU ID to use if CUDA is available (default: 0).
*   `-netd`, `--net-depth`: Depth of the encoder network (default: 4).
*   `-maxc`, `--max-convdepth`: Maximum depth of convolutional layers in the network (default: 32).
*   `--visualize-training`: Flag to enable visualization of training progress (TensorBoard images and local validation sample images) (default: disabled).
</details>

### Testing

The model is tested on datasets not seen during training. Various visualization options are available.

#### Testing Script

```bash
python test.py \
    -ts [TEST_DATASET] \
    -wb [WHITE_BALANCE_FLAG] \
    -n [MODEL_NAME_OR_ID] \
    -cfefn [CFE_FEATURE_NUMBER]
```

#### Example

```bash
python test.py \
    -ts cube+ \
    -wb True \
    -n test_Cube+ \
    -cfefn 8
```

#### Required Parameters

*   `-ts`, `--test_subsets`: Test dataset name/subset to evaluate (e.g., `cube+`). Default: `None`.
*   `-n`, `--model-name`: Name of the trained model to load (e.g., `test_Cube+`, `test_Gehler-Shi`, `test_NUS` in `models/` directory). Default: `None` (expects `.pth` extension).
*   `-cfefn`, `--cfe-feature-num`: Number of channels for the Camera Fingerprint Embedding (CFE) feature used by the loaded model (default: 8).

<details>
<summary>Optional Parameters</summary>

*   `-b`, `--batch-size`: Batch size for testing (default: 64).
*   `-s`, `--input-size`: Size of the input histogram/image (default: 64).
*   `-ntrd`, `--testing-dir-in`: Path to the root directory of the test dataset (e.g., `../../dataset/CCMNet/`). Default: `../dataset/CCMNet/`.
*   `-lh`, `--load-hist`: Load pre-computed histograms if available (flag, default: enabled).
*   `-g`, `--gpu`: GPU ID to use if CUDA is available (default: 0).
*   `-netd`, `--net-depth`: Depth of the encoder network in the loaded model (default: 4).
*   `-maxc`, `--max-convdepth`: Maximum depth of convolutional layers in the loaded model (default: 32).
</details>

#### Visualization Parameters

*   `-wb`, `--white-balance`: Flag to perform white balancing on test images and save them (default: `False`, example uses `True`).
*   `--visualize-gt`: Flag to save ground truth white-balanced images (default: disabled).
*   `--visualize-intermediate`: Flag to save intermediate outputs of the network (P, F_chroma, B, etc.) (default: disabled).
*   `--add-color-bar`: Flag to add a color bar indicating illuminant color to saved images (default: disabled). If this flag is present, color bars are added.

### Evaluation

After running the test script, the results are saved in the `results/[MODEL_NAME]` directory relative to the project root. To evaluate these results, use the evaluation script from the `evaluation` directory:

#### Evaluation Script

```bash
cd evaluation
python evaluation.py --model_name [MODEL_NAME]
```

#### Example

```bash
cd evaluation
python evaluation.py --model_name test_Cube+
```

This will calculate and display the following metrics for the test results:
- Mean angular error
- Median angular error
- Best 25% mean error
- Worst 25% mean error
- Worst 5% mean error
- Tri-mean error
- Maximum error

The evaluation script reads the ground truth (`gt.mat`) and predicted results (`results.mat`) from the `../results/[MODEL_NAME]` directory and computes various error metrics to assess the model's performance.

## Acknowledgements

This work is based on the **C5** [https://github.com/mahmoudnafifi/C5]. We thank the authors for their excellent work.

## Citation

TBD

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
