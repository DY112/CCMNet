import cv2, os, sys, json, colour, time
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, current_process, Lock
import argparse
import warnings
from colour.utilities import ColourUsageWarning

warnings.filterwarnings("ignore", category=ColourUsageWarning)

lock = Lock()

ALL_CAMERAS_BY_DATASET = {
    'Intel-TAU': ['Canon_5DSR', 'Nikon_D810'],
    'Cube+': ['Canon550D'],
    'Gehler-Shi': ['Canon1D', 'Canon5D'],
    'NUS': ['Canon1DsMkIII', 'Canon600D', 'NikonD5200', 'SamsungNX2000', 'SonyA57', 'FujifilmXM1', 'OlympusEPL6', 'PanasonicGX1']
}

ALL_UNIQUE_CAMERAS_SORTED = sorted(list(set(cam for cams in ALL_CAMERAS_BY_DATASET.values() for cam in cams)))

def get_camera_name(cam):
    # for NUS dataset
    if cam == 'Canon1DsMkIII':
        return 'canon eos-1ds mark iii'
    elif cam == 'Canon600D':
        return 'canon eos 550d'
    elif cam == 'NikonD5200':
        return 'nikon d5200'
    elif cam == 'SamsungNX2000':
        return 'samsung nx2000'
    elif cam == 'SonyA57':
        return 'sony slt-a57'
    elif cam == 'FujifilmXM1':
        return 'fujifilm x-m1'
    elif cam == 'OlympusEPL6':
        return 'olympus e-pl6'
    elif cam == 'PanasonicGX1':
        return 'panasonic dmc-gx1'
    # for Intel TAU dataset
    elif cam == 'Canon_5DSR':
        return 'Canon_5DSR'
    elif cam == 'Nikon_D810':
        return 'Nikon_D810'
    elif cam == 'Sony_IMX135_BLCCSC':
        return 'Sony_IMX135_BLCCSC'
    # for Cube+ dataset
    elif cam == 'Canon550D':
        return 'canon eos 550d'
    # for Gehler-Shi dataset
    elif cam == 'Canon1D':
        return 'canon eos-1ds'
    elif cam == 'Canon5D':
        return 'canon eos 5d'
    return cam # Fallback for any camera names not explicitly mapped


def get_color_temp(query, illum_list):
    query_norm = query / np.linalg.norm(query)
    illum_list_norm = illum_list / np.linalg.norm(illum_list, axis=1, keepdims=True)
    angular_dist = np.arccos(np.clip(np.sum(query_norm * illum_list_norm, axis=1), -1, 1))
    return 2500 + np.argmin(angular_dist)

def interpolate_ccm(cct, ccm1, ccm2):
    """Interpolate color matrix based on color temperature."""
    g = (1 / cct - 1 / 2856) / (1 / 6504 - 1 / 2856)
    g = np.clip(g, 0, 1)
    return g * ccm1 + (1 - g) * ccm2

def camneutral_to_xyz(camneutral, cm1, cm2):
    xy = np.array([0.3127, 0.3290])
    for i in range(20):
        cct = colour.temperature.xy_to_CCT(xy, method="Kang 2002")
        
        color_matrix = interpolate_ccm(cct, cm1, cm2)
        color_matrix_inv = np.linalg.inv(color_matrix)
        xyz = np.dot(color_matrix_inv, camneutral)
        X, Y, Z = xyz
        if (X + Y + Z) == 0: # Avoid division by zero
            return (xyz / 1e-6 if Y == 0 else xyz / Y), cct # Handle black input
        xy_new = np.array([X / (X + Y + Z), Y / (X + Y + Z)])
        if np.allclose(xy, xy_new, atol=1e-6):
            return (xyz / (Y if Y!=0 else 1e-6)), cct # Normalize by Y
        xy = xy_new
    return (xyz / (Y if Y!=0 else 1e-6)), cct # Normalize by Y

def interpolate_calibration_matrix(calidict, color_temp):
    cm1 = np.array(calidict['ColorMatrix1'])
    cm2 = np.array(calidict['ColorMatrix2'])
    fm1 = np.array(calidict['ForwardMatrix1'])
    fm2 = np.array(calidict['ForwardMatrix2'])
    black_level = calidict['black_level']
    white_level = calidict['white_level']

    g = (1/color_temp - 1/2856) / (1/6504 - 1/2856)
    g = np.clip(g, 0, 1)
    cm = g * cm1 + (1 - g) * cm2
    fm = g * fm1 + (1 - g) * fm2

    return cm, fm, black_level, white_level

def synthesize_raw(args_tuple):
    xyz_path, illum_pool_sourcecam, calidict_sourcecam, calidict_targetcam, target_size_tuple, \
    target_root_path, source_cam_name, _, target_cam_name, \
    use_ratio_one_val, use_ratio_oneorzero_abl = args_tuple # Unpack
    
    base_fname = os.path.basename(xyz_path).split('.')[0]
    np.random.seed(os.getpid() * int(time.time() * 1e6) % 2**32)
    
    xyz_rgb = cv2.imread(xyz_path, cv2.IMREAD_UNCHANGED)
    if xyz_rgb is None:
        print(f"Warning: Could not read XYZ image {xyz_path}", file=sys.stderr)
        return
    xyz_rgb = xyz_rgb[:,:,::-1].astype(np.float32) / 65535
    
    mask_path = xyz_path.replace('xyz', 'mask')
    mask = cv2.imread(mask_path)
    if mask is None:
        print(f"Warning: Could not read mask {mask_path}, assuming full mask.", file=sys.stderr)
        mask = np.ones_like(xyz_rgb, dtype=np.uint8) * 255 # Create a full mask if not found
    mask = mask / 255.0
    
    xyz_rgb = xyz_rgb * mask
    if xyz_rgb.shape[0] > xyz_rgb.shape[1]:
        xyz_rgb = cv2.rotate(xyz_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if (xyz_rgb.shape[1], xyz_rgb.shape[0]) != target_size_tuple: # cv2 uses (width, height)
        xyz_rgb = cv2.resize(xyz_rgb, target_size_tuple, interpolation=cv2.INTER_LINEAR)
    
    if not illum_pool_sourcecam:
        print(f"Warning: Empty illumination pool for source camera {source_cam_name}. Skipping {base_fname}.", file=sys.stderr)
        return
    random_illum_rgb = np.array(illum_pool_sourcecam[np.random.randint(len(illum_pool_sourcecam))])
    
    illum_rgb_s = random_illum_rgb / random_illum_rgb[1]
    illum_xyz, illum_cct = camneutral_to_xyz(illum_rgb_s, np.array(calidict_sourcecam['ColorMatrix1']), np.array(calidict_sourcecam['ColorMatrix2']))
    
    cm_s, fm_s, black_level_s, white_level_s = interpolate_calibration_matrix(calidict_sourcecam, illum_cct)
    cm_t, fm_t, black_level_t, white_level_t = interpolate_calibration_matrix(calidict_targetcam, illum_cct)
    
    illum_rgb_t = np.dot(cm_t, illum_xyz)
    illum_rgb_t = illum_rgb_t / (illum_rgb_t[1] if illum_rgb_t[1] != 0 else 1e-6) # Avoid division by zero
    
    # Ensure white_level is not zero to prevent division by zero
    maxval_s = white_level_s - black_level_s if white_level_s > black_level_s else 1.0
    maxval_t = white_level_t - black_level_t if white_level_t > black_level_t else 1.0


    wb_s = np.clip(np.dot(xyz_rgb, np.linalg.inv(fm_s).T), 0, 1)
    wb_t = np.clip(np.dot(xyz_rgb, np.linalg.inv(fm_t).T), 0, 1)

    raw_s = wb_s * illum_rgb_s[None, None, :]
    raw_s = np.clip(raw_s, 0, 1)

    raw_t = wb_t * illum_rgb_t[None, None, :]
    raw_t = np.clip(raw_t, 0, 1)

    if use_ratio_one_val:
        blending_ratio = 0.0 # Should be 0 or 1 to pick one camera, often target for validation? Or is it ratio for source?
                            # Assuming 0 means 100% target camera if "validation" implies testing target camera properties
    elif use_ratio_oneorzero_abl:
        blending_ratio = np.random.choice([0., 1.])
    else:
        blending_ratio = np.random.rand()
        
    synth_raw = blending_ratio * raw_s + (1 - blending_ratio) * raw_t
    synth_raw = np.clip(synth_raw, 0, 1)
    synth_illum = blending_ratio * illum_rgb_s + (1 - blending_ratio) * illum_rgb_t
    
    # synth_wb = synth_raw / (synth_illum[None, None, :] + 1e-6) # Add epsilon for stability
    synth_cm1 = blending_ratio * np.array(calidict_sourcecam['ColorMatrix1']) + (1 - blending_ratio) * np.array(calidict_targetcam['ColorMatrix1'])
    synth_cm2 = blending_ratio * np.array(calidict_sourcecam['ColorMatrix2']) + (1 - blending_ratio) * np.array(calidict_targetcam['ColorMatrix2'])
    synth_cm = blending_ratio * cm_s + (1 - blending_ratio) * cm_t
    synth_fm1 = blending_ratio * np.array(calidict_sourcecam['ForwardMatrix1']) + (1 - blending_ratio) * np.array(calidict_targetcam['ForwardMatrix1'])
    synth_fm2 = blending_ratio * np.array(calidict_sourcecam['ForwardMatrix2']) + (1 - blending_ratio) * np.array(calidict_targetcam['ForwardMatrix2'])
    synth_fm = blending_ratio * fm_s + (1 - blending_ratio) * fm_t
    min_val = np.min(synth_raw).item() # Ensure scalar
    max_val = np.max(synth_raw).item() # Ensure scalar

    synth_raw_uint16 = (synth_raw * 65535).astype(np.uint16)

    synth_meta = {
        'illum_rgb_s': illum_rgb_s.tolist(),
        'illum_rgb_t': illum_rgb_t.tolist(),
        'illum_xyz': illum_xyz.tolist(),
        'illuminant_color_raw': synth_illum.tolist(),
        'cm_s': cm_s.tolist(),
        'cm_t': cm_t.tolist(),
        'cm1': synth_cm1.tolist(),
        'cm2': synth_cm2.tolist(),
        'cm': synth_cm.tolist(),
        'fm_s': fm_s.tolist(),
        'fm_t': fm_t.tolist(),
        'fm1': synth_fm1.tolist(),
        'fm2': synth_fm2.tolist(),
        'fm': synth_fm.tolist(),
        'color_temp': int(illum_cct),
        'blending_ratio': float(blending_ratio), # Ensure scalar
        'min_val': min_val,
        'max_val': max_val,
        'source_camera': source_cam_name,
        'target_camera': target_cam_name,
        'original_xyz_file': os.path.basename(xyz_path)
    }

    fname = f'{source_cam_name}_{base_fname}_ratio_{blending_ratio:.3f}_{target_cam_name}.png'
    target_path = os.path.join(target_root_path, fname)
    
    with lock:
        try:
            cv2.imwrite(target_path, synth_raw_uint16[:,:,::-1])
            with open(target_path.replace('.png', '_metadata.json'), 'w') as f:
                json.dump(synth_meta, f, indent=4)
        except Exception as e:
            print(f"Error writing file {target_path}: {e}", file=sys.stderr)

def get_args():
    parser = argparse.ArgumentParser(description="Generate imaginary camera RAW images by interpolation.")
    parser.add_argument('--target_width', type=int, default=384, help="Width of the synthesized RAW image.")
    parser.add_argument('--target_height', type=int, default=256, help="Height of the synthesized RAW image.")
    parser.add_argument('--n_target_cams_per_scene', type=int, default=1, help="Number of synthesized imaginary cameras per source scene/XYZ image.")
    parser.add_argument('--use_ratio_one_for_validation', action='store_true', help="If set, blending_ratio will be 0 for validation (uses 100% target camera).")
    parser.add_argument('--use_ratio_oneorzero_for_ablation', action='store_true', help="If set, blending_ratio will be randomly 0 or 1 for ablation.")
    
    parser.add_argument('--datasets_for_interpolation', type=str, nargs='+', required=False, default=['Cube+', 'Gehler-Shi', 'NUS', 'Intel-TAU'],
                        help="List of dataset names to use for interpolation if custom pools are not provided. Cameras from these datasets will form the pool.")
    parser.add_argument('--custom_source_cam_pool', type=str, nargs='+', default=None, help="Custom list of source camera names for interpolation.")
    parser.add_argument('--custom_target_cam_pool', type=str, nargs='+', default=None, help="Custom list of target camera names for interpolation.")
    
    parser.add_argument('--base_data_root', type=str, default='../../dataset/CCMNet/preprocessed_for_augmentation/', help="Root directory of preprocessed XYZ images.")
    parser.add_argument('--output_root_prefix', type=str, default='../../dataset/CCMNet/augmented_dataset/', help="Prefix for the output directory of augmented data.")
    parser.add_argument('--augmented_dataset_name', type=str, default=None, help="Name of the output augmented dataset folder. Auto-generated if not provided.")
    
    parser.add_argument('--gt_illum_json', type=str, default='./gt_illumination.json', help="Path to ground truth illumination JSON file.")
    parser.add_argument('--aug_illum_json', type=str, default='./sampled_illumination.json', help="Path to augmented (sampled) illumination JSON file.")
    parser.add_argument('--calibration_json', type=str, default='../../dataset/CCMNet/preprocessed_for_augmentation/calibration_metadata.json', help="Path to calibration metadata JSON file.")
    
    return parser.parse_args()

def get_camera_indices_str(camera_list):
    indices = [str(ALL_UNIQUE_CAMERAS_SORTED.index(cam)) for cam in camera_list if cam in ALL_UNIQUE_CAMERAS_SORTED]
    indices = "".join(sorted(indices, key=lambda x: int(x)))
    return indices

def prepare_synthesis_tasks(args):
    target_size_tuple = (args.target_width, args.target_height)

    if args.use_ratio_one_for_validation and args.use_ratio_oneorzero_for_ablation:
        print("Error: --use_ratio_one_for_validation and --use_ratio_oneorzero_for_ablation cannot be set simultaneously.", file=sys.stderr)
        sys.exit(1)

    if (args.custom_source_cam_pool is None and args.custom_target_cam_pool is not None) or \
       (args.custom_source_cam_pool is not None and args.custom_target_cam_pool is None):
        print("Error: --custom_source_cam_pool and --custom_target_cam_pool must be provided together, or neither.", file=sys.stderr)
        sys.exit(1)

    source_cameras = []
    target_cameras = []
    augmentation_set_name = ""

    if args.custom_source_cam_pool and args.custom_target_cam_pool:
        print("Using custom source and target camera pools. --datasets_for_interpolation will be ignored.")
        source_cameras = args.custom_source_cam_pool
        target_cameras = args.custom_target_cam_pool
        if args.augmented_dataset_name:
            augmentation_set_name = args.augmented_dataset_name
        else:
            s_indices = get_camera_indices_str(source_cameras)
            t_indices = get_camera_indices_str(target_cameras)
            augmentation_set_name = f"s{s_indices}_t{t_indices}"
    else:
        print(f"Using cameras from --datasets_for_interpolation: {args.datasets_for_interpolation}")
        cam_pool = []
        dataset_initials = []
        for dataset_name in sorted(list(set(args.datasets_for_interpolation))): # Sort for consistent naming
            if dataset_name in ALL_CAMERAS_BY_DATASET:
                cam_pool.extend(ALL_CAMERAS_BY_DATASET[dataset_name])
                
                initial = dataset_name[0].upper()
                dataset_initials.append(initial)
            else:
                print(f"Warning: Dataset '{dataset_name}' not found in ALL_CAMERAS_BY_DATASET. Skipping.", file=sys.stderr)
        
        source_cameras = list(set(cam_pool))
        target_cameras = list(set(cam_pool))
        
        if args.augmented_dataset_name:
            augmentation_set_name = args.augmented_dataset_name
        else:
            augmentation_set_name = "".join(dataset_initials) if dataset_initials else "default_pool"

    if not source_cameras:
        print("Error: Source camera pool is empty. Check dataset names or custom pool.", file=sys.stderr)
        sys.exit(1)
    if not target_cameras:
        print("Error: Target camera pool is empty. Check dataset names or custom pool.", file=sys.stderr)
        sys.exit(1)
        
    target_root_final = os.path.join(args.output_root_prefix, augmentation_set_name)
    os.makedirs(target_root_final, exist_ok=True)
    print(f"Augmented data will be saved to: {target_root_final}")

    try:
        gt_illum_meta = json.load(open(args.gt_illum_json, 'r'))
        aug_illum_meta = json.load(open(args.aug_illum_json, 'r'))
    except FileNotFoundError as e:
        print(f"Error: Illumination JSON file not found. {e}", file=sys.stderr)
        sys.exit(1)
        
    merged_illum_meta = {}
    for cam_list_key in list(gt_illum_meta.keys()) + list(aug_illum_meta.keys()):
        merged_illum_meta[cam_list_key] = gt_illum_meta.get(cam_list_key, []) + aug_illum_meta.get(cam_list_key, [])

    base_xyz_pool = []
    try:
        calibration_meta = json.load(open(args.calibration_json, 'r'))
    except FileNotFoundError as e:
        print(f"Error: Calibration JSON file not found. {e}", file=sys.stderr)
        sys.exit(1)

    for cam_name in source_cameras:
        cam_data_path = os.path.join(args.base_data_root, cam_name)
        if not os.path.isdir(cam_data_path):
            print(f"Warning: Camera data path not found for '{cam_name}' at '{cam_data_path}'. Skipping this camera for XYZ file collection.", file=sys.stderr)
            continue
        if cam_name not in calibration_meta:
            print(f"Warning: Calibration data not found for source camera '{cam_name}'. Skipping this camera.", file=sys.stderr)
            continue

        xyz_list = [os.path.join(cam_data_path, f) for f in os.listdir(cam_data_path) if f.endswith('xyz.png')]
        base_xyz_pool.extend([(xyz, cam_name) for xyz in xyz_list])

    if not base_xyz_pool:
        print("Error: No XYZ images found for the selected source cameras in the specified base_data_root. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    np.random.shuffle(base_xyz_pool)

    args_list_for_pool = []

    for (xyz_path, src_cam_name) in base_xyz_pool:
        xyz_fname = os.path.basename(xyz_path)
        xyz_basename = xyz_fname.split('.')[0]
        filename_parts = xyz_basename.split('_')
        scene_or_img_idx_str = filename_parts[1] if len(filename_parts) > 1 and filename_parts[1].isdigit() else xyz_basename

        illum_pool_key = get_camera_name(src_cam_name)
        if illum_pool_key not in merged_illum_meta or not merged_illum_meta[illum_pool_key]:
            print(f"Warning: Illumination data not found or empty for source camera '{src_cam_name}' (key: '{illum_pool_key}'). Skipping scenes from this camera.", file=sys.stderr)
            continue
            
        illum_pool_sourcecam_data = merged_illum_meta[illum_pool_key]
        calidict_sourcecam_data = calibration_meta[src_cam_name]

        for _ in range(args.n_target_cams_per_scene):
            eligible_target_cams = [tc for tc in target_cameras if tc != src_cam_name]
            if not eligible_target_cams:
                continue

            tgt_cam_name = np.random.choice(eligible_target_cams)
            if tgt_cam_name not in calibration_meta:
                 print(f"Warning: Calibration data not found for target camera '{tgt_cam_name}'. Skipping this target.", file=sys.stderr)
                 continue
            calidict_targetcam_data = calibration_meta[tgt_cam_name]
            
            args_list_for_pool.append((xyz_path, illum_pool_sourcecam_data, calidict_sourcecam_data, 
                                   calidict_targetcam_data, target_size_tuple, target_root_final, 
                                   src_cam_name, scene_or_img_idx_str, tgt_cam_name,
                                   args.use_ratio_one_for_validation, args.use_ratio_oneorzero_for_ablation))
    
    if not args_list_for_pool:
        print("Error: No valid argument pairs generated for processing. Check camera pools, calibration, and illumination data.", file=sys.stderr)
        sys.exit(1)

    return args_list_for_pool

if __name__ == '__main__':
    args = get_args()
    args_list_for_pool = prepare_synthesis_tasks(args)

    # The core augmentation logic remains here
    print(f"Starting RAW image synthesis with {len(args_list_for_pool)} tasks using {cpu_count()} CPUs...")
    with Pool(cpu_count()) as pool:
        for _ in tqdm(pool.imap_unordered(synthesize_raw, args_list_for_pool), total=len(args_list_for_pool), desc='Total Progress', leave=True):
            pass
    print("Finished RAW image synthesis.")
