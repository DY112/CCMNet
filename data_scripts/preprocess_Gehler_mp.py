import os, json, colour, scipy, sys
import numpy as np
import rawpy
import cv2
from exiftool import ExifToolHelper
from tqdm import tqdm
from mcc_utils import get_binary_mask
from multiprocessing import Pool, cpu_count

def norm_matrix(mat):
    return mat / np.sum(mat, axis=1, keepdims=True)

def exif_to_nparray(exif_matrix):
    matrix_str = exif_matrix.split(' ')
    matrix = np.array([float(x) for x in matrix_str])
    if len(matrix) == 9:
        matrix = matrix.reshape((3, 3))
    elif len(matrix) == 3:
        matrix = np.diag(matrix)
    return matrix

def get_camrgb_for_colortemp(calibration_dict, illum_colortemp, calillum1_cct=2856, calillum2_cct=6504, return_calibration_matrices=False):
    assert illum_colortemp != 0
    ab = calibration_dict["ab"]
    cm1 = calibration_dict["cm1"]
    cm2 = calibration_dict["cm2"]
    fm1 = calibration_dict["fm1"]
    fm2 = calibration_dict["fm2"]

    illum_xy = colour.temperature.CCT_to_xy(illum_colortemp, method='Kang 2002')
    illum_XYZ = colour.xy_to_XYZ(illum_xy)

    g = (1/illum_colortemp - 1/calillum2_cct) / (1/calillum1_cct - 1/calillum2_cct)
    g = np.clip(g, 0, 1)
    cm = g * cm1 + (1 - g) * cm2
    fm = g * fm1 + (1 - g) * fm2

    WBCam2XYZ = fm
    XYZ2WBCam = np.linalg.inv(WBCam2XYZ)
    XYZ2Cam = cm
    Cam2XYZ = np.linalg.inv(XYZ2Cam)
    cam_neutral = np.dot(XYZ2Cam, illum_XYZ)

    if return_calibration_matrices:
        return cam_neutral, WBCam2XYZ, XYZ2WBCam, Cam2XYZ, XYZ2Cam

    return cam_neutral

def get_nearest_illum(query, illum_rgb_list):
    query = np.array(query)
    illum_rgb_list = np.array(illum_rgb_list)
    query_norm = query / np.linalg.norm(query)
    illum_rgb_list_norm = illum_rgb_list / np.linalg.norm(illum_rgb_list, axis=1, keepdims=True)

    angular_dist = np.arccos(np.clip(np.dot(illum_rgb_list_norm, query_norm), -1, 1))
    nearest_idx = np.argmin(angular_dist)
    return nearest_idx

def process_single_raw_image(args_tuple):
    raw_path, cam_name, dng_black_levels_passed, dng_white_level_passed, \
    gt_illum_current, camera_illum_rgb_list, \
    camera_wbraw2xyz_list, camera_xyz2wbraw_list, camera_xyz2raw_list, camera_raw2xyz_list, \
    img_target_size, cam_target_dir, cam_resized_dir, mcc_data_root, \
    do_save_preprocess, do_save_original, \
    camera_cm1_list, camera_cm2_list, camera_fm1_list, camera_fm2_list = args_tuple

    fname = os.path.basename(raw_path).split('.')[0]
    raw_img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)[:,:,::-1].astype(np.float32)
    
    black_level_val = dng_black_levels_passed[0]
    if cam_name == 'Canon5D':
        black_level_val = 129
    sat_level_val = dng_white_level_passed
    
    max_level = sat_level_val - black_level_val
    raw_img_processed = raw_img - black_level_val # Use a new variable for processed raw
    raw_img_processed = np.clip(raw_img_processed, 0, max_level)
    
    wb_img = raw_img_processed / (gt_illum_current / gt_illum_current[1])
    wb_img = np.clip(wb_img, 0, max_level)

    nearest_idx = get_nearest_illum(gt_illum_current, camera_illum_rgb_list)
    wbraw2xyz = np.array(camera_wbraw2xyz_list[nearest_idx])
    kelvin_temp = int(2500 + nearest_idx)
    
    single_img_meta = {
        'gt_illum': gt_illum_current.tolist(),
        'closest_kelvin_temp': kelvin_temp,
        'xyz2wbraw': camera_xyz2wbraw_list[nearest_idx],
        'wbraw2xyz': camera_wbraw2xyz_list[nearest_idx],
        'xyz2raw': camera_xyz2raw_list[nearest_idx],
        'raw2xyz': camera_raw2xyz_list[nearest_idx],
    }

    norm_wb_img = wb_img / max_level
    xyz_img = np.dot(norm_wb_img, wbraw2xyz.T) 
    xyz_img = np.clip(xyz_img, 0.,1.) * (2 ** 16 - 1)

    coord_file = os.path.join(mcc_data_root, 'coordinates', f'{fname}_macbeth.txt')
    with open(coord_file, 'r') as f:
        lines = f.readlines()
    points = [list(map(float, line.strip().split())) for line in lines[1:5]]
    points = [[int(p[0]/2), int(p[1]/2)] for p in points]
    points_0, points_1, points_2, points_3 = points[0], points[1], points[2], points[3]
    points[0], points[1] = points_1, points_0 # Swap
    mask = get_binary_mask(h=raw_img.shape[0], w=raw_img.shape[1], point_list=points)[:,:,None]

    if do_save_preprocess:
        raw_img_to_save = cv2.resize(raw_img_processed, img_target_size, interpolation=cv2.INTER_AREA)
        wb_img_to_save = cv2.resize(wb_img, img_target_size, interpolation=cv2.INTER_AREA)
        xyz_img_to_save = cv2.resize(xyz_img, img_target_size, interpolation=cv2.INTER_AREA)
        mask_to_save = cv2.resize(mask, img_target_size, interpolation=cv2.INTER_AREA)
        if mask_to_save.ndim == 2: # Ensure mask is 3 channel for consistency if needed, or handle saving
            mask_to_save = mask_to_save[:,:,np.newaxis]


        cv2.imwrite(os.path.join(cam_target_dir, f'{fname}_raw.png'), raw_img_to_save[:,:,::-1].astype(np.uint16))
        cv2.imwrite(os.path.join(cam_target_dir, f'{fname}_wb.png'), wb_img_to_save[:,:,::-1].astype(np.uint16))
        cv2.imwrite(os.path.join(cam_target_dir, f'{fname}_xyz.png'), xyz_img_to_save[:,:,::-1].astype(np.uint16))
        cv2.imwrite(os.path.join(cam_target_dir, f'{fname}_mask.png'), (mask_to_save*255).astype(np.uint8))

    if do_save_original:
        # Apply mask before resizing for original raw
        raw_img_masked = raw_img_processed * mask # Use processed raw before normalization for saving original
        raw_img_resized = cv2.resize(raw_img_masked, img_target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize based on its own max_level, not global max, to preserve dynamic range of the masked content
        # if np.max(raw_img_resized) > 0: # Avoid division by zero if image is all black after mask
        #     raw_img_resized_norm = raw_img_resized / np.max(raw_img_resized) * 65535.0
        # else:
        #     raw_img_resized_norm = raw_img_resized # Already zero
        # Normalizing to its own max_level is better than to fixed 65535, or use max_level
        raw_img_resized_norm = np.clip(raw_img_resized / max_level * 65535.0, 0, 65535.0)


        resized_meta = {
            'illuminant_color_raw': (gt_illum_current/gt_illum_current[1]).tolist(),
            'cm1': camera_cm1_list,
            'cm2': camera_cm2_list,
            'fm1': camera_fm1_list,
            'fm2': camera_fm2_list
        }
        target_name = f'{fname}_sensorname_{cam_name}.png'
        cv2.imwrite(os.path.join(cam_resized_dir, target_name), raw_img_resized_norm[:,:,::-1].astype(np.uint16))
    
        with open(os.path.join(cam_resized_dir, target_name.replace('.png', '_metadata.json')), 'w') as f:
            json.dump(resized_meta, f, indent=4)
            
    return fname, single_img_meta

if __name__ == '__main__':
    target_size = (384, 256)
    SAVE_PREPROCESS = True
    SAVE_ORIGINAL = True
    data_root = '../../dataset/Gehler_Shi/'
    # Changed target_dir to be a base for preprocessed data
    preprocessed_base_dir = '../../dataset/CCMNet/preprocessed_for_augmentation/'
    # Renamed resized_dir to be more specific as it's for Gehler-Shi dataset
    gehler_shi_resized_dir = '../../dataset/CCMNet/original_resized/Gehler-Shi/'
    
    os.makedirs(preprocessed_base_dir, exist_ok=True)
    os.makedirs(gehler_shi_resized_dir, exist_ok=True)
    
    cams = ['Canon1D', 'Canon5D']

    calibration_metadata_file = os.path.join(preprocessed_base_dir, 'calibration_metadata.json')
    if os.path.exists(calibration_metadata_file):
        all_cam_calibration_data = json.load(open(calibration_metadata_file))
    else:
        all_cam_calibration_data = {}

    real_illum_data_path = os.path.join(data_root, 'real_illum_568.mat')
    real_illum_data_mat = scipy.io.loadmat(real_illum_data_path)
    real_illum_list_full = real_illum_data_mat['real_rgb']
    real_illum_list_full = real_illum_list_full / real_illum_list_full[:, 1][:, None]

    with ExifToolHelper() as et:
        cam_pbar = tqdm(cams, leave=True, desc="Processing Cameras")
        for cam in cam_pbar:
            # cam_pbar.set_description(f"Processing {cam}") # tqdm updates itself
            
            # Per-camera target directory for preprocessed images
            target_cam_dir_for_preprocess = os.path.join(preprocessed_base_dir, cam)
            os.makedirs(target_cam_dir_for_preprocess, exist_ok=True)

            # For Gehler-Shi, original resized images go into a shared dataset-specific folder.
            # The camera name is part of the filename.
            # So, gehler_shi_resized_dir is used directly for SAVE_ORIGINAL path constructions.

            cam_dng_dir = os.path.join(data_root, cam, 'dng')
            dng_list = [os.path.join(cam_dng_dir, f) for f in os.listdir(cam_dng_dir) if f.endswith('.dng')]
            if not dng_list:
                print(f"Warning: No DNG files found for {cam} in {cam_dng_dir}. Skipping camera.", file=sys.stderr)
                continue
            
            first_dng_file = dng_list[0]
            try:
                dng_raw_obj_temp = rawpy.imread(first_dng_file)
                dng_black_levels = dng_raw_obj_temp.black_level_per_channel
                dng_white_level = dng_raw_obj_temp.white_level
                del dng_raw_obj_temp 
            except Exception as e:
                print(f"Error reading DNG file {first_dng_file} for {cam}: {e}. Skipping camera.", file=sys.stderr)
                continue
            
            meta_dict = et.get_metadata(first_dng_file)[0]
            illum1 = meta_dict["EXIF:CalibrationIlluminant1"]
            illum2 = meta_dict["EXIF:CalibrationIlluminant2"]
            cm1 = exif_to_nparray(meta_dict["EXIF:ColorMatrix1"])
            cm2 = exif_to_nparray(meta_dict["EXIF:ColorMatrix2"])
            fm1 = exif_to_nparray(meta_dict["EXIF:ForwardMatrix1"])
            fm2 = exif_to_nparray(meta_dict["EXIF:ForwardMatrix2"])
            ab = exif_to_nparray(meta_dict["EXIF:AnalogBalance"])

            calibration_dict_internal = {"ab": ab, "cm1": cm1, "cm2": cm2, "fm1": fm1, "fm2": fm2}

            cam_illum_rgb_list, cam_xyz2wbraw_list, cam_wbraw2xyz_list, cam_xyz2raw_list, cam_raw2xyz_list = [], [], [], [], []
            for colortemp_val in range(2500, 7501):
                illumrgb, wbraw2xyz_m, xyz2wbraw_m, raw2xyz_m, xyz2raw_m = get_camrgb_for_colortemp(calibration_dict_internal, colortemp_val, return_calibration_matrices=True)
                illumrgb = illumrgb / illumrgb[1]
                cam_illum_rgb_list.append(illumrgb.tolist())
                cam_xyz2wbraw_list.append(xyz2wbraw_m.tolist())
                cam_wbraw2xyz_list.append(wbraw2xyz_m.tolist())
                cam_xyz2raw_list.append(xyz2raw_m.tolist())
                cam_raw2xyz_list.append(raw2xyz_m.tolist())

            all_cam_calibration_data[cam] = {
                'black_level': dng_black_levels[0] if cam != 'Canon5D' else 129, # Store effective black level used
                'white_level': dng_white_level,
                'CalibrationIlluminant1': illum1,
                'CalibrationIlluminant2': illum2,
                'ColorMatrix1': cm1.tolist(),
                'ColorMatrix2': cm2.tolist(),
                'ForwardMatrix1': fm1.tolist(),
                'ForwardMatrix2': fm2.tolist(),
                'AnalogBalance': ab.tolist(),
                'IlluminantRGB': cam_illum_rgb_list, # This is a list of many illuminants, maybe not needed here if per-image meta is enough
            }
            
            cam_raw_img_dir = os.path.join(data_root, cam, 'png')
            cam_raw_img_paths = sorted([os.path.join(cam_raw_img_dir, f) for f in os.listdir(cam_raw_img_dir) if f.endswith('.png')])
            
            current_cam_real_illum_list = []
            if cam == 'Canon1D':
                current_cam_real_illum_list = real_illum_list_full[:86]
            elif cam == 'Canon5D':
                current_cam_real_illum_list = real_illum_list_full[86:]
            
            if len(cam_raw_img_paths) != len(current_cam_real_illum_list):
                print(f"Warning: Mismatch between number of raw images ({len(cam_raw_img_paths)}) and real illuminants ({len(current_cam_real_illum_list)}) for {cam}. Skipping image processing for this camera.", file=sys.stderr)
                continue

            tasks_for_pool = []
            for idx, raw_path_item in enumerate(cam_raw_img_paths):
                gt_illum_val = current_cam_real_illum_list[idx]
                task_args = (
                    raw_path_item, cam, dng_black_levels, dng_white_level,
                    gt_illum_val, cam_illum_rgb_list,
                    cam_wbraw2xyz_list, cam_xyz2wbraw_list, cam_xyz2raw_list, cam_raw2xyz_list,
                    target_size, target_cam_dir_for_preprocess, gehler_shi_resized_dir, # Use the correct resized dir
                    data_root, # for MCC coordinates
                    SAVE_PREPROCESS, SAVE_ORIGINAL,
                    cm1.tolist(), cm2.tolist(), fm1.tolist(), fm2.tolist()
                )
                tasks_for_pool.append(task_args)

            per_cam_meta_results = {}
            if tasks_for_pool:
                with Pool(processes=cpu_count()) as pool:
                    results_iterator = pool.imap_unordered(process_single_raw_image, tasks_for_pool)
                    for fname_res, single_img_meta_res in tqdm(results_iterator, total=len(tasks_for_pool), desc=f"Processing images for {cam}", leave=False):
                        per_cam_meta_results[fname_res] = single_img_meta_res
            
            with open(os.path.join(target_cam_dir_for_preprocess, 'metadata.json'), 'w') as f:
                json.dump(per_cam_meta_results, f, indent=4)

    with open(calibration_metadata_file, 'w') as f:
        json.dump(all_cam_calibration_data, f, indent=4)

    print('Done')