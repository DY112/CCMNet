import os, json, colour, scipy, sys
import numpy as np
import rawpy
import cv2
from exiftool import ExifToolHelper
from tqdm import tqdm
sys.path.append('../') # Ensure mcc_utils can be found if it's in parent dir
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

def process_single_raw_image_cube(args_tuple):
    raw_path, cam_name_fixed, img_idx_from_fname, current_gt_illum, \
    dng_black_level_val, dng_saturation_level_val, max_pixel_val, \
    camera_illum_rgb_list_ref, camera_wbraw2xyz_list_ref, camera_xyz2wbraw_list_ref, \
    camera_xyz2raw_list_ref, camera_raw2xyz_list_ref, \
    img_target_size_ref, cam_preprocess_target_dir_ref, cam_original_resized_dir_ref, \
    do_save_preprocess_flag, do_save_original_flag, \
    dng_cm1_ref_list, dng_cm2_ref_list, dng_fm1_ref_list, dng_fm2_ref_list = args_tuple

    original_fname_base = os.path.basename(raw_path).split('.')[0] # Used as key for cam_meta
    
    raw_img_data = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)[:,:,::-1].astype(np.float32) - dng_black_level_val
    raw_img_processed = np.clip(raw_img_data, 0, max_pixel_val)
    
    wb_img = raw_img_processed / (current_gt_illum / current_gt_illum[1])
    wb_img = np.clip(wb_img, 0, max_pixel_val)

    nearest_idx = get_nearest_illum(current_gt_illum, camera_illum_rgb_list_ref)
    wbraw2xyz_m = np.array(camera_wbraw2xyz_list_ref[nearest_idx])
    kelvin_temp_val = int(2500 + nearest_idx)
    
    img_meta_data = {
        'gt_illum': current_gt_illum.tolist(),
        'closest_kelvin_temp': kelvin_temp_val,
        'xyz2wbraw': camera_xyz2wbraw_list_ref[nearest_idx],
        'wbraw2xyz': camera_wbraw2xyz_list_ref[nearest_idx],
        'xyz2raw': camera_xyz2raw_list_ref[nearest_idx],
        'raw2xyz': camera_raw2xyz_list_ref[nearest_idx],
    }

    norm_wb_img = wb_img / max_pixel_val
    xyz_img = np.dot(norm_wb_img, wbraw2xyz_m.T) 
    xyz_img = np.clip(xyz_img, 0.,1.) * (2 ** 16 - 1)

    mask_img = np.ones(raw_img_processed.shape[:2], dtype=np.uint8)
    mask_img[1050:, 2050:] = 0 # Cube+ specific mask
    mask_img_3channel = mask_img[:,:,None]

    # Formatted filename for saving, using the original index from filename (which is fname itself for Cube+)
    # The problem states fname is int(fname), so idx is int(original_fname_base)
    save_fname_base = f'{cam_name_fixed}_{img_idx_from_fname:04}'

    if do_save_preprocess_flag:
        raw_img_save = cv2.resize(raw_img_processed, img_target_size_ref, interpolation=cv2.INTER_AREA)
        wb_img_save = cv2.resize(wb_img, img_target_size_ref, interpolation=cv2.INTER_AREA)
        xyz_img_save = cv2.resize(xyz_img, img_target_size_ref, interpolation=cv2.INTER_AREA)
        mask_to_save = cv2.resize(mask_img, img_target_size_ref, interpolation=cv2.INTER_AREA) # Resize 2D mask
        
        cv2.imwrite(os.path.join(cam_preprocess_target_dir_ref, f'{save_fname_base}_raw.png'), raw_img_save[:,:,::-1].astype(np.uint16))
        cv2.imwrite(os.path.join(cam_preprocess_target_dir_ref, f'{save_fname_base}_wb.png'), wb_img_save[:,:,::-1].astype(np.uint16))
        cv2.imwrite(os.path.join(cam_preprocess_target_dir_ref, f'{save_fname_base}_xyz.png'), xyz_img_save[:,:,::-1].astype(np.uint16))
        cv2.imwrite(os.path.join(cam_preprocess_target_dir_ref, f'{save_fname_base}_mask.png'), (mask_to_save*255).astype(np.uint8))

    if do_save_original_flag:
        raw_img_masked_orig = raw_img_processed * mask_img_3channel # Use 3-channel mask for multiplication
        raw_img_resized_orig = cv2.resize(raw_img_masked_orig, img_target_size_ref, interpolation=cv2.INTER_AREA)
        if np.max(raw_img_resized_orig) > 0:
            raw_img_resized_norm_orig = raw_img_resized_orig / np.max(raw_img_resized_orig) * 65535.0
        else:
            raw_img_resized_norm_orig = raw_img_resized_orig
        raw_img_resized_norm_orig = np.clip(raw_img_resized_norm_orig, 0, 65535.0)
        
        resized_img_meta = {
            'illuminant_color_raw': (current_gt_illum/current_gt_illum[1]).tolist(),
            'cm1': dng_cm1_ref_list, 'cm2': dng_cm2_ref_list,
            'fm1': dng_fm1_ref_list, 'fm2': dng_fm2_ref_list
        }
        # Original script uses `fname` (which is original_fname_base) for target_name for original resized
        target_filename_orig = f'{original_fname_base}_sensorname_{cam_name_fixed}.png'
        cv2.imwrite(os.path.join(cam_original_resized_dir_ref, target_filename_orig), raw_img_resized_norm_orig[:,:,::-1].astype(np.uint16))

        with open(os.path.join(cam_original_resized_dir_ref, target_filename_orig.replace('.png', '_metadata.json')), 'w') as f_meta:
            json.dump(resized_img_meta, f_meta, indent=4)
            
    return original_fname_base, img_meta_data # Return original base name for cam_meta key

if __name__ == '__main__':
    target_size_config = (384, 256)
    SAVE_PREPROCESS_CONFIG = True
    SAVE_ORIGINAL_CONFIG = True
    
    base_data_root = '../../dataset/cube+/'
    preprocessed_output_base_dir = '../../dataset/CCMNet/preprocessed_for_augmentation/'
    original_resized_output_base_dir = '../../dataset/CCMNet/original_resized/cube+/'
    
    os.makedirs(preprocessed_output_base_dir, exist_ok=True)
    os.makedirs(original_resized_output_base_dir, exist_ok=True)
    
    # Cube+ dataset only has Canon550D
    camera_name_fixed = 'Canon550D' 

    calibration_metadata_path = os.path.join(preprocessed_output_base_dir, 'calibration_metadata.json')
    if os.path.exists(calibration_metadata_path):
        all_cameras_calibration_data = json.load(open(calibration_metadata_path))
    else:
        all_cameras_calibration_data = {}

    with ExifToolHelper() as et_instance:
        # No camera loop needed, process directly for Canon550D
        print(f"Processing camera: {camera_name_fixed}")
        current_cam_preprocess_dir = os.path.join(preprocessed_output_base_dir, camera_name_fixed)
        os.makedirs(current_cam_preprocess_dir, exist_ok=True)

        cam_dng_search_dir = os.path.join(base_data_root, 'dng')
        dng_files_in_path = [os.path.join(cam_dng_search_dir, f) for f in os.listdir(cam_dng_search_dir) if f.endswith('.dng')]
        if not dng_files_in_path:
            print(f"Error: No .dng files found in {cam_dng_search_dir} for Cube+ dataset. Exiting.", file=sys.stderr)
            sys.exit(1)
        
        first_dng_path = dng_files_in_path[0]
        try:
            dng_raw_obj = rawpy.imread(first_dng_path)
            dng_black_level = dng_raw_obj.black_level_per_channel[0]
            dng_saturation_level = dng_raw_obj.white_level
            max_pixel_value_calc = dng_saturation_level - dng_black_level
            del dng_raw_obj
        except Exception as e:
            print(f"Error reading DNG {first_dng_path} for {camera_name_fixed}: {e}. Exiting.", file=sys.stderr)
            sys.exit(1)
        
        exif_metadata = et_instance.get_metadata(first_dng_path)[0]
        calib_illum1 = exif_metadata["EXIF:CalibrationIlluminant1"]
        calib_illum2 = exif_metadata["EXIF:CalibrationIlluminant2"]
        dng_cm1_arr = exif_to_nparray(exif_metadata["EXIF:ColorMatrix1"])
        dng_cm2_arr = exif_to_nparray(exif_metadata["EXIF:ColorMatrix2"])
        dng_fm1_arr = exif_to_nparray(exif_metadata["EXIF:ForwardMatrix1"])
        dng_fm2_arr = exif_to_nparray(exif_metadata["EXIF:ForwardMatrix2"])
        dng_ab_arr = exif_to_nparray(exif_metadata["EXIF:AnalogBalance"])

        cam_specific_calib_dict = {
            "ab": dng_ab_arr, "cm1": dng_cm1_arr, "cm2": dng_cm2_arr,
            "fm1": dng_fm1_arr, "fm2": dng_fm2_arr,
        }

        gen_illum_rgb_list, gen_xyz2wbraw_list, gen_wbraw2xyz_list, gen_xyz2raw_list, gen_raw2xyz_list = [], [], [], [], []
        for temp in range(2500, 7501):
            illum_rgb, wbraw2xyz_mat, xyz2wbraw_mat, raw2xyz_mat, xyz2raw_mat = get_camrgb_for_colortemp(cam_specific_calib_dict, temp, return_calibration_matrices=True)
            gen_illum_rgb_list.append((illum_rgb / illum_rgb[1]).tolist())
            gen_xyz2wbraw_list.append(xyz2wbraw_mat.tolist())
            gen_wbraw2xyz_list.append(wbraw2xyz_mat.tolist())
            gen_xyz2raw_list.append(xyz2raw_mat.tolist())
            gen_raw2xyz_list.append(raw2xyz_mat.tolist())

        all_cameras_calibration_data[camera_name_fixed] = {
            'black_level': dng_black_level,
            'white_level': dng_saturation_level,
            'CalibrationIlluminant1': calib_illum1, 'CalibrationIlluminant2': calib_illum2,
            'ColorMatrix1': dng_cm1_arr.tolist(), 'ColorMatrix2': dng_cm2_arr.tolist(),
            'ForwardMatrix1': dng_fm1_arr.tolist(), 'ForwardMatrix2': dng_fm2_arr.tolist(),
            'AnalogBalance': dng_ab_arr.tolist(), 'IlluminantRGB': gen_illum_rgb_list,
        }
        
        cam_png_dir_path = os.path.join(base_data_root, 'PNG')
        gt_txt_file_path = os.path.join(base_data_root, 'cube+_gt.txt')
        try:
            with open(gt_txt_file_path, 'r') as gt_f:
                all_gt_illuminants = [list(map(float, line.strip().split())) for line in gt_f.readlines()]
        except FileNotFoundError:
            print(f"Error: GT file {gt_txt_file_path} not found for Cube+ dataset. Exiting.", file=sys.stderr)
            sys.exit(1)

        png_files_in_dir = sorted([os.path.join(cam_png_dir_path, f) for f in os.listdir(cam_png_dir_path) if f.endswith('.PNG')])
        current_cam_img_meta_data = {}
        tasks_for_cube_pool_current_cam = []

        for png_path in png_files_in_dir:
            fname_base = os.path.basename(png_path).split('.')[0]
            try:
                img_idx = int(fname_base) # For Cube+, fname is the index
            except ValueError:
                print(f"Warning: Could not parse image index from filename {fname_base}. Skipping {png_path}", file=sys.stderr)
                continue

            if not (0 < img_idx <= len(all_gt_illuminants)):
                print(f"Warning: Image index {img_idx} from {fname_base} is out of bounds for GT illuminant list (len: {len(all_gt_illuminants)}). Skipping.", file=sys.stderr)
                continue
            
            current_img_gt_illum_val = np.array(all_gt_illuminants[img_idx-1])
            
            task_args_tuple = (
                png_path, camera_name_fixed, img_idx, current_img_gt_illum_val,
                dng_black_level, dng_saturation_level, max_pixel_value_calc,
                gen_illum_rgb_list, gen_wbraw2xyz_list, gen_xyz2wbraw_list,
                gen_xyz2raw_list, gen_raw2xyz_list,
                target_size_config, current_cam_preprocess_dir, original_resized_output_base_dir,
                SAVE_PREPROCESS_CONFIG, SAVE_ORIGINAL_CONFIG,
                dng_cm1_arr.tolist(), dng_cm2_arr.tolist(), dng_fm1_arr.tolist(), dng_fm2_arr.tolist()
            )
            tasks_for_cube_pool_current_cam.append(task_args_tuple)
        
        if tasks_for_cube_pool_current_cam:
            with Pool(processes=cpu_count()) as pool:
                img_results_iterator = pool.imap_unordered(process_single_raw_image_cube, tasks_for_cube_pool_current_cam)
                for fname_result_key, single_img_meta_result in tqdm(img_results_iterator, total=len(tasks_for_cube_pool_current_cam), desc=f"Processing images for {camera_name_fixed}"):
                    current_cam_img_meta_data[fname_result_key] = single_img_meta_result # Use original_fname_base as key
        
        with open(os.path.join(current_cam_preprocess_dir, 'metadata.json'), 'w') as f_json:
            json.dump(current_cam_img_meta_data, f_json, indent=4)

    with open(calibration_metadata_path, 'w') as f_calib_json:
        json.dump(all_cameras_calibration_data, f_calib_json, indent=4)

    print('Done') 