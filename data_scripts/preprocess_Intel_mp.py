import os, json, colour, scipy, sys
import numpy as np
import rawpy
import cv2
from exiftool import ExifToolHelper
from tqdm import tqdm
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

def process_single_raw_image_intel(args_tuple):
    raw_path, cam_name, dng_saturation_level, gt_illum_val, \
    camera_illum_rgb_list_ref, camera_wbraw2xyz_list_ref, camera_xyz2wbraw_list_ref, \
    camera_xyz2raw_list_ref, camera_raw2xyz_list_ref, \
    img_target_size_ref, cam_preprocess_target_dir_ref, cam_original_resized_dir_ref, \
    do_save_preprocess_flag, do_save_original_flag, \
    dng_cm1_ref, dng_cm2_ref, dng_fm1_ref, dng_fm2_ref = args_tuple

    fname = os.path.basename(raw_path).split('.')[0]
    raw_img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)[:,:,::-1].astype(np.float32)
    # Intel dataset TIFFs are already black level subtracted by provider.

    # gt_illum_val is already normalized by G
    
    wb_img = raw_img / (gt_illum_val / gt_illum_val[1]) # Ensure G is 1 for wb_img calculation
    wb_img = np.clip(wb_img, 0, dng_saturation_level)

    nearest_idx = get_nearest_illum(gt_illum_val, camera_illum_rgb_list_ref)
    wbraw2xyz_m = np.array(camera_wbraw2xyz_list_ref[nearest_idx])
    kelvin_temp_val = int(2500 + nearest_idx)
    
    img_meta_data = {
        'gt_illum': gt_illum_val.tolist(),
        'closest_kelvin_temp': kelvin_temp_val,
        'xyz2wbraw': camera_xyz2wbraw_list_ref[nearest_idx],
        'wbraw2xyz': camera_wbraw2xyz_list_ref[nearest_idx],
        'xyz2raw': camera_xyz2raw_list_ref[nearest_idx],
        'raw2xyz': camera_raw2xyz_list_ref[nearest_idx],
    }

    norm_wb_img = wb_img / dng_saturation_level
    xyz_img = np.dot(norm_wb_img, wbraw2xyz_m.T) 
    xyz_img = np.clip(xyz_img, 0.,1.) * (2 ** 16 - 1)

    mask_img = np.ones_like(raw_img[:,:,0], dtype=np.uint8)[:,:,None] # Full mask

    if do_save_preprocess_flag:
        raw_img_save = cv2.resize(raw_img, img_target_size_ref, interpolation=cv2.INTER_AREA)
        wb_img_save = cv2.resize(wb_img, img_target_size_ref, interpolation=cv2.INTER_AREA)
        xyz_img_save = cv2.resize(xyz_img, img_target_size_ref, interpolation=cv2.INTER_AREA)
        mask_img_save = cv2.resize(mask_img, img_target_size_ref, interpolation=cv2.INTER_AREA)
        if mask_img_save.ndim == 2: mask_img_save = mask_img_save[:,:,np.newaxis]


        cv2.imwrite(os.path.join(cam_preprocess_target_dir_ref, f'{fname}_raw.png'), raw_img_save[:,:,::-1].astype(np.uint16))
        cv2.imwrite(os.path.join(cam_preprocess_target_dir_ref, f'{fname}_wb.png'), wb_img_save[:,:,::-1].astype(np.uint16))
        # wb_vis might be too bright if sat_level is high, consider normalizing differently if an issue
        wb_vis_save_path = os.path.join(cam_preprocess_target_dir_ref, f'{fname}_wb_vis.png')
        if np.max(wb_img_save) > 0 :
             cv2.imwrite(wb_vis_save_path, (wb_img_save[:,:,::-1]/np.max(wb_img_save)*255).astype(np.uint8))
        else: # all zero image
             cv2.imwrite(wb_vis_save_path, (wb_img_save[:,:,::-1]).astype(np.uint8))

        cv2.imwrite(os.path.join(cam_preprocess_target_dir_ref, f'{fname}_xyz.png'), xyz_img_save[:,:,::-1].astype(np.uint16))
        cv2.imwrite(os.path.join(cam_preprocess_target_dir_ref, f'{fname}_mask.png'), (mask_img_save*255).astype(np.uint8))

    if do_save_original_flag:
        # For original, resize raw_img directly as black level is 0. Apply full mask.
        raw_img_masked_orig = raw_img * mask_img 
        raw_img_resized_orig = cv2.resize(raw_img_masked_orig, img_target_size_ref, interpolation=cv2.INTER_AREA)
        if np.max(raw_img_resized_orig) > 0:
            raw_img_resized_norm_orig = raw_img_resized_orig / np.max(raw_img_resized_orig) * 65535.0
        else:
            raw_img_resized_norm_orig = raw_img_resized_orig
        raw_img_resized_norm_orig = np.clip(raw_img_resized_norm_orig, 0, 65535.0)
        
        resized_img_meta = {
            'illuminant_color_raw': (gt_illum_val/gt_illum_val[1]).tolist(),
            'cm1': dng_cm1_ref, 'cm2': dng_cm2_ref,
            'fm1': dng_fm1_ref, 'fm2': dng_fm2_ref
        }
        target_filename = f'{fname}_sensorname_{cam_name}.png'
        cv2.imwrite(os.path.join(cam_original_resized_dir_ref, target_filename), raw_img_resized_norm_orig[:,:,::-1].astype(np.uint16))
        
        with open(os.path.join(cam_original_resized_dir_ref, target_filename.replace('.png', '_metadata.json')), 'w') as f_meta:
            json.dump(resized_img_meta, f_meta, indent=4)
            
    return fname, img_meta_data

if __name__ == '__main__':
    target_size_config = (384, 256)
    SAVE_PREPROCESS_CONFIG = True
    SAVE_ORIGINAL_CONFIG = True
    
    base_data_root = '../../dataset/Intel-TAU/'
    preprocessed_output_base_dir = '../../dataset/CCMNet/preprocessed_for_augmentation/'
    original_resized_output_base_dir = '../../dataset/CCMNet/original_resized/Intel-TAU/'
    
    os.makedirs(preprocessed_output_base_dir, exist_ok=True)
    os.makedirs(original_resized_output_base_dir, exist_ok=True)
    
    camera_list = ['Canon_5DSR', 'Nikon_D810'] # 'Sony_IMX135_BLCCSC' might need different handling or is excluded
    sub_directory_list = ['field_1_cameras', 'field_3_cameras', 'lab_printouts', 'lab_realscene']

    calibration_metadata_path = os.path.join(preprocessed_output_base_dir, 'calibration_metadata.json')
    if os.path.exists(calibration_metadata_path):
        all_cameras_calibration_data = json.load(open(calibration_metadata_path))
    else:
        all_cameras_calibration_data = {}

    with ExifToolHelper() as et_instance:
        overall_cam_pbar = tqdm(camera_list, leave=True, desc="Processing Cameras")
        for cam_name_iter in overall_cam_pbar:
            # overall_cam_pbar.set_description(f"Processing {cam_name_iter}") # tqdm updates itself
            
            current_cam_preprocess_dir = os.path.join(preprocessed_output_base_dir, cam_name_iter)
            os.makedirs(current_cam_preprocess_dir, exist_ok=True)
            # Original resized images for Intel-TAU go into a shared dataset-specific folder.

            cam_data_path = os.path.join(base_data_root, cam_name_iter)
            dng_files_in_cam_path = [f for f in os.listdir(cam_data_path) if f.endswith('.dng')]
            if not dng_files_in_cam_path:
                print(f"Warning: No .dng files found for camera {cam_name_iter} in {cam_data_path}. Skipping.", file=sys.stderr)
                continue
            
            first_dng_path = os.path.join(cam_data_path, dng_files_in_cam_path[0])
            try:
                dng_raw_obj = rawpy.imread(first_dng_path)
                # Intel dataset .tiff files are already black level subtracted. So black_level is effectively 0 for processing.
                # However, dng_raw.black_level_per_channel[0] from DNG might be non-zero. For calibration_metadata, store DNG's value.
                # For actual processing of TIFFs, black level is assumed 0. Saturation is from DNG.
                dng_black_level_for_meta = dng_raw_obj.black_level_per_channel[0]
                dng_saturation_level_val = dng_raw_obj.white_level
                del dng_raw_obj
            except Exception as e:
                print(f"Error reading DNG {first_dng_path} for {cam_name_iter}: {e}. Skipping camera.", file=sys.stderr)
                continue
            
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

            all_cameras_calibration_data[cam_name_iter] = {
                'black_level': dng_black_level_for_meta, # DNG's black level for metadata
                'white_level': dng_saturation_level_val,
                'CalibrationIlluminant1': calib_illum1, 'CalibrationIlluminant2': calib_illum2,
                'ColorMatrix1': dng_cm1_arr.tolist(), 'ColorMatrix2': dng_cm2_arr.tolist(),
                'ForwardMatrix1': dng_fm1_arr.tolist(), 'ForwardMatrix2': dng_fm2_arr.tolist(),
                'AnalogBalance': dng_ab_arr.tolist(), 'IlluminantRGB': gen_illum_rgb_list,
            }
            
            current_cam_all_img_meta = {}
            tasks_for_intel_pool_current_cam = []

            for sub_dir_name in sub_directory_list:
                current_sub_dir_path = os.path.join(base_data_root, cam_name_iter, sub_dir_name)
                if not os.path.isdir(current_sub_dir_path):
                    print(f"Subdirectory {current_sub_dir_path} not found. Skipping.", file=sys.stderr)
                    continue

                tiff_files_in_subdir = sorted([os.path.join(current_sub_dir_path, f) for f in os.listdir(current_sub_dir_path) if f.endswith('.tiff')])

                for tiff_path in tiff_files_in_subdir:
                    wp_path = tiff_path.replace('.tiff', '.wp')
                    if not os.path.exists(wp_path):
                        print(f"Warning: .wp file not found for {tiff_path}. Skipping this image.", file=sys.stderr)
                        continue
                    with open(wp_path, 'r') as f_wp:
                        gt_illum_from_wp = np.array([float(x) for x in f_wp.read().strip().split()])
                    gt_illum_from_wp = gt_illum_from_wp / gt_illum_from_wp[1] # Normalize by G
                    
                    task_args_tuple = (
                        tiff_path, cam_name_iter, dng_saturation_level_val, gt_illum_from_wp,
                        gen_illum_rgb_list, gen_wbraw2xyz_list, gen_xyz2wbraw_list,
                        gen_xyz2raw_list, gen_raw2xyz_list,
                        target_size_config, current_cam_preprocess_dir, original_resized_output_base_dir,
                        SAVE_PREPROCESS_CONFIG, SAVE_ORIGINAL_CONFIG,
                        dng_cm1_arr.tolist(), dng_cm2_arr.tolist(), dng_fm1_arr.tolist(), dng_fm2_arr.tolist()
                    )
                    tasks_for_intel_pool_current_cam.append(task_args_tuple)
            
            if tasks_for_intel_pool_current_cam:
                with Pool(processes=cpu_count()) as pool:
                    img_results_iterator = pool.imap_unordered(process_single_raw_image_intel, tasks_for_intel_pool_current_cam)
                    for fname_result, single_img_meta_result in tqdm(img_results_iterator, total=len(tasks_for_intel_pool_current_cam), desc=f"Processing images for {cam_name_iter}", leave=False):
                        current_cam_all_img_meta[fname_result] = single_img_meta_result
            
            with open(os.path.join(current_cam_preprocess_dir, 'metadata.json'), 'w') as f_json:
                json.dump(current_cam_all_img_meta, f_json, indent=4)

    with open(calibration_metadata_path, 'w') as f_calib_json:
        json.dump(all_cameras_calibration_data, f_calib_json, indent=4)

    print('Done') 