import os, json, colour, scipy, sys
import numpy as np
import rawpy
import cv2
from exiftool import ExifToolHelper
from tqdm import tqdm
from mcc_utils import get_binary_mask
from multiprocessing import Pool, cpu_count

'''
Processed dataset will be saved in the following structure:
cc_wb/
    Canon1DsMkIII/
    Canon600D/
    NikonD5200/
    SamsungNX2000/
    SonyA57/
    calibration_metadata.json

calibration_metadata.json will contain calibration related metadata for each camera:
- CalibrationIlluminant1 : EXIF:CalibrationIlluminant1
- CalibrationIlluminant2 : EXIF:CalibrationIlluminant2
- ColorMatrix1 : EXIF:ColorMatrix1
- ColorMatrix2 : EXIF:ColorMatrix2
- ForwardMatrix1 : EXIF:ForwardMatrix1
- ForwardMatrix2 : EXIF:ForwardMatrix2
- AnalogBalance : EXIF:AnalogBalance
- CameraCalibration1 : EXIF:CameraCalibration1
- CameraCalibration2 : EXIF:CameraCalibration2
- IlluminantRGB : list of illuminant RGBs for each color temperature from 2500K to 7500K

for each camera sub-directory, it will contain:
- metadata.json : metadata for each raw image ex) gt_illum, closest_kelvin_temp, xyz2wbraw, wbraw2xyz, xyz2raw, raw2xyz
- raw images : raw images with black level subtracted
- wb images : white balanced images
- xyz images : xyz images
- mask images : MCC mask images
'''

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

def process_single_raw_image_nus(args_tuple):
    raw_path, cam_name, img_idx, current_gt_illum, current_cc_coords, \
    processing_black_level, processing_sat_level, \
    camera_illum_rgb_list, camera_wbraw2xyz_list, camera_xyz2wbraw_list, camera_xyz2raw_list, camera_raw2xyz_list, \
    img_target_size, cam_target_dir_preprocess, cam_resized_dir_original, \
    do_save_preprocess, do_save_original, \
    dng_cm1_list, dng_cm2_list, dng_fm1_list, dng_fm2_list = args_tuple

    fname = os.path.basename(raw_path).split('.')[0]
    
    raw_img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)[:,:,::-1].astype(np.float32) - processing_black_level
    max_level = processing_sat_level - processing_black_level
    raw_img_processed = np.clip(raw_img, 0, max_level)
    
    wb_img = raw_img_processed / (current_gt_illum / current_gt_illum[1])
    wb_img = np.clip(wb_img, 0, max_level)

    nearest_idx = get_nearest_illum(current_gt_illum, camera_illum_rgb_list)
    wbraw2xyz = np.array(camera_wbraw2xyz_list[nearest_idx])
    kelvin_temp = int(2500 + nearest_idx)
    
    single_img_meta = {
        'gt_illum': current_gt_illum.tolist(),
        'closest_kelvin_temp': kelvin_temp,
        'xyz2wbraw': camera_xyz2wbraw_list[nearest_idx],
        'wbraw2xyz': camera_wbraw2xyz_list[nearest_idx],
        'xyz2raw': camera_xyz2raw_list[nearest_idx],
        'raw2xyz': camera_raw2xyz_list[nearest_idx],
    }

    norm_wb_img = wb_img / max_level
    xyz_img = np.dot(norm_wb_img, wbraw2xyz.T) 
    xyz_img = np.clip(xyz_img, 0.,1.) * (2 ** 16 - 1)

    y1, y2, x1, x2 = current_cc_coords
    mcc_points_list = np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]]).astype(int).reshape((-1,1,2))
    mask = get_binary_mask(h=raw_img_processed.shape[0], w=raw_img_processed.shape[1], point_list=mcc_points_list)[:,:,None]

    if do_save_preprocess:
        raw_img_to_save = cv2.resize(raw_img_processed, img_target_size, interpolation=cv2.INTER_AREA)
        wb_img_to_save = cv2.resize(wb_img, img_target_size, interpolation=cv2.INTER_AREA)
        xyz_img_to_save = cv2.resize(xyz_img, img_target_size, interpolation=cv2.INTER_AREA)
        mask_to_save = cv2.resize(mask, img_target_size, interpolation=cv2.INTER_AREA)
        if mask_to_save.ndim == 2:
            mask_to_save = mask_to_save[:,:,np.newaxis]

        cv2.imwrite(os.path.join(cam_target_dir_preprocess, f'{fname}_raw.png'), raw_img_to_save[:,:,::-1].astype(np.uint16))
        cv2.imwrite(os.path.join(cam_target_dir_preprocess, f'{fname}_wb.png'), wb_img_to_save[:,:,::-1].astype(np.uint16))
        cv2.imwrite(os.path.join(cam_target_dir_preprocess, f'{fname}_xyz.png'), xyz_img_to_save[:,:,::-1].astype(np.uint16))
        cv2.imwrite(os.path.join(cam_target_dir_preprocess, f'{fname}_mask.png'), (mask_to_save*255).astype(np.uint8))

    if do_save_original:
        raw_img_masked = raw_img_processed * mask 
        raw_img_resized = cv2.resize(raw_img_masked, img_target_size, interpolation=cv2.INTER_AREA)
        if np.max(raw_img_resized) > 0:
             raw_img_resized_norm = raw_img_resized / np.max(raw_img_resized) * 65535.0
        else:
             raw_img_resized_norm = raw_img_resized # Already zero
        raw_img_resized_norm = np.clip(raw_img_resized_norm, 0, 65535.0)
        
        resized_meta = {
            'illuminant_color_raw': (current_gt_illum/current_gt_illum[1]).tolist(),
            'cm1': dng_cm1_list,
            'cm2': dng_cm2_list,
            'fm1': dng_fm1_list,
            'fm2': dng_fm2_list
        }
        target_name = f'{fname}_sensorname_{cam_name}.png'
        cv2.imwrite(os.path.join(cam_resized_dir_original, target_name), raw_img_resized_norm[:,:,::-1].astype(np.uint16))
    
        with open(os.path.join(cam_resized_dir_original, target_name.replace('.png', '_metadata.json')), 'w') as f:
            json.dump(resized_meta, f, indent=4)
            
    return fname, single_img_meta

if __name__ == '__main__':
    target_size = (384, 256)
    SAVE_PREPROCESS = True
    SAVE_ORIGINAL = True
    data_root = '../../dataset/NUS-8/'
    preprocessed_base_dir = '../../dataset/CCMNet/preprocessed_for_augmentation/'
    nus_resized_base_dir = '../../dataset/CCMNet/original_resized/NUS-8/' # Changed name for clarity
    
    os.makedirs(preprocessed_base_dir, exist_ok=True)
    os.makedirs(nus_resized_base_dir, exist_ok=True)
    
    cams = [
        'Canon1DsMkIII', 'Canon600D', 'NikonD5200', 'SamsungNX2000',
        'SonyA57', 'FujifilmXM1', 'OlympusEPL6', 'PanasonicGX1',
    ]

    calibration_metadata_file = os.path.join(preprocessed_base_dir, 'calibration_metadata.json')
    if os.path.exists(calibration_metadata_file):
        all_cam_calibration_data = json.load(open(calibration_metadata_file))
    else:
        all_cam_calibration_data = {}

    with ExifToolHelper() as et:
        cam_pbar = tqdm(cams, leave=True, desc="Processing Cameras")
        for cam in cam_pbar:
            target_cam_dir_for_preprocess_current_cam = os.path.join(preprocessed_base_dir, cam)
            os.makedirs(target_cam_dir_for_preprocess_current_cam, exist_ok=True)

            cam_dng_dir = os.path.join(data_root, cam, 'dng')
            dng_list = [os.path.join(cam_dng_dir, f) for f in os.listdir(cam_dng_dir) if f.endswith('.dng')]
            if not dng_list:
                print(f"Warning: No DNG files found for {cam} in {cam_dng_dir}. Skipping camera.", file=sys.stderr)
                continue
            first_dng_file = dng_list[0]

            try:
                dng_raw_obj_temp = rawpy.imread(first_dng_file)
                dng_black_level_from_rawpy = dng_raw_obj_temp.black_level_per_channel[0]
                dng_white_level_from_rawpy = dng_raw_obj_temp.white_level
                del dng_raw_obj_temp
            except Exception as e:
                print(f"Error reading DNG file {first_dng_file} for {cam}: {e}. Skipping camera.", file=sys.stderr)
                continue
            
            meta_dict_exif = et.get_metadata(first_dng_file)[0]
            dng_illum1 = meta_dict_exif["EXIF:CalibrationIlluminant1"]
            dng_illum2 = meta_dict_exif["EXIF:CalibrationIlluminant2"]
            dng_cm1 = exif_to_nparray(meta_dict_exif["EXIF:ColorMatrix1"])
            dng_cm2 = exif_to_nparray(meta_dict_exif["EXIF:ColorMatrix2"])
            dng_fm1 = exif_to_nparray(meta_dict_exif["EXIF:ForwardMatrix1"])
            dng_fm2 = exif_to_nparray(meta_dict_exif["EXIF:ForwardMatrix2"])
            dng_ab = exif_to_nparray(meta_dict_exif["EXIF:AnalogBalance"])

            internal_calibration_dict = {"ab": dng_ab, "cm1": dng_cm1, "cm2": dng_cm2, "fm1": dng_fm1, "fm2": dng_fm2}

            cam_generated_illum_rgb_list, cam_gen_xyz2wbraw_list, cam_gen_wbraw2xyz_list, cam_gen_xyz2raw_list, cam_gen_raw2xyz_list = [], [], [], [], []
            for temp_val in range(2500, 7501):
                illumrgb, wbraw2xyz_m, xyz2wbraw_m, raw2xyz_m, xyz2raw_m = get_camrgb_for_colortemp(internal_calibration_dict, temp_val, return_calibration_matrices=True)
                illumrgb = illumrgb / illumrgb[1]
                cam_generated_illum_rgb_list.append(illumrgb.tolist())
                cam_gen_xyz2wbraw_list.append(xyz2wbraw_m.tolist())
                cam_gen_wbraw2xyz_list.append(wbraw2xyz_m.tolist())
                cam_gen_xyz2raw_list.append(xyz2raw_m.tolist())
                cam_gen_raw2xyz_list.append(raw2xyz_m.tolist())

            # Load GT data from .mat file for this camera
            cam_gt_mat_path = os.path.join(data_root, cam, f'{cam}_gt.mat')
            try:
                cam_gt_data_mat = scipy.io.loadmat(cam_gt_mat_path)
            except FileNotFoundError:
                print(f"Warning: GT .mat file not found for {cam} at {cam_gt_mat_path}. Skipping camera.", file=sys.stderr)
                continue
            
            gt_illuminants_for_cam = cam_gt_data_mat['groundtruth_illuminants']
            gt_cc_coords_for_cam = cam_gt_data_mat['CC_coords']
            processing_black_level_from_mat = cam_gt_data_mat['darkness_level'][0][0]
            processing_sat_level_from_mat = cam_gt_data_mat['saturation_level'][0][0]

            all_cam_calibration_data[cam] = {
                'black_level': dng_black_level_from_rawpy, # Using DNG rawpy value for consistency in calibration_metadata
                'white_level': dng_white_level_from_rawpy,
                'CalibrationIlluminant1': dng_illum1,
                'CalibrationIlluminant2': dng_illum2,
                'ColorMatrix1': dng_cm1.tolist(),
                'ColorMatrix2': dng_cm2.tolist(),
                'ForwardMatrix1': dng_fm1.tolist(),
                'ForwardMatrix2': dng_fm2.tolist(),
                'AnalogBalance': dng_ab.tolist(),
                'IlluminantRGB': cam_generated_illum_rgb_list,
            }
            
            cam_png_dir = os.path.join(data_root, cam, 'PNG') # Note: PNG in uppercase
            cam_raw_img_paths = sorted([os.path.join(cam_png_dir, f) for f in os.listdir(cam_png_dir) if f.endswith('.PNG')])
            
            tasks_for_nus_pool = []
            if len(cam_raw_img_paths) != len(gt_illuminants_for_cam):
                 print(f"Warning: Mismatch image count ({len(cam_raw_img_paths)}) and GT illuminant count ({len(gt_illuminants_for_cam)}) for {cam}. Skipping.", file=sys.stderr)
                 continue

            for i, current_raw_path in enumerate(cam_raw_img_paths):
                fname_base = os.path.basename(current_raw_path).split('.')[0]
                img_idx_from_fname = int(fname_base.split('_')[1]) # NUS image names are like IMG_0001.PNG
                
                # Check if img_idx_from_fname is valid for gt_illuminants_for_cam and gt_cc_coords_for_cam
                if not (0 <= img_idx_from_fname - 1 < len(gt_illuminants_for_cam) and 0 <= img_idx_from_fname - 1 < len(gt_cc_coords_for_cam)):
                    print(f"Warning: Image index {img_idx_from_fname} from {fname_base} is out of bounds for GT data for camera {cam}. Skipping this image.", file=sys.stderr)
                    continue

                current_img_gt_illum = gt_illuminants_for_cam[img_idx_from_fname-1]
                current_img_cc_coords = gt_cc_coords_for_cam[img_idx_from_fname-1]

                task_args = (
                    current_raw_path, cam, img_idx_from_fname,
                    current_img_gt_illum, current_img_cc_coords,
                    processing_black_level_from_mat, processing_sat_level_from_mat,
                    cam_generated_illum_rgb_list, 
                    cam_gen_wbraw2xyz_list, cam_gen_xyz2wbraw_list, 
                    cam_gen_xyz2raw_list, cam_gen_raw2xyz_list,
                    target_size, target_cam_dir_for_preprocess_current_cam, nus_resized_base_dir,
                    SAVE_PREPROCESS, SAVE_ORIGINAL,
                    dng_cm1.tolist(), dng_cm2.tolist(), dng_fm1.tolist(), dng_fm2.tolist()
                )
                tasks_for_nus_pool.append(task_args)

            per_cam_img_meta_results = {}
            if tasks_for_nus_pool:
                with Pool(processes=cpu_count()) as pool:
                    results_iterator = pool.imap_unordered(process_single_raw_image_nus, tasks_for_nus_pool)
                    for fname_res, single_img_meta_res in tqdm(results_iterator, total=len(tasks_for_nus_pool), desc=f"Processing images for {cam}", leave=False):
                        per_cam_img_meta_results[fname_res] = single_img_meta_res
            
            with open(os.path.join(target_cam_dir_for_preprocess_current_cam, 'metadata.json'), 'w') as f:
                json.dump(per_cam_img_meta_results, f, indent=4)

    with open(calibration_metadata_file, 'w') as f:
        json.dump(all_cam_calibration_data, f, indent=4)

    print('Done')