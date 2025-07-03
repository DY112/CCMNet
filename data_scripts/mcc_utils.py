import os,json,cv2,rawpy,math
import numpy as np
from tqdm import tqdm

"""
cellchart contains 24 color patch coordinates (x, y)

0   1   2   3   4   5
6   7   8   9   10  11
12  13  14  15  16  17
18  19  20  21  22  23

Each color patch coordinates start from upper left, clockwise order
"""
CELLCHART = np.float32([[
    [0.25, 0.25],
    [2.75, 0.25],
    [2.75, 2.75],
    [0.25, 2.75],
    [3.00, 0.25],
    [5.50, 0.25],
    [5.50, 2.75],
    [3.00, 2.75],
    [5.75, 0.25],
    [8.25, 0.25],
    [8.25, 2.75],
    [5.75, 2.75],
    [8.50, 0.25],
    [11.00, 0.25],
    [11.00, 2.75],
    [8.50, 2.75],
    [11.25, 0.25],
    [13.75, 0.25],
    [13.75, 2.75],
    [11.25, 2.75],
    [14.00, 0.25],
    [16.50, 0.25],
    [16.50, 2.75],
    [14.00, 2.75],
    [0.25, 3.00],
    [2.75, 3.00],
    [2.75, 5.50],
    [0.25, 5.50],
    [3.00, 3.00],
    [5.50, 3.00],
    [5.50, 5.50],
    [3.00, 5.50],
    [5.75, 3.00],
    [8.25, 3.00],
    [8.25, 5.50],
    [5.75, 5.50],
    [8.50, 3.00],
    [11.00, 3.00],
    [11.00, 5.50],
    [8.50, 5.50],
    [11.25, 3.00],
    [13.75, 3.00],
    [13.75, 5.50],
    [11.25, 5.50],
    [14.00, 3.00],
    [16.50, 3.00],
    [16.50, 5.50],
    [14.00, 5.50],
    [0.25, 5.75],
    [2.75, 5.75],
    [2.75, 8.25],
    [0.25, 8.25],
    [3.00, 5.75],
    [5.50, 5.75],
    [5.50, 8.25],
    [3.00, 8.25],
    [5.75, 5.75],
    [8.25, 5.75],
    [8.25, 8.25],
    [5.75, 8.25],
    [8.50, 5.75],
    [11.00, 5.75],
    [11.00, 8.25],
    [8.50, 8.25],
    [11.25, 5.75],
    [13.75, 5.75],
    [13.75, 8.25],
    [11.25, 8.25],
    [14.00, 5.75],
    [16.50, 5.75],
    [16.50, 8.25],
    [14.00, 8.25],
    [0.25, 8.50],
    [2.75, 8.50],
    [2.75, 11.00],
    [0.25, 11.00],
    [3.00, 8.50],
    [5.50, 8.50],
    [5.50, 11.00],
    [3.00, 11.00],
    [5.75, 8.50],
    [8.25, 8.50],
    [8.25, 11.00],
    [5.75, 11.00],
    [8.50, 8.50],
    [11.00, 8.50],
    [11.00, 11.00],
    [8.50, 11.00],
    [11.25, 8.50],
    [13.75, 8.50],
    [13.75, 11.00],
    [11.25, 11.00],
    [14.00, 8.50],
    [16.50, 8.50],
    [16.50, 11.00],
    [14.00, 11.00]]])
MCCBOX = np.float32([[0.00, 0.00], [16.75, 0.00], [16.75, 11.25], [0.00, 11.25]])

def get_chart_chromas(img, mcc_coord, white_level=1):
    """
    img         : RGB image
    mcc_coord   : MCC chart coordinates
    white_level : white level of the image

    returns     : numpy array with shape (24,3)
                  (patch, RGB channel sum)
    """
    CHART_SCALE = 20
    cellchart = CELLCHART * CHART_SCALE
    cellchart = np.reshape(cellchart, (24,4,2))
    chart_chromas = np.zeros((24,3))
    is_saturated_mat = np.zeros(24, dtype=bool)

    src = mcc_coord
    dst = MCCBOX * CHART_SCALE
    chart_width, chart_height = int((MCCBOX[2]*CHART_SCALE)[0]), int((MCCBOX[2]*CHART_SCALE)[1])

    # Get perspective transform matrix, apply transform to cellchart
    M = cv2.getPerspectiveTransform(src, dst)
    transformed_chart = cv2.warpPerspective(img, M, (chart_width, chart_height))

    # visualize transformed chart
    cv2.imwrite("transformed_chart.png", transformed_chart[...,::-1]*255)

    # Reduce the box size by 50%
    for cell_idx in range(24):
        centerPoint = np.sum(cellchart[cell_idx], axis=0) / 4
        for j in range(4):
            cellchart[cell_idx][j] = (cellchart[cell_idx][j] + centerPoint) / 2

    # generate mask for patches & record avg value, check if saturated
    for cell_idx in range(24):
        mask = np.zeros_like(transformed_chart)
        cell = np.array([[cellchart[cell_idx,0]], [cellchart[cell_idx,1]], [cellchart[cell_idx,2]], [cellchart[cell_idx,3]]]).astype(int)
        cv2.drawContours(mask, [cell], 0, (1,1,1), -1) # fill inside the contour
        masked_img = transformed_chart*mask
        masked_img[masked_img == 0] = np.nan

        # channelwise average
        patch_avg = np.nanmean(masked_img, axis=(0,1))
        chart_chromas[cell_idx, :] = patch_avg
        # saturation check
        if True in (masked_img >= white_level):
            is_saturated_mat[cell_idx] = True

    return chart_chromas, is_saturated_mat

def get_binary_mask_of_mcc_list(h,w,mcc_list):
    mask = np.ones((h,w), dtype=np.uint8)
    for mcc in mcc_list:
        pts = np.array(mcc).astype(int)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(mask, [pts], (0))

    return mask

def get_binary_mask(h,w,point_list):
    mask = np.ones((h,w), dtype=np.uint8)
    pts = np.array(point_list, np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(mask, [pts], (0))

    return mask