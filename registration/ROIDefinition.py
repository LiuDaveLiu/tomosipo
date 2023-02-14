import numpy as np


def ROImanual(xmin, xmax, ymin, ymax, n, **kwargs):
    ROI_1 = [0, n, 0, n]
    ROI_3 = restrictROI([xmin, xmax, ymin, ymax], **kwargs)
    ROI_2 = restrictROI(intermediateROI(ROI_3, n), **kwargs)
    return [ROI_1, ROI_2, ROI_3]


def ROIForwardProject(x, proj_spacing, proj_dimensions, n, pmatrix, r=100, **kwargs):
    ROI_1 = [0, n, 0, n]

    xproj = pmatrix @ np.append(x,1)
    xproj = xproj[:2]/xproj[2]
    xproj = xproj/proj_spacing[:2] + (proj_dimensions[:2]-1)/2
    xproj = xproj.astype(int)
    ROI_3 = restrictROI([xproj[0]-r/2,xproj[0]+r/2,xproj[1]-r/2,xproj[1]+r/2], **kwargs)

    ROI_2 = restrictROI(intermediateROI(ROI_3, n), **kwargs)
    return [ROI_1, ROI_2, ROI_3]


def ROIUniformGrid(n, nrows=5, ncols=5, roi_width=None, pad_top=0, pad_bottom=0, pad_left=0, pad_right=0, **kwargs):
    if not roi_width:
        roi_width = 0.2*n  # 20% of projection dimensions

    ur = pad_top+roi_width/2
    lr = n-pad_bottom-roi_width/2
    lc = pad_left+roi_width/2
    rc = n-pad_right-roi_width/2
    center_pts_r = np.round(np.linspace(ur, lr, nrows))
    center_pts_c = np.round(np.linspace(lc, rc, ncols))

    grid = np.meshgrid(center_pts_r, center_pts_c)

    ROI_1 = [0, n, 0, n]
    ROIs = []
    for i in range(grid[0].shape[0]):
        for j in range(grid[0].shape[1]):
            float_roi = [grid[0][i][j]-roi_width/2, grid[0][i][j]+roi_width/2, grid[1][i][j]-roi_width/2, grid[1][i][j]+roi_width/2]
            ROI_3 = restrictROI(float_roi, n, **kwargs)
            ROI_2 = restrictROI(intermediateROI(ROI_3, n), **kwargs)
            ROIs.append([ROI_1, ROI_2, ROI_3])

    return ROIs


def restrictROI(ROI, n, buffer=0):
    "Restricts ROI to fit within bounds of (n x n) image"
    if ROI[0] < 0:
        ROI[0] += -ROI[0]+buffer
        ROI[1] += -ROI[0]+buffer
    if ROI[1] > n:
        ROI[0] -= ROI[1]-n+buffer
        ROI[1] -= ROI[1]-n+buffer
    if ROI[2] < 0:
        ROI[2] += -ROI[2]+buffer
        ROI[3] += -ROI[2]+buffer
    if ROI[3] > n:
        ROI[2] -= ROI[3]-n+buffer
        ROI[3] -= ROI[3]-n+buffer
    return [int(x) for x in ROI]


def intermediateROI(local_ROI, n):
    r = local_ROI[1]-local_ROI[0]
    center_pt = np.array([local_ROI[1]-r/2, local_ROI[3]-r/2])
    size = n-(n-r)/2

    return [center_pt[0]-size/2, center_pt[0]+size/2, center_pt[1]-size/2, center_pt[1]+size/2]
