import numpy as np
import transforms3d as t3d
import pathlib

def generate_projection_matrix_Siemens(filename, panel_spacing, panel_dimensions, **kwargs):
    ''' Siemens Geometry from Cios. Load from npz file '''
    transform_wcs = kwargs.pop('transform_wcs', affine(rotation=(0,0,0)))
    if kwargs:
        raise TypeError(f"{kwargs.keys()} are invalid keyword arguments")

    filename = pathlib.Path(filename)
    if filename.suffix == '.npz':
        with np.load(str(filename)) as f:
            pmatrices, encoder_angles = f['ProjectionMatrices'], f['RotorAngles']
    else:
        raise NotImplementedError('Unknown file extension.')

    pmatrices = np.array([np.linalg.multi_dot([
                          t3d.axangles.axangle2mat([0, 0, 1], np.pi),  # rotate detector
                          t3d.axangles.axangle2mat([0, 1, 0], np.pi),  # flip ray direction
                          t3d.affines.compose((0,0), np.eye(2,2), panel_spacing[:2]),  # px to mm
                          t3d.affines.compose(np.array(panel_dimensions[:2]) / -2, np.eye(2,2), (1,1)),  # corner to center
                          x,
                          transform_wcs]) for x in pmatrices])
    return pmatrices, encoder_angles
