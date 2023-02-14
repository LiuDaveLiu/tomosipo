from typing import Tuple, Union, List
import numpy as np
import xmltodict
from scipy.spatial.transform import Rotation
import pydicom

# detector domain adaptions
to_mm = np.array([[0.304, 0, 0],
                  [0, 0.304, 0],
                  [0, 0, 1]])  # not tested
to_origin = np.array([[1, 0, -488],
                      [0, 1, -488],
                      [0, 0, 1]])  # not tested
FLIPU = np.array([[0, -1, 975],
                  [1, 0, 0],
                  [0, 0, 1]])

flipu = lambda width: np.array([[0, -1, width - 1],
                                [1, 0, 0],
                                [0, 0, 1]])

# this is used with the IStar cudatools
IStar_flip = np.dot(Rotation.from_rotvec(np.array([0, 0, 1]) * np.pi).as_matrix(),  # rotate detector
                    Rotation.from_rotvec(np.array([0, 1, 0]) * np.pi).as_matrix())  # flip ray direction

# volume domain adaptions
v_mm = 0.313  # voxel side length in mm
xyz_from_iso = np.array([[v_mm, 0, 0, 0],
                         [0, v_mm, 0, 0],
                         [0, 0, v_mm, 0],
                         [0, 0, 0, 1]])
c = 255.5  # (512 / 2) - 0.5
iso_from_ijk = np.array([[1, 0, 0, -c],
                         [0, 1, 0, -c],
                         [0, 0, 1, -c],
                         [0, 0, 0, 1]])


class ProjMatrix:
    def __init__(self, R, K, t):
        self.R = np.array(R, dtype=np.float32)
        self.t = np.array(t, dtype=np.float32)
        self.K = np.array(K, dtype=np.float32)
        self.P = np.matmul(self.K, np.concatenate((self.R, np.expand_dims(self.t, 1)), axis=1))
        self.rtk_inv = np.matmul(np.transpose(self.R), np.linalg.inv(self.K))

    def set_rotation(self, R_new):
        self.R = R_new
        self.rtk_inv = np.matmul(np.transpose(self.R), np.linalg.inv(self.K))
        self.P = np.matmul(self.K, np.concatenate((self.R, np.expand_dims(self.t, 1)), axis=1))

    def get_camera_ceter(self):
        return np.matmul(np.transpose(self.R), self.t)

    def get_principle_axis(self):
        axis = self.R[2, :] / self.K[2, 2]
        return axis


def RQfactorize(A):
    A_flip = np.flip(A, axis=0).T  # shape 4x3
    Q_flip, R_flip = np.linalg.qr(A_flip)  # shapes 4x4, 4x3
    for i in range(3):
        if R_flip[i, i] < 0:
            R_flip[i, :] *= -1
            Q_flip[:, i] *= -1

    assert R_flip[0, 0] > 0
    assert R_flip[1, 1] > 0
    assert R_flip[2, 2] > 0

    R = np.flip(np.flip(R_flip.T, axis=1), axis=0)
    Q = np.flip(Q_flip.T, axis=0)
    return R, Q


def decompose(P, scaleIntrinsic=False):
    # 1. Split P in 3x3 (first three columns) matrix and 3x vector (forthcolumn)
    P3x3, forthColumn = P[:, :3].copy(), P[:, 3].copy()

    # 2. RQ - factorisation of 3x3 P
    K, R = RQfactorize(P3x3)

    # 3. If determinant of Rotation Matrix is negative --> make positive and adjust sourcepoint
    if np.linalg.det(R) < 0:
        print("negative determinant")
        R *= -1
        forthColumn *= -1

    # 4. calculate translation by back substitution
    t = np.zeros(3)
    t[2] = forthColumn[2] / K[2, 2]
    t[1] = (forthColumn[1] - K[1, 2] * t[2]) / K[1, 1]
    t[0] = (forthColumn[0] - K[0, 1] * t[1] - K[0, 2] * t[2]) / K[0, 0]

    # 5. Optionally scale the camera intrinsic
    if scaleIntrinsic:
        K *= (1 / K[2, 2])

    return K, R, t


def canonical_form_istar(spin_matrices):
    """
    Transforms projection matrix in uv_from_ijk format to Source point and forward matrices RtK
    :param spin_matrices: in shape (n, 3, 4)
    :return: matrices and camera center for use with projector. arrays in shape (n, 3, 3) (n, 3)
    """
    n = spin_matrices.shape[0]
    src_points = np.zeros((n, 3))
    forward_matrices = np.zeros((n, 3, 3))

    for i in range(n):
        K, R, t = decompose(spin_matrices[i], scaleIntrinsic=True)
        rtk_inv = R.T @ np.linalg.inv(K)
        S = (-1) * np.matmul(np.transpose(R), t)
        forward_matrices[i] = rtk_inv
        src_points[i] = S

    return forward_matrices, src_points


def spin_matrices_from_xml(path_to_projection_matrices: str) -> Tuple[np.ndarray, np.ndarray]:
    """[Load image , i0s and projection matrices from multi frame dicom.]

    Args:
        path_to_projection_matrices (str): [path to a .xml file containing the projection matrices]

    Returns:
        [array like]: array of projection matrices in shape (n, 3, 4)
        [array like]: array of I0s in shape (n,)
        [array like]: array of angles in shape (n,)
    """
    assert str(path_to_projection_matrices).endswith('.xml')
    with open(path_to_projection_matrices) as fd:

        contents = xmltodict.parse(fd.read())
        matrices = contents['hdr']['ElementList']['PROJECTION_MATRICES']
        i0s_dict = contents['hdr']['ElementList']['I0_VALUES']
        # proj_angles_dict = contents['hdr']['ElementList']['TRIGGER_ANGLES']

        # backprojection matrices project 2d-projections into 3d-space (homogenous coordinates allow for translation)
        proj_mat = np.zeros((len(matrices.keys()), 3, 4))
        i0s = np.zeros(len(i0s_dict.keys()))
        # proj_angles = np.zeros(len(proj_angles_dict.keys()))

        # parsing the ordered dict to numpy matrices
        for i, key in enumerate(matrices.keys()):
            value_string = matrices[key]

            # expecting 12 entries to construct 3x4 matrix in row-major order (C-contiguous)
            proj_mat[i] = np.array(value_string.split(" "), order='C').reshape((3, 4))

        # parsing i0s
        for i, key in enumerate(i0s_dict.keys()):
            value_string = i0s_dict[key]
            i0s[i] = float(value_string)

    return proj_mat, i0s


def matrices_from_dcm(path_to_projections):
    """loads the projection matrices and I0s from a reference Dicom"""
    orig = dcmread(path_to_projections)
    projmat = np.frombuffer(orig[0x0017, 0x10f6].value).reshape(-1, 3, 4).copy().astype(np.float32)
    I0s = np.array(orig[0x001710fb].value, dtype=np.float32)[::4]
    return projmat, I0s


def generate_projection_matrix_standard_plane(angles=np.array([0, 90]), siemens_format=False, degrees=True):
    """
    Get projection matrices at acquisition angle. Rotation is around z-axis, standard position is on x-axis.

    :param degrees: if true, angles are interpreted as degrees. otherwise its radians
    :param angles: rotation angle around rotation axis z with 0 being projecting along x-axis. shape = (p,)
    :return: P mapping from Volume index to Detector index. Centers are centers of volume and detector.
    """
    # construct standard projection parameters
    v_mm = 0.313  # voxel side length
    px, py = 488, 488  # center of detector with resolution of 976x976
    sid_mm = 622
    sdd_mm = 1164
    K = np.array([[sdd_mm / v_mm, 0, px],
                  [0, sdd_mm / v_mm, py],
                  [0, 0, 1]])  # focal length is twice the iso center distance, pierce point is center of detector
    R = np.array([[0, 0, 1],
                  [0, 1, 0],
                  [1, 0, 0]])  # rotate view direction (z -> x-axis) such that it makes sense with a source on x-axis
    c = np.array([-sid_mm / v_mm, 0, 0])  # camera sits on z-axis at z = -622mm

    # add world rotation
    matrices = np.zeros((angles.shape[0], 3, 4))
    FLIPUinv = np.linalg.inv(FLIPU)
    for i, angle in enumerate(angles):
        R_world = Rotation.from_euler('z', angle, degrees=degrees).as_matrix()  # shape (3, 3)
        R_c = R_world @ R
        It = np.eye(3, 4)
        It[:, 3] = -c
        t = -R_c @ c
        R_ext = R_c.T
        Rt = np.column_stack([R_ext, t])  # shape (3, 4)
        if siemens_format:
            # maps world coordinates (isocentered, mm) to detector indices (0-976)
            matrices[i] = FLIPUinv @ K @ Rt @ np.linalg.inv(xyz_from_iso)
        else:
            # maps volume indices (0-512) to detector indices (0-976)
            matrices[i] = K @ Rt @ iso_from_ijk  # extrinsic definition
    return matrices


def istar_from_cios(matrices):
    """
    Adapt projection matrices to follow IStar Conventions.
    :param matrices: world [0-center, mm] to detector [488-centered, idx] (k, 3, 4)
    :return: P_ijk: world [0-center, mm] to detector [0-center, mm] (k, 3, 4)
    """
    P_ijk = np.array([IStar_flip @ to_mm @ to_origin @ _P for _P in matrices])
    return P_ijk


def uv_from_ijk_format(matrices):
    """
    convert standard matrices to pixel from voxel notation
    :param matrices: pixel-idx from world convention (straight from cios spin) in shape (n, 3, 4)
    :return: pfv-convention in shape (n, 3, 4)
    """
    return np.asarray([_P @ xyz_from_iso @ iso_from_ijk for _P in matrices])


def matrix_from_dicom_attributes(dicom_path):
    '''
    Constructs projection matrix to map iso-centered volume coordinates in mm to detector pixel indices in range [0-976)
    :param dicom_path: dicom file with needed attributes in header (like angular and orbital angle)
    :return: projection matrix P in shape (3, 4)
    '''
    dicom = pydicom.dcmread(dicom_path)
    assert 'Cios_Spin' in dicom[0x0021, 0x1017].value, 'this function is only tested for the Siemens Cios Spin'

    def to4x4(a):
        assert a.shape == (3, 3)
        ret = np.eye(4)
        ret[:3, :3] = a
        return ret

    def chain_manipulations(code):
        assert code == '<FlipH=NO><FlipV=NO><Rotate=0Âº>', 'didnt implement flips yet. look at bjoerns reference code'
        return np.eye(4)

    # read values from dicom header
    orb = dicom.PositionerPrimaryAngle
    ang = dicom.PositionerSecondaryAngle
    sdd = dicom.DistanceSourceToDetector
    sid = 622  # hardcoded for Cios Spin
    img_shape = np.array([dicom.Rows, dicom.Columns])
    pixel_size = dicom.ImagerPixelSpacing
    processingString = dicom.DerivationDescription
    image_manipulations = chain_manipulations(processingString)

    # compose detector domain transform intrinsic
    trans2CenterMx = np.eye(4)
    trans2CenterMx[:2, 3] = - img_shape / 2 - .5
    scale2mmMx = np.diag([*pixel_size, 1, 1])
    trans2ScannerOrientationMxy = to4x4(Rotation.from_euler('XZ', [180, 90], degrees=True).as_matrix())
    trans3DMx = trans2ScannerOrientationMxy @ scale2mmMx @ image_manipulations @ trans2CenterMx
    intrinsics = np.delete(np.delete(trans3DMx, 2, 0), 2, 1)  # removes third column and row from 4x4 matrix

    # compose volume transform extrinsic
    source_from_center = np.eye(4)
    source_from_center[2, 3] = sid
    source_from_iso = source_from_center @ to4x4(Rotation.from_euler('XY', [-orb, -ang], degrees=True).as_matrix())

    # pure projection (and scale)
    project_scaled = np.zeros((3, 4))
    project_scaled[:3, :3] = np.diag([sdd, sdd, 1])

    # compose entire projection matrix
    _P = np.linalg.inv(intrinsics) @ project_scaled @ source_from_iso

    # calculate source point
    _S = np.linalg.inv(source_from_iso) @ np.asarray([0, 0, 0, 1])

    print(f'Extrinsics Orb: {orb}, Ang: {ang}')
    return _P / _P[2, 3]


def pixel_from_voxel(_mats: Union[List[np.ndarray], np.ndarray], voxel_size_mm: float = 0.313,
                     volume_shape_px: float = 512.):
    """
    This method transforms matrices straight from the Dicom header to be used with image data. Note, that both detector
    and volume coordinates are flipped. This is because numpy uses fast-last notation, and as the data is stored rows-
    first, the last coordinate is the x-axis. Because, per convention, the projection matrices assume xyz order, they
    need to be converted to zyx order to be compatible with voxel and pixel indices.
    :param _mats: matrices in shape (n, 3, 4)
    :param voxel_size_mm: voxel size in mm.
    :param volume_shape_px: volume side lenght in voxels
    :return: projection matrices to map a homogeneous voxel coordinate ijk to a pixel uv via: uv1 = P @ ijk1
    """
    if type(_mats) == np.ndarray:
        assert _mats.ndim == 3 and _mats.shape[1:] == (3, 4)
    v = voxel_size_mm
    uv_from_vu = np.array([[0, 1, 0],
                           [1, 0, 0],
                           [0, 0, 1]])
    FLIPU = np.array([[-1, 0, 975],
                      [0, 1, 0],
                      [0, 0, 1]])
    xyz_from_zyx = np.array([[0, 0, 1, 0],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 0, 0, 1]])
    xzy_from_iso = np.array([[v, 0, 0, 0],
                             [0, v, 0, 0],
                             [0, 0, v, 0],
                             [0, 0, 0, 1]])
    c = (volume_shape_px / 2) - 0.5
    iso_from_ijk = np.array([[1, 0, 0, -c],
                             [0, 1, 0, -c],
                             [0, 0, 1, -c],
                             [0, 0, 0, 1]])

    # calculate offset in voxels
    _mats = np.asarray([uv_from_vu @ FLIPU @ P @ xzy_from_iso @ iso_from_ijk @ xyz_from_zyx for P in _mats])
    return _mats


if __name__ == '__main__':
    from tifffile import imsave

    '''
    To test if your coordinate setup works, use a binarized volume of a calibration phantom and project the dots
    using the script below. Overlap with acquired projection images of the same scan to check.

    vol (l.337): binarized volume of calibration phantom
    projections (l.338): projection images (load e.g. using pydicom.dcmread(<path>).pixel_array)
    matrices (l.348): if projections are dicom, supply the same path here.
    '''

    vol = None  # Read in a volume here
    projections = None  # Read in projection images here

    # make sure vol is binary
    assert np.all(np.logical_xor((vol == 1), (vol == 0))), 'volume must be binary'

    # extract points where mask is one
    S_ijk = np.where(vol == 1)
    S_ijk = np.array([*S_ijk, np.ones(S_ijk[0].shape[0])]).T  # make homogenous (3, n) -> (4, n)
    print(f"samples: {S_ijk.shape}")

    matrices = matrices_from_dcm('path_to_dcm_projections')[0]  # use this if you have projection images as dicom
    # P = spin_matrices_from_xml('path_to_matrices.xml')  # if the matrices are available as .xml

    dec_img = np.zeros((len(matrices), 976, 976))
    for i, P in enumerate(matrices):
        # adapt projection matrix to map voxel indices to pixel indices
        P = pixel_from_voxel(P, voxel_size_mm=0.313, volume_shape_px=512)

        # project coordinates onto detector
        samples_dec_homog = P @ S_ijk

        samples_dec = samples_dec_homog[:2] / samples_dec_homog[-1]

        samples_dec_boundary_checked = samples_dec[:, np.all((samples_dec < 975.5) * (samples_dec > 0), axis=0)]
        outside_ratio = (samples_dec.shape[1] - samples_dec_boundary_checked.shape[1]) / samples_dec.shape[1]
        positions = np.round(samples_dec_boundary_checked).astype(int)  # (2, n)

        # create detector image by accumulating samples on detector
        pos_u, counts = np.unique(positions, axis=1, return_counts=True)
        dec_img[i, pos_u[1], pos_u[0]] = counts
        print(f"View {i}, lost samples: {np.round(outside_ratio * 100, decimals=1)}%")

    imsave('test_projection.tif', dec_img)
