import tomosipo as ts
import numpy as np
from medpy.io import load
import matplotlib.pyplot as plt
import torch
import time

base_path='C:/Users/lliu10/OneDrive - Inside MD Anderson/siemenproject/data/'
mage_data, image_header = load(base_path+'17_1.mha')

data=np.load(base_path+'SiemensCiosSpin_Siemens.npz')

vg = ts.volume(shape=np.shape(mage_data), size=(1, 1, 1))

angles = np.linspace(0, np.pi, 10, endpoint=False)
R = ts.rotate(pos=0, axis=(1, 0, 0), angles=angles)
R1 = ts.rotate(pos=0, axis=(0, 1, 0), angles=angles)
R2 = ts.rotate(pos=0, axis=(0, 0, 1), angles=angles)
T = ts.translate(((0, 0, 1),(0, 0, 1),(0,1,0),(1,0,1),(0, 0, 1),(0,1,0),(1,0,1),(0, 0, 1),(0,1,0),(1,0,1)))
vg_rot=R*R1*R2*T*vg.to_vec()
pg = ts.cone(angles=1, shape=(512, 512), size=(2, 2), src_orig_dist=5, src_det_dist=10)
svg = ts.svg(vg_rot, pg)
svg.save("intro_forward_projection_geometries_rot.svg")
A = ts.operator(vg_rot, pg)
#%%
from 3D2D import ProjectionMatrix

