import tomosipo as ts
import numpy as np
from scipy import ndimage
from scipy import signal
from medpy.io import load
import matplotlib.pyplot as plt
import torch
import time
from registration import ProjectionMatrix
from registration import helper_functions
from registration import normxcorr2
from registration import DownweightingMap

base_path='C:/Users/lliu10/OneDrive - Inside MD Anderson/siemenproject/20230214_ChestPhantom/'
mage_data, image_header = load(base_path+'6_1')

# fluro, fh = load(base_path+'43_1')
#%% plot
plt.hist(mage_data.flatten())
#%%
plt.imshow(mage_data[:, :, 0],cmap='gray',vmin=0, vmax=10000)
#%% mage data threshold
for i in range(np.shape(mage_data)[0]):
    mage_data[i,:,:]=helper_functions.thr_image(mage_data[i,:,:],0,1700)
#%% mage data pad

#%%
vg = ts.volume(shape=np.shape(mage_data), size=(1, 1, 1))
pg = ts.cone(angles=9, shape=(512, 512), size=(1.5, 1.5), src_orig_dist=5, src_det_dist=10)
# vg = ts.volume(shape=np.shape(mage_data_pad), size=(1, 1, 1))
# pg = ts.cone(angles=9, shape=(512, 512), size=(1, 1), src_orig_dist=5, src_det_dist=10)
svg = ts.svg(vg, pg)
svg.save("intro_forward_projection_geometries_cone.svg")
A = ts.operator(vg, pg)
#%%
start_time = time.time()
y=A(mage_data)
# y=A(mage_data_pad)
print("--- %s seconds ---" % (time.time() - start_time))
#%% gpu
mage_data_gpu=torch.from_numpy(mage_data)
start_time = time.time()
y=A(mage_data_gpu)
print("--- %s seconds ---" % (time.time() - start_time))
#%%
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(y[:, i, :],cmap='gray') # first projection
    plt.axis('off')
#%% gradient image
for i in range(9):
    img=DownweightingMap.downweightMap(y[:, i, :])
    plt.subplot(3, 3, i+1)
    plt.imshow(img,cmap=plt.cm.gray)
    plt.axis('off')
#%%
# cor = signal.correlate2d(sx, sy)
cor = normxcorr2.normxcorr2(y[:, i, :], y[:, i, :], mode='same')
#%%