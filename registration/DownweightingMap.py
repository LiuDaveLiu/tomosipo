import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.util import invert


def downweightMap(drr, **kwargs):
    grad_drr = map_to_interval(np.array(getGradientMagnitude(drr)), (0,1))
    inv_grad_drr = invert(grad_drr)
    w_prime = gaussian(inv_grad_drr, **kwargs)
    return w_prime


def map_to_interval(x, to_interval, from_interval=None):
    a, b = (np.min(x), np.max(x)) if from_interval is None else from_interval
    c, d = to_interval
    return (x - a) * (d - c) / (b - a) + c


def getGradientMagnitude(im):
    "Get magnitude of gradient for given image"
    ddepth = cv2.CV_32F
    dx = cv2.Sobel(im, ddepth, 1, 0)
    dy = cv2.Sobel(im, ddepth, 0, 1)
    mag = cv2.magnitude(dx, dy)
    return mag
