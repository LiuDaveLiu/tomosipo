
import DownweightingMap as dm

dx1, dy1 = dm.(img1, ddepth, 0, 1)
dx2 = cv2.Sobel(img2, ddepth, 1, 0)
dy2 = cv2.Sobel(img2, ddepth, 0, 1)

def downweightMap(drr, **kwargs):
    ddepth = cv2.CV_32F
    