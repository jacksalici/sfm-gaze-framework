import numpy as np
import cv2


img_path = '/Users/jacksalici/Desktop/SfmTesting/Test2/GazeOutput/img0.jpg'

img = cv2.imread(img_path)

npz = np.load('/Users/jacksalici/Desktop/SfmTesting/Test2/GazeOutput/img0.npz')



cv2.circle(img, npz['gaze_center_in_rgb_pixels'].astype(int), 4,(255,0,0),3)

cv2.imshow("test", img)
cv2.waitKey()