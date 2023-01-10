# import cv2
import os
from PIL import Image,ImageFilter, ImageOps
import shutil

ori_path = 'xiaomi10_labtv_219.jpg'
output_path = 'segmented.jpg'
img = Image.open(ori_path)

# https://pythontic.com/image-processing/pillow/edge-detection
# Like the other image filter implementations provided
# by Pillow, edge detection filter as well is implemented
# using a convolution of a specific kernel onto the image.
# The convolution matrix used by pillow for the edge detection
# is given by:
#  (-1, -1, -1,
#   -1,  8, -1,
#   -1, -1, -1)

edge_filter = ImageFilter.FIND_EDGES # kernel for conv
temp_img = img.filter(edge_filter) # start filtering

# convert and invert the image
convrt_img = temp_img.convert('L')
img_output = ImageOps.invert(convrt_img)

img_output.save(output_path)