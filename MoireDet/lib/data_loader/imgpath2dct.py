from jpeg2dct.numpy import load, loads
import cv2
import os
import numpy as np

# img_dir = '/home/users/zhenyu.yang/big_data/research/smoke/train_data_png'
# img_list = os.listdir(img_dir)
# img_list = [os.path.join(img_dir,v) for v in img_list]
#
# img_png = cv2.imread(img_list[4], cv2.IMREAD_UNCHANGED)
# img = img_png[:, :, :3]
#
#
# dst_dir = '/home/users/zhenyu.yang/big_data/research/smoke/dct_test_4'
# if not os.path.exists(dst_dir):
#     os.makedirs(dst_dir)
# dst_path = 'temp.jpg'
# dst_path = os.path.join(dst_dir,dst_path)

def path2dct(png_path):
    img_png = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    img = img_png[:, :, :3]

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(95)]
    res, encimg = cv2.imencode('.jpg', img, encode_param)
    dct_y, dct_cb, dct_cr = loads(encimg.tobytes())

    return dct_y, dct_cb, dct_cr

def img2dct(img):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(95)]
    res, encimg = cv2.imencode('.jpg', img, encode_param)
    dct_y, dct_cb, dct_cr = loads(encimg.tobytes())

    return dct_y, dct_cb, dct_cr


# dct_dict = {
#     'dct_y':dct_y,
#     'dct_cb':dct_cb,
#     'dct_cr':dct_cr}
#
# for key,value in dct_dict.items():
#     for i in range(64):
#         img_name = '{}-{}.jpg'.format(key,str(i).zfill(3))
#         img_path = os.path.join(dst_dir,img_name)
#         min_img = value[:,:,i].astype('float')
#         min_img =(min_img/np.max(np.abs(min_img)) +1)/2*255
#         min_img = np.uint8(min_img)
#         cv2.imwrite(img_path,min_img)