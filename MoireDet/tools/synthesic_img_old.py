import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
import scipy.io
import scipy.stats as st

STEP = 50

def syn_part(output_image_t, output_image_r, att):
    '''Input image scale in [0, 1]'''
    sigma = 3 * np.random.random() + 2
    # for sigma in np.linspace(2, 5, 15):

    sz = int(2*np.ceil(2*sigma)+1)

    # r_blur = cv2.GaussianBlur(output_image_r, (sz, sz), sigma, sigma, 0)
    r_blur = output_image_r
    blend = output_image_t + r_blur
    maski = blend[:, :] > 1
    mean_i = max(1., np.sum(blend[:, :]*maski)/(maski.sum()+1e-6))
    r_blur[:, :] = r_blur[:, :]-(mean_i-1)*att

    r_blur[r_blur >= 1] = 1
    r_blur[r_blur <= 0] = 0

    res = r_blur + output_image_t
    res[res >= 1] = 1
    res[res <= 0] = 0
    return res

def syn_part_rgb(output_image_t, output_image_r, att):
    '''Input image scale in [0, 1]'''
    sigma = 3 * np.random.random() + 2
    # for sigma in np.linspace(2, 5, 15):

    sz = int(2*np.ceil(2*sigma)+1)

    # r_blur = cv2.GaussianBlur(output_image_r, (sz, sz), sigma, sigma, 0)
    r_blur = output_image_r
    blend = output_image_t + r_blur
    for i in range(3):
        maski = blend[:, :, i] > 1
        mean_i = max(1., np.sum(blend[:, :, i]*maski)/(maski.sum()+1e-6))
        r_blur[:, :, i] = r_blur[:, :, i]-(mean_i-1)*att

    r_blur[r_blur >= 1] = 1
    r_blur[r_blur <= 0] = 0

    res = r_blur + output_image_t
    # res = np.power(res, 1/2.2)
    res[res >= 1] = 1
    res[res <= 0] = 0
    return res


if __name__ == "__main__":
    # for att in np.linspace(0.9, 4, 20):
    att = 2.5
    syn_image1_path = '/data/zhenyu/moire/train/natural/coco/000000196747.jpg'
    syn_image2_path = '/data/zhenyu/moire/train/layers_ori/note3_labtv_544.jpg'
    moire_path = '/data/zhenyu/moire/train/layers_conv/note3_labtv_544.png'
    moire = cv2.imread(moire_path)

    syn_image1 = cv2.imread(syn_image1_path, -1)
    syn_image2 = cv2.imread(syn_image2_path, -1)
    # neww = np.random.randint(256, 480)
    neww = 638
    newh = round((neww/syn_image1.shape[1])*syn_image1.shape[0])
    output_image_t = cv2.resize(np.float32(
        syn_image1), (neww, newh), cv2.INTER_CUBIC)/255.0
    output_image_r = cv2.resize(np.float32(
        syn_image2), (neww, newh), cv2.INTER_CUBIC)/255.0
    res = syn_part_rgb(output_image_t, output_image_r, att)

    cv2.imwrite('test.jpg',res)
    cv2.imwrite('ori.jpg', syn_image1)
    cv2.imwrite('moire.jpg', syn_image2)
    cv2.imwrite('moire_conv.jpg', moire)



    #######################################################
    # for i in range(0, output_image_t.shape[0], STEP):
    #     for j in range(0, output_image_t.shape[1], STEP):
    #         if i + STEP > output_image_t.shape[0]:
    #             x2 = output_image_t.shape[0]
    #         else:
    #             x2 = i + STEP
    #         if j + STEP > output_image_t.shape[1]:
    #             y2 = output_image_t.shape[1]
    #         else:
    #             y2 = j + STEP
    #         res[i:x2, j:y2] = syn_part(cv2.equalizeHist((output_image_t[i:x2, j:y2] * 255).astype(np.uint8)) / 255, output_image_r[i:x2, j:y2])
    
    # cv2.imshow('res', res)
    # cv2.waitKey(0)
