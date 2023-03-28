import numpy as np
import matplotlib.pyplot as plt
import os
import time
import cv2
import scipy.io
import scipy.stats as st

def gkern(kernlen=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2*nsig+1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    kernel = kernel/kernel.max()
    return kernel

g_mask = gkern(560, 3)
g_mask = gkern(680, 3)

g_mask = np.dstack((g_mask, g_mask, g_mask))


def syn_data(t, r, sigma):
    t = np.power(t, 2.2)
    r = np.power(r, 2.2)

    sz = int(2*np.ceil(2*sigma)+1)
    r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
    blend = r_blur+t

    att = 1.08+np.random.random()/10.0

    for i in range(3):
        maski = blend[:, :, i] > 1
        mean_i = max(1., np.sum(blend[:, :, i]*maski)/(maski.sum()+1e-6))
        r_blur[:, :, i] = r_blur[:, :, i]-(mean_i-1)*att
    r_blur[r_blur >= 1] = 1
    r_blur[r_blur <= 0] = 0

    h, w = r_blur.shape[0:2]
    # neww = np.random.randint(0, 560-w-10)
    # newh = np.random.randint(0, 560-h-10)

    neww=newh=0
    alpha1 = g_mask[newh:newh+h, neww:neww+w, :]
    alpha2 = 1-np.random.random()/5.0
    r_blur_mask = np.multiply(r_blur, alpha1)
    blend = r_blur_mask+t*alpha2

    t = np.power(t, 1/2.2)
    r_blur_mask = np.power(r_blur_mask, 1/2.2)
    blend = np.power(blend, 1/2.2)
    blend[blend >= 1] = 1
    blend[blend <= 0] = 0

    return t, r_blur_mask, blend


if __name__ == "__main__":
    k_sz = np.linspace(1, 5, 80)
    # syn_image1_path = r'C:\Users\lingzhi.zhu\Downloads\1135px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg'
    # syn_image2_path = r'C:\Users\lingzhi.zhu\Downloads\1_NZ_699.jpg'

    syn_image1_path = '/data/zhenyu/moire/train/natural/coco/000000196747.jpg'
    syn_image2_path = '/data/zhenyu/moire/test/layers_ori/xiaomi10_labtv_219.jpg'

    # os.makedirs('./ZHENYU')

    for sigma in k_sz:

        syn_image1 = cv2.imread(syn_image1_path, -1)
        syn_image2 = cv2.imread(syn_image2_path, -1)
        # neww = np.random.randint(256, 480)
        # neww = 480
        # newh = round((neww/syn_image1.shape[1])*syn_image1.shape[0])
        newh,neww = syn_image1.shape[:2]
        output_image_t = cv2.resize(np.float32(
            syn_image1), (neww, newh), cv2.INTER_CUBIC)/255.0
        output_image_r = cv2.resize(np.float32(syn_image2), (neww, newh), cv2.INTER_CUBIC)/255.0
        # file = os.path.splitext(os.path.basename(syn_image1_list[id]))[0]
        # sigma = k_sz[np.random.randint(0, len(k_sz))]
        assert np.mean(output_image_t)*1/2 < np.mean(output_image_r)

        _, output_image_r, input_image = syn_data(
            output_image_t, output_image_r, sigma)

        output_image_r = np.uint8(output_image_r*255)
        input_image = np.uint8(input_image * 255)
        cv2.imwrite(f"./ZHENYU/new_syn_{sigma:.3f}.png", input_image)

