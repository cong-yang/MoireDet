# import cv2
import os
from PIL import Image,ImageFilter, ImageOps
import shutil

# # settings
# moire_img_path = 'F:\Cong\moire_images'
# moire_layer_path = '/data/cong/moire/ori'
# moire_layer_check = '/data/cong/moire/conv_checking'
# crop_percentage = [0.05, 0.1, 0.05, 0.05] # left, right, top, bottom
# filter_thres = {'jianguo_aoc':25.0887,
#                 'jianguo_labtv':23.0492,
#                 'jianguo_toshiba':25.3285,
#                 'note3_aoc':77.6915,
#                 'note3_labtv':94.4947,
#                 'note3_toshiba':132.9853,
#                 'xiaomi10_aoc':37.1271,
#                 'xiaomi10_labtv':40.154,
#                 'xiaomi10_toshiba':54.1836}
# moire_layer_final = '/data/cong/moire/final'

# # first step: walk all images
# image_paths = {'ori_path':[], 'new_path':[]}
# categoryList = [c for c in sorted(os.listdir(moire_img_path))
#                    if c[0] != '.' and
#                    os.path.isdir(os.path.join(moire_img_path, c))]
#
# for category in categoryList:
#    if category:
#       walkPath = os.path.join(moire_img_path, category)
#       names = os.listdir(walkPath)
#       for name in names:
#          ori_path = os.path.join(walkPath, name)
#          new_name = category + "_" + name
#          new_path = os.path.join(moire_layer_path, new_name)
#          image_paths['ori_path'].append(ori_path)
#          image_paths['new_path'].append(new_path)

# # second step: crop images to avoid the frame edges
# for i in range(len(image_paths['ori_path'])):
#    ori_img = cv2.imread(image_paths['ori_path'][i])
#    height = ori_img.shape[0]
#    width = ori_img.shape[1]
#    left_start = int(width * crop_percentage[0])
#    left_end = int(width - width * crop_percentage[1])
#    top_start = int(height * crop_percentage[2])
#    top_end = int(height - height * crop_percentage[3])
#    crop_image = ori_img[top_start:top_end, left_start:left_end, :]
#    cv2.imwrite(image_paths['new_path'][i], crop_image)
#    print(i, '-->', len(image_paths['ori_path']))

# # third step: calculate conv for filtering
moire_layer_path = '/home/users/zhenyu.yang/data/research/moire_new/train/moire_LCD'
moire_layer_check = '/home/users/zhenyu.yang/data/research/moire_new/train/moire_LCD_convert'


moire_layer_path = '/home/users/zhenyu.yang/projects/moire_detail_detection/data_merge/moire_LCD_6'
moire_layer_check = '/home/users/zhenyu.yang/projects/moire_detail_detection/data_merge/moire_LCD_6_conv'


try:
    os.makedirs(os.path.join(moire_layer_check, 'single'))
    os.makedirs(os.path.join(moire_layer_check, 'combine'))
except:
    pass

names = os.listdir(moire_layer_path)
def get_moire(name):
   ori_path = os.path.join(moire_layer_path, name)
   new_name = name.split('.jpg')[0] + '.png'
   temp_path = os.path.join(moire_layer_check, 'single', new_name)
   temp_compare = os.path.join(moire_layer_check, 'combine', name)
   if not os.path.exists(temp_compare):
      # https://pythontic.com/image-processing/pillow/edge-detection
      # Like the other image filter implementations provided
      # by Pillow, edge detection filter as well is implemented
      # using a convolution of a specific kernel onto the image.
      # The convolution matrix used by pillow for the edge detection
      # is given by:
      #  (-1, -1, -1,
      #   -1,  8, -1,
      #   -1, -1, -1)
      img = Image.open(ori_path)
      img = ImageOps.invert(img)
      width,height = img.size
      img_new = Image.new('RGB',(2*width,height)) # to display both images
      img_new.paste(img,(0,0)) # paste the ori image to the left
      edge_filter = ImageFilter.FIND_EDGES # kernel for conv
      temp_img = img.filter(edge_filter) # start filtering

      # convert and invert the image
      convrt_img = temp_img.convert('L')
      img_output = ImageOps.invert(convrt_img)

      # save the processed filter and comparision
      img_new.paste(img_output, (width,0)) # paste the filtered image to the right
      img_output.save(temp_path)
      img_new.save(temp_compare)

from multiprocessing import Pool
pool = Pool(32)
pool.map(get_moire,names)
pool.close()
pool.join()

# fourth step: check texture density to find threshold (codes in matlab)
# clear;
# PathRoot='F:\Cong\moire_layers\conv_checking\single\';
# PathCombine = 'F:\Cong\moire_layers\conv_checking\combine\';
#
# namelist = dir(fullfile(PathRoot));
# len = length(namelist);
# for i = 3:len
#     name = namelist(i).name;
#     img_path = [PathRoot, name];
#     temp_img = imread(img_path);
#     thres = sum(temp_img(:)<250)/10000;
#     temp_name = regexp(name, '.png', 'split');
#     file_ori = [PathCombine, temp_name{1}, '.jpg'];
#     new_name = [temp_name{1},'_', num2str(thres), '.jpg'];
#     file_new = [PathCombine, new_name];
#     movefile(file_ori, file_new);
#     disp([num2str(i),'_', num2str(len),'_',num2str(thres)])
# end

# # fifth step: select qualified moire layers
# for key in filter_thres:
#    threshold = filter_thres[key]
#    walkPath = os.path.join(moire_layer_check, 'combine', key)
#    names = os.listdir(walkPath)
#    for name in names:
#       own_thres = float(name.split('_')[3].split('.jpg')[0])
#       basic_name = name.split('_')[0]+'_'+name.split('_')[1]+'_'+name.split('_')[2]
#       if own_thres > threshold:
#          # copy the original layer
#          path_ori = os.path.join(moire_layer_path,basic_name+'.jpg')
#          path_tar = os.path.join(moire_layer_final, 'layers_ori', basic_name+'.jpg')
#          if not os.path.exists(path_tar):
#             shutil.copy(path_ori, path_tar)
#
#          # copy the conv layer
#          path_ori = os.path.join(moire_layer_check,'single',basic_name+'.png')
#          path_tar = os.path.join(moire_layer_final,'layers_conv',basic_name+'.png')
#          if not os.path.exists(path_tar):
#             shutil.copy(path_ori, path_tar)
#
#          # copy the combined images
#          path_ori = os.path.join(walkPath, name)
#          path_tar = os.path.join(moire_layer_final,'checking',key,name)
#          if not os.path.exists(path_tar):
#             shutil.copy(path_ori, path_tar)
#       print(key, name)

# temp copy files
# moire_layer_path = '/data/cong/moire/ori'
# moire_layer_check = '/data/cong/moire/conv_checking'
# filter_thres = {'jianguo_aoc':25.0887,
#                 'jianguo_labtv':23.0492,
#                 'jianguo_toshiba':25.3285,
#                 'note3_aoc':77.6915,
#                 'note3_labtv':94.4947,
#                 'note3_toshiba':132.9853,
#                 'xiaomi10_aoc':37.1271,
#                 'xiaomi10_labtv':40.154,
#                 'xiaomi10_toshiba':54.1836}
# moire_layer_final = '/data/cong/moire/final'
#
# data=open("/data/cong/moire/final/list.txt")
# #data=open("F:\Cong\moire_layers\\final\list.txt")
# for line in data.readlines():
#    name = line
#    basic_name = name.split('.png')[0]
#    phonename = name.split('_')[0] + '_' + name.split('_')[1]
#
#    # copy the ori layer
#    path_ori = os.path.join(moire_layer_path,basic_name+'.jpg')
#    path_tar = os.path.join(moire_layer_final, 'layers_ori', basic_name+'.jpg')
#    if not os.path.exists(path_tar):
#       shutil.copy(path_ori, path_tar)
#
#    # copy the conv layer
#    path_ori = os.path.join(moire_layer_check,'single',basic_name+'.png')
#    path_tar = os.path.join(moire_layer_final,'layers_conv',basic_name+'.png')
#    if not os.path.exists(path_tar):
#       shutil.copy(path_ori, path_tar)
#
#    # copy the combined images
#    path_ori = os.path.join(moire_layer_check, 'combine', phonename, basic_name+'.jpg')
#    path_tar = os.path.join(moire_layer_final,'checking',phonename, basic_name+'.jpg')
#    if not os.path.exists(path_tar):
#       shutil.copy(path_ori, path_tar)
#
#    print(basic_name)