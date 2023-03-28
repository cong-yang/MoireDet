# -*- coding: utf-8 -*-
# @Time    : 2019/8/23 21:52
# @Author  : zhoujun

from torch.utils.data import DataLoader
from torchvision import transforms
import copy
import pathlib
from .dataset import Dataloader
# from .dataset_dct import IC15Loader
# from .online_dataset import IC15Loader


def get_datalist(train_data_path, validation_split=0.1):
    """
    获取训练和验证的数据list
    :param train_data_path: 训练的dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param validation_split: 验证集的比例，当val_data_path为空时使用
    :return:
    """
    train_data_list = []
    for train_path in train_data_path:
        train_data = []
        for p in train_path:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').replace('.jpg ', '.jpg\t').split('\t')
                    if len(line) > 1:
                        img_path = pathlib.Path(line[0].strip(' '))
                        label_path = pathlib.Path(line[1].strip(' '))
                        if img_path.exists() and img_path.stat().st_size > 0 and label_path.exists() and label_path.stat().st_size > 0:
                            train_data.append((str(img_path), str(label_path)))
        train_data_list.append(train_data)
    return train_data_list


def get_dataset(data_list, module_name, transform, dataset_args):
    """
    获取训练dataset
    :param data_list: dataset文件列表，每个文件内以如下格式存储 ‘path/to/img\tlabel’
    :param module_name: 所使用的自定义dataset名称，目前只支持data_loaders.ImageDataset
    :param transform: 该数据集使用的transforms
    :param dataset_args: module_name的参数
    :return: 如果data_path列表不为空，返回对于的ConcatDataset对象，否则None
    """
    s_dataset = getattr(dataset, module_name)(transform=transform, data_list=data_list,
                                              **dataset_args)
    return s_dataset


def get_dataloader(module_name, module_args,is_transform=True):

    # 创建数据集
    dataset_args = copy.deepcopy(module_args['dataset'])
    # train_data_path = dataset_args.pop('train_data_path')
    # train_data_ratio = dataset_args.pop('train_data_ratio')
    dataset_args.pop('val_data_path')

    base_dir = dataset_args.get('base_dir','/data/zhenyu/moire/train')

    shuffle = module_args['loader']['shuffle']
    if not is_transform:
        shuffle = False

    try:
        img_size = module_args['loader']['img_size']
    except:
        img_size = 320

    train_loader = DataLoader(dataset=Dataloader(base_dir,is_transform,img_size=(img_size,img_size)),
                                  batch_size=module_args['loader']['train_batch_size'],
                                  shuffle=shuffle,
                                  num_workers=module_args['loader']['num_workers'])
    # train_loader.dataset_len = len(train_dataset_list[0])
    # else:
    #     raise Exception('no images found')
    return train_loader
