"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os
# debug cityscapes
import json
from data.city_helpers.annotation import Annotation
from data.city_helpers.labels import labels, name2label
from skimage.draw import polygon
import numpy as np

import PIL.Image
import pickle

import torch
import cv2

from skimage import measure

from random import shuffle

import scipy.io as sio


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt
        if self.opt.retrival:
            label_paths, image_paths, instance_paths, label_paths_train, image_paths_train = self.get_paths(opt)
            util.natural_sort(label_paths_train)
            util.natural_sort(image_paths_train)
            label_paths_train = label_paths_train[:opt.max_dataset_size]
            image_paths_train = image_paths_train[:opt.max_dataset_size]
            self.label_paths_train = label_paths_train
            self.image_paths_train = image_paths_train
        elif self.opt.retrival_memory:
            label_paths, image_paths, instance_paths, label_paths_train, image_paths_train = self.get_paths(opt)
            util.natural_sort(label_paths_train)
            util.natural_sort(image_paths_train)
            label_paths_train = label_paths_train[:opt.max_dataset_size]
            image_paths_train = image_paths_train[:opt.max_dataset_size]
            self.label_paths_train = label_paths_train
            self.image_paths_train = image_paths_train
        else:
            label_paths, image_paths, instance_paths, label_paths_train, image_paths_train = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)
        util.natural_sort(instance_paths)

        
        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        perm = list(range(len(label_paths)))
        shuffle(perm)
        label_paths_fine = [label_paths[index] for index in perm]
        image_paths_fine = [image_paths[index] for index in perm]
        if not opt.no_instance:
            instance_paths_fine = [instance_paths[index] for index in perm]


        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)
            for path1, path2 in zip(label_paths_fine, image_paths_fine):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        self.label_paths_fine = label_paths_fine
        self.image_paths_fine = image_paths_fine
        if not opt.no_instance:
            self.instance_paths_fine = instance_paths_fine

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    #get item for training for cityscapes
    # def __getitem__(self, index):
    #     #load json for cityscape

    #     # Label Image
    #     if self.opt.box:
    #         label_path = self.label_paths[index]
    #         label = Image.open(label_path)
    #         params = get_params(self.opt, label.size)
    #         transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
    #         fine_label_tensor = transform_label(label) * 255.0
    #         fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1),fine_label_tensor.size(2))

    #         # box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
    #         box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
    #         label_np = pickle.load(open(box_path,'rb'))
    #         label_tensor = torch.from_numpy(label_np).float()


    #     elif self.opt.box_unpair:
    #         label_path = self.label_paths[index]
    #         label_path_fine = self.label_paths_fine[index]
    #         label = Image.open(label_path_fine)
    #         params = get_params(self.opt, label.size)
    #         transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
    #         fine_label_tensor = transform_label(label) * 255.0
    #         fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1),fine_label_tensor.size(2))

    #         # box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
    #         box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
    #         label_np = pickle.load(open(box_path,'rb'))
    #         label_tensor = torch.from_numpy(label_np).float()

    #         # input image (real images fine)
    #         image_path_fine = self.image_paths_fine[index]
    #         assert self.paths_match(label_path_fine, image_path_fine), \
    #             "The label_path %s and image_path %s don't match." % \
    #             (label_path_fine, image_path_fine)
    #         image_fine = Image.open(image_path_fine)
    #         image_fine = image_fine.convert('RGB')

    #         transform_image = get_transform(self.opt, params)
    #         image_tensor_fine = transform_image(image_fine)


    #     else:
    #         label_path = self.label_paths[index]
    #         label = Image.open(label_path)
    #         params = get_params(self.opt, label.size)
    #         transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
    #         label_tensor = transform_label(label) * 255.0
    #         fine_label_tensor = label_tensor


    #     label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

    #     # input image (real images)
    #     image_path = self.image_paths[index]
    #     assert self.paths_match(label_path, image_path), \
    #         "The label_path %s and image_path %s don't match." % \
    #         (label_path, image_path)
    #     image = Image.open(image_path)
    #     image = image.convert('RGB')

    #     transform_image = get_transform(self.opt, params)
    #     image_tensor = transform_image(image)

    #     # if using instance maps
    #     if self.opt.no_instance:
    #         instance_tensor = 0
    #     else:
    #         instance_path = self.instance_paths[index]
    #         instance = Image.open(instance_path)
    #         if instance.mode == 'L':
    #             instance_tensor = transform_label(instance) * 255
    #             instance_tensor = instance_tensor.long()
    #         else:
    #             instance_tensor = transform_label(instance)
    #     if self.opt.box_unpair:
    #         input_dict = {'label': label_tensor,
    #                       'instance': instance_tensor,
    #                       'image': image_tensor,
    #                       'path': image_path,
    #                       'fine_label': fine_label_tensor,
    #                       'image_fine':image_tensor_fine,
    #                       }
    #     else:
    #         input_dict = {'label': label_tensor,
    #                       'instance': instance_tensor,
    #                       'image': image_tensor,
    #                       'path': image_path,
    #                       'fine_label': fine_label_tensor,
    #                       }

    #     # Give subclasses a chance to modify the final output
    #     self.postprocess(input_dict)

        # return input_dict


    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

    # ade20k boxes

    # def __getitem__(self, index):
    #     #load json for cityscape
    #     # Label Image
    #     label_path = self.label_paths[index]
    #     label = Image.open(label_path)
    #     params = get_params(self.opt, label.size)
    #     transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
    #     label_tensor = transform_label(label) * 255.0


    #     label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

    #     # input image (real images)
    #     image_path = self.image_paths[index]
    #     assert self.paths_match(label_path, image_path), \
    #         "The label_path %s and image_path %s don't match." % \
    #         (label_path, image_path)
    #     image = Image.open(image_path)
    #     image = image.convert('RGB')

    #     transform_image = get_transform(self.opt, params,normalize=False)
    #     image_tensor = transform_image(image)

    #     # if using instance maps
    #     if self.opt.no_instance:
    #         instance_tensor = 0
    #     else:
    #         instance_path = self.instance_paths[index]
    #         instance = Image.open(instance_path)
    #         if instance.mode == 'L':
    #             instance_tensor = transform_label(instance) * 255
    #             instance_tensor = instance_tensor.long()
    #         else:
    #             instance_tensor = transform_label(instance)

    #     image_np = image_tensor.data.cpu().numpy()
    #     label_np = label_tensor.data.cpu().numpy()[0]
    #     label_np = label_np-1
    #     # ins_np = instance_tensor.data.cpu().numpy()[0]
    #     objects_stuff_mat = sio.loadmat('/datadrive/yandong/sceneparsing/objectSplit35-115.mat')
    #     objects_stuff_list = objects_stuff_mat['stuffOrobject']
    #     object_list = []
    #     for object_id in range(len(objects_stuff_list)):
    #         if objects_stuff_list[object_id] == 2:
    #             object_list.append(object_id)
    #     # print(len(object_list))

    #     # background class is 34
    #     save_img = np.zeros(image_np.shape)
    #     # other as background
    #     save_label = np.zeros(label_np.shape) 
    #     # save_instance = np.zeros(ins_np.shape) 

    #     label_onehot = np.zeros((1,151,) + label_np.shape) 
    #     label_onehot_tensor = torch.from_numpy(label_onehot).float()
    #     # print(label_onehot_tensor.size())

    #     # save_label = np.zeros(label_np.shape) + 34
    #     # save_instance = np.zeros(ins_np.shape) + 34

    #     label_seq = np.unique(label_np)
    #     # ins_seq = np.unique(ins_np)
    #     # print(label_seq)

    #     # target_folder = 'cityscapes_seq_test'
    #     # print(ins_seq)
    #     # print(label_seq)
    #     # foregrounds = [17,18,19,20,24,25,26,27,28,29,30,31,32,33]
    #     foregrounds = [24,25,26,27,28,29,30,31,32,33]
    #     # object_list = [126,150,139]
    #     # foregrounds = [26]
    #     target_folder = 'ade20k_box_test'

    #     for label_id in label_seq:
    #         for obj in object_list:
    #             if str(obj) in str(label_id):
    #                 temp_label = np.zeros(label_np.shape)
    #                 temp_label[label_np==label_id] = label_id
    #                 # ret, thresh = cv2.threshold(save_label, 125, 255, 0)
    #                 # thresh = thresh.astype('uint8')
    #                 temp_label = temp_label.astype('uint8')
    #                 contours, hierarchy = cv2.findContours(temp_label,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #                 for conid in range(len(contours)):
    #                     mask = np.zeros(label_np.shape, dtype="uint8") * 255
    #                     cv2.drawContours(mask, contours, conid, int(label_id), -1)
    #                     i, j = np.where(mask==label_id)
    #                     indices = np.meshgrid(np.arange(min(i), max(i) + 1),
    #                                           np.arange(min(j), max(j) + 1),
    #                                           indexing='ij')
    #                     save_label = np.zeros(label_np.shape)
    #                     save_label[indices] = obj
    #                     # save_label[label_np==label_id] = obj
    #                     save_label_batch = save_label.reshape((1,1,)+save_label.shape)
    #                     save_label_tensor = torch.from_numpy(save_label_batch).long()
    #                     label_onehot_tensor.scatter_(1, save_label_tensor, 1.0)

    #                     # save_img[0][indices] = image_np[0][indices]
    #                     # save_img[1][indices] = image_np[1][indices]
    #                     # save_img[2][indices] = image_np[2][indices]


    #     # save_img_trans = np.asarray(save_img*255.0, dtype=np.uint8).transpose(1,2,0)
    #     # PIL.Image.fromarray(save_img_trans, 'RGB').save('test.png')
    #     # PIL.Image.fromarray(save_label, 'L').save('test2.png')

    #     # Test scatter
    #     # cars = label_onehot_tensor[0][26]
    #     # cars_np = cars.data.numpy()
    #     # print(np.sum(cars_np[label_np==26]))
    #     # print(np.sum(label_np==26))
    #     label_onehot_np = label_onehot_tensor.data.numpy().reshape((self.opt.label_nc+1,) + label_np.shape)
    #     label_onehot_np = np.asarray(label_onehot_np,dtype=np.uint8)

    #     label_onehot_bit = np.packbits(label_onehot_np, axis=None)

    #     label_onehot_byte = np.unpackbits(label_onehot_bit, axis=None)[:label_onehot_np.size].reshape(label_onehot_np.shape).astype(np.uint8)

    #     print(label_onehot_np.shape)
    #     save_label_path = label_path.replace('ADEChallengeData2016',target_folder)
    #     save_label_path = save_label_path.replace('.png','_boxes.pkl')


    #     print(label_path)
    #     print(save_label_path)


    #     # pickle.dump(label_onehot_bit,open(save_label_path,'wb'),True)
    #     load_label_byte = pickle.load(open(save_label_path,'rb'))

    #     label_one_np = np.asarray(np.zeros((151,)+label_tensor[0].size()),dtype=np.uint8)
    #     print(label_one_np.shape)

    #     load_label_byte = np.unpackbits(load_label_byte, axis=None)[:label_one_np.size].reshape(label_one_np.shape).astype(np.uint8)

    #     print(np.array_equal(label_onehot_np, load_label_byte))
    #     # print(label_onehot_np)
    #     # print(load_label_byte)
    #     # index = np.arange(0,151*256*256).reshape(151,256,256)
    #     # index = np.arange(0,256*256).reshape(256,256)

    #     # print(index[load_label_byte != label_np].shape)
    #     # print(label_np[load_label_byte != label_np])
    #     # print(load_label_byte[load_label_byte != label_np])
    #     # print((load_label_byte==label_onehot_np).all())

    #     input_dict = {'label': label_tensor,
    #                   'instance': instance_tensor,
    #                   'image': image_tensor,
    #                   'path': image_path,
    #                   }

    #     # Give subclasses a chance to modify the final output
    #     self.postprocess(input_dict)

    #     return input_dict


    # prepare city_seq_dataset
    # python train.py --name self_city_debug --dataset_mode cityscapes --dataroot datasets/cityscapes/ --serial_batches
    # def __getitem__(self, index):
    #     #load json for cityscape

    #     # Label Image
    #     label_path = self.label_paths[index]
    #     label = Image.open(label_path)
    #     params = get_params(self.opt, label.size)
    #     transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
    #     label_tensor = transform_label(label) * 255.0


    #     label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

    #     # input image (real images)
    #     image_path = self.image_paths[index]
    #     assert self.paths_match(label_path, image_path), \
    #         "The label_path %s and image_path %s don't match." % \
    #         (label_path, image_path)
    #     image = Image.open(image_path)
    #     image = image.convert('RGB')

    #     transform_image = get_transform(self.opt, params,normalize=False)
    #     image_tensor = transform_image(image)

    #     # if using instance maps
    #     if self.opt.no_instance:
    #         instance_tensor = 0
    #     else:
    #         instance_path = self.instance_paths[index]
    #         instance = Image.open(instance_path)
    #         if instance.mode == 'L':
    #             instance_tensor = transform_label(instance) * 255
    #             instance_tensor = instance_tensor.long()
    #         else:
    #             instance_tensor = transform_label(instance)

    #     image_np = image_tensor.data.cpu().numpy()
    #     label_np = label_tensor.data.cpu().numpy()[0]
    #     ins_np = instance_tensor.data.cpu().numpy()[0]


    #     # background class is 34
    #     save_img = np.zeros(image_np.shape)
    #     # other as background
    #     save_label = np.zeros(label_np.shape) 
    #     save_instance = np.zeros(ins_np.shape) 

    #     # save_label = np.zeros(label_np.shape) + 34
    #     # save_instance = np.zeros(ins_np.shape) + 34

    #     label_seq = np.unique(label_np)

    #     # target_folder = 'cityscapes_seq_test'
    #     seg_len = len(label_seq)/5
    #     for segid in range(5):
    #         seg_list = label_seq[int(segid*seg_len):int((segid+1)*seg_len)]
    #         for lable_id in seg_list:
    #             save_img[0][label_np==lable_id] = image_np[0][label_np==lable_id]
    #             save_img[1][label_np==lable_id] = image_np[1][label_np==lable_id]
    #             save_img[2][label_np==lable_id] = image_np[2][label_np==lable_id]

    #             save_label[label_np==lable_id] = label_np[label_np==lable_id]
    #             save_instance[label_np==lable_id] = ins_np[label_np==lable_id]
    #         # print(np.unique(save_label))
    #         # print(np.unique(save_instance))

    #         save_img_path = image_path.replace('cityscapes',target_folder)
    #         save_label_path = label_path.replace('cityscapes',target_folder)
    #         save_instance_path = instance_path.replace('cityscapes',target_folder)


    #         save_img_path = save_img_path.replace('_leftImg8bit','_'+str(segid)+'_leftImg8bit')
    #         save_label_path = save_label_path.replace('_gtFine_labelIds','_'+str(segid)+'_gtFine_labelIds')
    #         save_instance_path = save_instance_path.replace('_gtFine_instanceIds','_'+str(segid)+'_gtFine_instanceIds')

    #         save_img_trans = np.asarray(save_img*255.0, dtype=np.uint8).transpose(1,2,0)
    #         save_label = np.asarray(save_label, dtype=np.uint8)
    #         save_instance = np.asarray(save_instance, dtype=np.uint8)


    #         PIL.Image.fromarray(save_img_trans, 'RGB').save(save_img_path)
    #         PIL.Image.fromarray(save_label, 'L').save(save_label_path)
    #         PIL.Image.fromarray(save_instance, 'L').save(save_instance_path)


    #     # json_file = self.label_paths[index][:-13]+'_polygons.json'
    #     # annotation = Annotation()
    #     # annotation.fromJsonFile(json_file)
    #     # for obj in annotation.objects:
    #     #     label   = obj.label
    #     #     polygon_city = obj.polygon

    #     #     rrlist = []
    #     #     cclist = []
    #     #     for point in polygon_city:
    #     #         rrlist.append(point.x)
    #     #         cclist.append(point.y)
    #     #     # print(rrlist)
    #     #     # print(cclist)
    #     #     rr = np.asarray(rrlist)
    #     #     cc = np.asarray(cclist)
    #     #     rrall, ccall = polygon(rr, cc)
    #     #     print(instance_tensor[0][cc, rr])

    #     input_dict = {'label': label_tensor,
    #                   'instance': instance_tensor,
    #                   'image': image_tensor,
    #                   'path': image_path,
    #                   }

    #     # Give subclasses a chance to modify the final output
    #     self.postprocess(input_dict)

    #     return input_dict
    # prepare city_box
    # def __getitem__(self, index):
    #     #load json for cityscape

    #     # Label Image
    #     label_path = self.label_paths[index]
    #     label = Image.open(label_path)
    #     params = get_params(self.opt, label.size)
    #     transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
    #     label_tensor = transform_label(label) * 255.0


    #     label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

    #     # input image (real images)
    #     image_path = self.image_paths[index]
    #     assert self.paths_match(label_path, image_path), \
    #         "The label_path %s and image_path %s don't match." % \
    #         (label_path, image_path)
    #     image = Image.open(image_path)
    #     image = image.convert('RGB')

    #     transform_image = get_transform(self.opt, params,normalize=False)
    #     image_tensor = transform_image(image)

    #     # if using instance maps
    #     if self.opt.no_instance:
    #         instance_tensor = 0
    #     else:
    #         instance_path = self.instance_paths[index]
    #         instance = Image.open(instance_path)
    #         if instance.mode == 'L':
    #             instance_tensor = transform_label(instance) * 255
    #             instance_tensor = instance_tensor.long()
    #         else:
    #             instance_tensor = transform_label(instance)

    #     image_np = image_tensor.data.cpu().numpy()
    #     label_np = label_tensor.data.cpu().numpy()[0]
    #     ins_np = instance_tensor.data.cpu().numpy()[0]


    #     # background class is 34
    #     save_img = np.zeros(image_np.shape)
    #     # other as background
    #     save_label = np.zeros(label_np.shape) 
    #     save_instance = np.zeros(ins_np.shape) 

    #     label_onehot = np.zeros((1,35,) + ins_np.shape) 
    #     label_onehot_tensor = torch.from_numpy(label_onehot).float()
    #     # print(label_onehot_tensor.size())

    #     # save_label = np.zeros(label_np.shape) + 34
    #     # save_instance = np.zeros(ins_np.shape) + 34

    #     label_seq = np.unique(label_np)
    #     ins_seq = np.unique(ins_np)

    #     # target_folder = 'cityscapes_seq_test'
    #     # print(ins_seq)
    #     # print(label_seq)
    #     # foregrounds = [17,18,19,20,24,25,26,27,28,29,30,31,32,33]
    #     foregrounds = [24,25,26,27,28,29,30,31,32,33]
    #     # foregrounds = [26]
    #     target_folder = 'cityscapes_box'
    #     for label_id in ins_seq:
    #         for obj in foregrounds:
    #             if str(obj) in str(label_id):
    #                 i, j = np.where(ins_np==label_id)
    #                 indices = np.meshgrid(np.arange(min(i), max(i) + 1),
    #                                       np.arange(min(j), max(j) + 1),
    #                                       indexing='ij')
    #                 save_label[indices] = obj
    #                 # save_label[ins_np==label_id] = obj
    #                 save_label_batch = save_label.reshape((1,1,)+save_label.shape)
    #                 save_label_tensor = torch.from_numpy(save_label_batch).long()
    #                 label_onehot_tensor.scatter_(1, save_label_tensor, 1.0)
    #     # Test scatter
    #     # cars = label_onehot_tensor[0][26]
    #     # cars_np = cars.data.numpy()
    #     # print(np.sum(cars_np[label_np==26]))
    #     # print(np.sum(label_np==26))
    #     label_onehot_np = label_onehot_tensor.data.numpy().reshape((self.opt.label_nc,) + ins_np.shape)
    #     label_onehot_np = np.asarray(label_onehot_np,dtype=np.uint8)

    #     save_label_path = label_path.replace('cityscapes',target_folder)
    #     save_label_path = save_label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')

    #     pickle.dump(label_onehot_np,open(save_label_path,'wb'),-1)
    #     input_dict = {'label': label_tensor,
    #                   'instance': instance_tensor,
    #                   'image': image_tensor,
    #                   'path': image_path,
    #                   }

    #     # Give subclasses a chance to modify the final output
    #     self.postprocess(input_dict)

    #     return input_dict

    # prepare bounding box foreground
    # python train.py --name self_city_debug --dataset_mode cityscapes --dataroot datasets/cityscapes/ --serial_batches
    # def __getitem__(self, index):
    #     #load json for cityscape

    #     # Label Image
    #     label_path = self.label_paths[index]
    #     label = Image.open(label_path)
    #     params = get_params(self.opt, label.size)
    #     transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
    #     label_tensor = transform_label(label) * 255.0


    #     label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

    #     # input image (real images)
    #     image_path = self.image_paths[index]
    #     assert self.paths_match(label_path, image_path), \
    #         "The label_path %s and image_path %s don't match." % \
    #         (label_path, image_path)
    #     image = Image.open(image_path)
    #     image = image.convert('RGB')

    #     transform_image = get_transform(self.opt, params,normalize=False)
    #     image_tensor = transform_image(image)

    #     # if using instance maps
    #     if self.opt.no_instance:
    #         instance_tensor = 0
    #     else:
    #         instance_path = self.instance_paths[index]
    #         instance = Image.open(instance_path)
    #         if instance.mode == 'L':
    #             instance_tensor = transform_label(instance) * 255
    #             instance_tensor = instance_tensor.long()
    #         else:
    #             instance_tensor = transform_label(instance)

    #     image_np = image_tensor.data.cpu().numpy()
    #     label_np = label_tensor.data.cpu().numpy()[0]
    #     ins_np = instance_tensor.data.cpu().numpy()[0]


    #     # background class is 34
    #     save_img = np.zeros(image_np.shape)
    #     # other as background
    #     save_label = np.zeros(label_np.shape) 
    #     save_instance = np.zeros(ins_np.shape) 
    #     save_box = np.zeros(ins_np.shape) 

    #     label_onehot = np.zeros((1,35,) + ins_np.shape) 
    #     label_onehot_tensor = torch.from_numpy(label_onehot).float()
    #     # print(label_onehot_tensor.size())

    #     # save_label = np.zeros(label_np.shape) + 34
    #     # save_instance = np.zeros(ins_np.shape) + 34

    #     label_seq = np.unique(label_np)
    #     ins_seq = np.unique(ins_np)

    #     # target_folder = 'cityscapes_seq_test'
    #     # print(ins_seq)
    #     # print(label_seq)
    #     # foregrounds = [17,18,19,20,24,25,26,27,28,29,30,31,32,33]
    #     foregrounds = [24,25,26,27,28,29,30,31,32,33]
    #     # foregrounds = [26]
    #     target_folder = 'cityscapes_foreground'
    #     for label_id in ins_seq:
    #         for obj in foregrounds:
    #             if str(obj) in str(label_id):
    #                 i, j = np.where(ins_np==label_id)
    #                 indices = np.meshgrid(np.arange(min(i), max(i) + 1),
    #                                       np.arange(min(j), max(j) + 1),
    #                                       indexing='ij')
    #                 save_instance[ins_np==label_id] = label_id
    #                 save_box[indices] = obj
    #                 save_img[0][indices] = image_np[0][indices]
    #                 save_img[1][indices] = image_np[1][indices]
    #                 save_img[2][indices] = image_np[2][indices]
    #                 save_label[label_np==obj] = label_np[label_np==obj]
    #                 # save_label[ins_np==label_id] = obj
    #                 save_box_batch = save_box.reshape((1,1,)+save_box.shape)
    #                 save_box_tensor = torch.from_numpy(save_box_batch).long()
    #                 label_onehot_tensor.scatter_(1, save_box_tensor, 1.0)
    #     # Test scatter
    #     # cars = label_onehot_tensor[0][26]
    #     # cars_np = cars.data.numpy()
    #     # print(np.sum(cars_np[label_np==26]))
    #     # print(np.sum(label_np==26))
    #     label_onehot_np = label_onehot_tensor.data.numpy().reshape((self.opt.label_nc,) + ins_np.shape)
    #     label_onehot_np = np.asarray(label_onehot_np,dtype=np.uint8)

    #     save_label_path = label_path.replace('cityscapes',target_folder)
    #     save_img_path = image_path.replace('cityscapes',target_folder)
    #     save_instance_path = instance_path.replace('cityscapes',target_folder)

    #     save_boxlabel_path = save_label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')

    #     save_img_path = save_img_path.replace('_leftImg8bit','_leftImg8bitforeground')
    #     save_seglabel_path = save_label_path.replace('_gtFine_labelIds','_gtFine_labelIdsforeground')
    #     save_instance_path = save_instance_path.replace('_gtFine_instanceIds','_gtFine_instanceIdsforeground')

    #     save_img_trans = np.asarray(save_img*255.0, dtype=np.uint8).transpose(1,2,0)
    #     save_label = np.asarray(save_label, dtype=np.uint8)
    #     save_instance = np.asarray(save_instance, dtype=np.uint8)


    #     PIL.Image.fromarray(save_img_trans, 'RGB').save(save_img_path)
    #     PIL.Image.fromarray(save_label, 'L').save(save_seglabel_path)
    #     PIL.Image.fromarray(save_instance, 'L').save(save_instance_path)


    #     pickle.dump(label_onehot_np,open(save_boxlabel_path,'wb'),-1)
    #     input_dict = {'label': label_tensor,
    #                   'instance': instance_tensor,
    #                   'image': image_tensor,
    #                   'path': image_path,
    #                   }

    #     # Give subclasses a chance to modify the final output
    #     self.postprocess(input_dict)

    #     return input_dict
