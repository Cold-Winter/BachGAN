"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_params, get_transform

from PIL import Image
import PIL.Image
import pickle
import numpy as np

import torch
import cv2
import os
from skimage import measure

from random import shuffle

import scipy.io as sio

class ADE20KDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=150)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        all_images = make_dataset(root, recursive=True, read_cache=False, write_cache=False)
        image_paths = []
        label_paths = []
        for p in all_images:
            if '_%s_' % phase not in p:
                continue
            if p.endswith('.jpg'):
                image_paths.append(p)
            elif p.endswith('.png'):
                label_paths.append(p)

        instance_paths = []  # don't use instance map for ade20k

        image_paths_train = []
        label_paths_train = []
        phase = 'train'
        for p in all_images:
            if '_%s_' % phase not in p:
                continue
            if p.endswith('.jpg'):
                image_paths_train.append(p)
            elif p.endswith('.png'):
                label_paths_train.append(p)

        return label_paths, image_paths, instance_paths,label_paths_train, image_paths_train
    def get_onehot_box_tensor(self, fine_label_tensor):
        label_np = fine_label_tensor.data.cpu().numpy()
        label_np = label_np-1
        # should be label[label == -1] = self.opt.label_nc unlabeled is 150
        # then the label_onehot = np.zeros((1,151,) + label_np.shape) + 150 --> 
        # ins_np = instance_tensor.data.cpu().numpy()[0]
        objects_stuff_mat = sio.loadmat('../sceneparsing/objectSplit35-115.mat')
        objects_stuff_list = objects_stuff_mat['stuffOrobject']
        object_list = []
        for object_id in range(len(objects_stuff_list)):
            if objects_stuff_list[object_id] == 2:
                object_list.append(object_id)
        # print(len(object_list))
        # other as background
        save_label = np.zeros(label_np.shape) 
        # save_instance = np.zeros(ins_np.shape) 

        label_onehot = np.zeros((1,151,) + label_np.shape) 
        label_onehot_tensor = torch.from_numpy(label_onehot).float()
        # print(label_onehot_tensor.size())

        # save_label = np.zeros(label_np.shape) + 34
        # save_instance = np.zeros(ins_np.shape) + 34

        label_seq = np.unique(label_np)
        target_folder = 'ade20k_box'

        obj_count = 0
        for label_id in label_seq:
            for obj in object_list:
                # if str(obj) in str(label_id):
                if label_id == obj:
                # if str(label_id).startswith(str(obj)):
                    temp_label = np.zeros(label_np.shape)
                    temp_label[label_np==label_id] = label_id
                    # ret, thresh = cv2.threshold(save_label, 125, 255, 0)
                    # thresh = thresh.astype('uint8')
                    temp_label = temp_label.astype('uint8')
                    contours, hierarchy = cv2.findContours(temp_label,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    for conid in range(len(contours)):
                        mask = np.zeros(label_np.shape, dtype="uint8") * 255
                        cv2.drawContours(mask, contours, conid, int(label_id), -1)
                        i, j = np.where(mask==label_id)
                        indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                                              np.arange(min(j), max(j) + 1),
                                              indexing='ij')
                        x1 = min(i)
                        y1 = min(j)
                        x2 = max(i)
                        y2 = max(j)
                        if ((x2-x1) * (y2-y1)) < 30:
                            continue
                        # save_label = np.zeros(label_np.shape)
                        save_label[indices] = obj
                        # save_label[label_np==label_id] = obj
                        save_label_batch = save_label.reshape((1,1,)+save_label.shape)
                        save_label_tensor = torch.from_numpy(save_label_batch).long()
                        label_onehot_tensor.scatter_(1, save_label_tensor, 1.0)

                        obj_count += 1
            # if obj_count >= 16:
            #     break
        label_onehot_tensor = label_onehot_tensor.squeeze(0)

        return label_onehot_tensor, save_label, object_list
    def get_onehot_box_tensor_flip(self, fine_label_tensor):
        label_np = fine_label_tensor.data.cpu().numpy()
        label_np = label_np-1
        # should be label[label == -1] = self.opt.label_nc unlabeled is 150
        # then the label_onehot = np.zeros((1,151,) + label_np.shape) + 150 --> 
        # ins_np = instance_tensor.data.cpu().numpy()[0]
        objects_stuff_mat = sio.loadmat('../sceneparsing/objectSplit35-115.mat')
        objects_stuff_list = objects_stuff_mat['stuffOrobject']
        object_list = []
        for object_id in range(len(objects_stuff_list)):
            if objects_stuff_list[object_id] == 2:
                object_list.append(object_id)
        # print(len(object_list))
        # other as background
        save_label = np.zeros(label_np.shape) 
        # save_instance = np.zeros(ins_np.shape) 

        label_onehot = np.zeros((1,151,) + label_np.shape) 
        label_onehot_tensor = torch.from_numpy(label_onehot).float()
        # print(label_onehot_tensor.size())

        # save_label = np.zeros(label_np.shape) + 34
        # save_instance = np.zeros(ins_np.shape) + 34
        label_np_flip = np.flip(label_np, 1)
        label_seq = np.unique(label_np)
        target_folder = 'ade20k_box'

        obj_count = 0
        for label_id in label_seq:
            for obj in object_list:
                # if str(obj) in str(label_id):
                # if label_id == obj:
                if str(label_id).startswith(str(obj)):
                    temp_label = np.zeros(label_np.shape)
                    if label_id == 8:

                        temp_label[label_np_flip==label_id] = label_id
                    else:
                        temp_label[label_np==label_id] = label_id
                    
                    # temp_label[label_np==label_id] = label_id
                    # ret, thresh = cv2.threshold(save_label, 125, 255, 0)
                    # thresh = thresh.astype('uint8')
                    temp_label = temp_label.astype('uint8')
                    contours, hierarchy = cv2.findContours(temp_label,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                    for conid in range(len(contours)):
                        mask = np.zeros(label_np.shape, dtype="uint8") * 255
                        cv2.drawContours(mask, contours, conid, int(label_id), -1)
                        i, j = np.where(mask==label_id)
                        indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                                              np.arange(min(j), max(j) + 1),
                                              indexing='ij')
                        x1 = min(i)
                        y1 = min(j)
                        x2 = max(i)
                        y2 = max(j)
                        if ((x2-x1) * (y2-y1)) < 30:
                            continue
                        # save_label = np.zeros(label_np.shape)
                        save_label[indices] = obj
                        # save_label[label_np==label_id] = obj
                        save_label_batch = save_label.reshape((1,1,)+save_label.shape)
                        save_label_tensor = torch.from_numpy(save_label_batch).long()
                        label_onehot_tensor.scatter_(1, save_label_tensor, 1.0)

                        obj_count += 1
            # if obj_count >= 16:
            #     break
        label_onehot_tensor = label_onehot_tensor.squeeze(0)

        return label_onehot_tensor, save_label, object_list
    # get item for training for ade20k
    def __getitem__(self, index):
        #load json for cityscape

        # Label Image
        if self.opt.box:
            label_path = self.label_paths[index]
            label = Image.open(label_path)
            params = get_params(self.opt, label.size)
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            fine_label_tensor = transform_label(label) * 255.0
            fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1),fine_label_tensor.size(2))
            

            # box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
            # box_path = label_path.replace('.png','_boxes.pkl')
            # label_onehot_bit = pickle.load(open(box_path,'rb'))
            # label_onehot_np = np.asarray(np.zeros((151,256,256)),dtype=np.uint8)

            # label_onehot_byte = np.unpackbits(label_onehot_bit, axis=None)[:label_onehot_np.size].reshape(label_onehot_np.shape).astype(np.uint8)

            label_np, _, _ = self.get_onehot_box_tensor(fine_label_tensor)

            # test_temp = (label_np[0] * 255).data.numpy()
            # PIL.Image.fromarray(test_temp.astype(np.uint8), 'L').save('test_back1.png')

            label_tensor = label_np.float()
        elif self.opt.retrival:
            label_path = self.label_paths[index]

            
            ious_pkl_path = label_path.replace('annotations','retrival').replace('.png','.pkl')
            if os.path.exists(ious_pkl_path):
                ious = pickle.load(open(ious_pkl_path,'rb'))
                image_iou_tuple = sorted(ious.items(), key=lambda d: d[1], reverse=True)[0]
                ret_img = image_iou_tuple[0].split('/')[-1][:-4]
                for label_path_temp in self.label_paths_train:
                    if ret_img in label_path_temp:
                        retrival_label_path = label_path_temp
                for img_path_temp in self.image_paths_train:
                    if ret_img in img_path_temp:
                        retrival_img_path = img_path_temp
                assert self.paths_match(retrival_label_path, retrival_img_path), \
                "The label_path %s and image_path %s don't match." % \
                (retrival_label_path, retrival_img_path)
                label_retrival = Image.open(retrival_label_path)

                label = Image.open(label_path)
                params = get_params(self.opt, label.size)
                transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                fine_label_tensor = transform_label(label) * 255.0

                fine_label_retrival_tensor = transform_label(label_retrival) * 255.0
                fine_label_retrival_tensor = fine_label_retrival_tensor - 1
                fine_label_retrival_tensor[fine_label_retrival_tensor == -1] = self.opt.label_nc

                image_retrival = Image.open(retrival_img_path)
                image_retrival = image_retrival.convert('RGB')
                transform_image = get_transform(self.opt, params)
                image_retrival_tensor = transform_image(image_retrival)

                label_onehot = np.zeros((1,151,fine_label_retrival_tensor.size(1),fine_label_retrival_tensor.size(2))) 
                label_back_tensor = torch.from_numpy(label_onehot).float()
                fine_label_retrival_tensor = fine_label_retrival_tensor.unsqueeze(0).long()
                label_back_tensor.scatter_(1, fine_label_retrival_tensor, 1.0)
                label_back_tensor = label_back_tensor.squeeze(0)

                fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1),fine_label_tensor.size(2))

            else:
                label = Image.open(label_path)
                params = get_params(self.opt, label.size)
                transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                fine_label_tensor = transform_label(label) * 255.0

                label_onehot = np.zeros((1,151,fine_label_tensor.size(1),fine_label_tensor.size(2))) 
                label_back_tensor = torch.from_numpy(label_onehot).float()
                label_back_tensor = label_back_tensor.squeeze(0)

                fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1),fine_label_tensor.size(2))

                image_path = self.image_paths[index]
                image_retrival = Image.open(image_path)
                image_retrival = image_retrival.convert('RGB')
                transform_image = get_transform(self.opt, params)
                image_retrival_tensor = transform_image(image_retrival)


            # box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
            # box_path = label_path.replace('.png','_boxes.pkl')
            # label_onehot_bit = pickle.load(open(box_path,'rb'))
            # label_onehot_np = np.asarray(np.zeros((151,256,256)),dtype=np.uint8)

            # label_onehot_byte = np.unpackbits(label_onehot_bit, axis=None)[:label_onehot_np.size].reshape(label_onehot_np.shape).astype(np.uint8)



            
            # params = get_params(self.opt, label.size)
            # transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            # for rebuttal filp
            label_np, label_foreground, foregrounds = self.get_onehot_box_tensor(fine_label_tensor)
            label_tensor = label_np.float()

            label_foreground[label_foreground!=0] = 1

            label_foreground = torch.from_numpy(label_foreground)
            label_back_tensor[:,label_foreground==1] = 0
            label_back_tensor[foregrounds] = label_tensor[foregrounds]

            label_tensor = label_back_tensor
            # label_save = label_foreground * 255
            # # label_save = fine_label_tensor.data.numpy()
            # label_save = label_save.astype(np.uint8)
            # # label_save = label_save.reshape(label_save.shape+(1,))
            # PIL.Image.fromarray(label_save, 'L').save('test_back1.png')

            # label_inmem = label_np.data.numpy().astype(np.uint8)
            # print(np.array_equal(label_inmem, label_onehot_byte))
        elif self.opt.retrival_memory:
            label_path = self.label_paths[index]

            
            ious_pkl_path = label_path.replace('annotations','retrival_halfset').replace('.png','.pkl')
            if os.path.exists(ious_pkl_path):
                label = Image.open(label_path)
                params = get_params(self.opt, label.size)
                transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                fine_label_tensor = transform_label(label) * 255.0
                fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1),fine_label_tensor.size(2))


                ious = pickle.load(open(ious_pkl_path,'rb'))
                image_iou_tuples = sorted(ious.items(), key=lambda d: d[1], reverse=True)[:6]
                retrival_label_list = []
                for it, query_image in enumerate(image_iou_tuples):
                    ret_img = image_iou_tuples[it][0].split('/')[-1][:-4]
                    for label_path_temp in self.label_paths_train:
                        if ret_img in label_path_temp:
                            retrival_label_path = label_path_temp
                    for img_path_temp in self.image_paths_train:
                        if ret_img in img_path_temp:
                            retrival_img_path = img_path_temp
                    assert self.paths_match(retrival_label_path, retrival_img_path), \
                    "The label_path %s and image_path %s don't match." % \
                    (retrival_label_path, retrival_img_path)
                    label_retrival = Image.open(retrival_label_path)

                    fine_label_retrival_tensor = transform_label(label_retrival) * 255.0
                    fine_label_retrival_tensor = fine_label_retrival_tensor - 1
                    fine_label_retrival_tensor[fine_label_retrival_tensor == -1] = self.opt.label_nc

                    image_retrival = Image.open(retrival_img_path)
                    image_retrival = image_retrival.convert('RGB')
                    transform_image = get_transform(self.opt, params)
                    image_retrival_tensor = transform_image(image_retrival)

                    label_onehot = np.zeros((1,151,fine_label_retrival_tensor.size(1),fine_label_retrival_tensor.size(2))) 
                    label_back_tensor = torch.from_numpy(label_onehot).float()
                    fine_label_retrival_tensor = fine_label_retrival_tensor.unsqueeze(0).long()
                    label_back_tensor.scatter_(1, fine_label_retrival_tensor, 1.0)
                    label_back_tensor = label_back_tensor.squeeze(0)

                    retrival_label_list.append(label_back_tensor)
                    if len(retrival_label_list) == 3:
                        break
            else:
                label = Image.open(label_path)
                params = get_params(self.opt, label.size)
                transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                fine_label_tensor = transform_label(label) * 255.0
                fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1),fine_label_tensor.size(2))


                retrival_label_list = []
                for it in range(6):

                    label_onehot = np.zeros((1,151,fine_label_tensor.size(0),fine_label_tensor.size(1))) 
                    label_back_tensor = torch.from_numpy(label_onehot).float()
                    label_back_tensor = label_back_tensor.squeeze(0)
                    retrival_label_list.append(label_back_tensor)

                    image_path = self.image_paths[index]
                    image_retrival = Image.open(image_path)
                    image_retrival = image_retrival.convert('RGB')
                    transform_image = get_transform(self.opt, params)
                    image_retrival_tensor = transform_image(image_retrival)

                    if len(retrival_label_list) == 3:
                        break


            # box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
            # box_path = label_path.replace('.png','_boxes.pkl')
            # label_onehot_bit = pickle.load(open(box_path,'rb'))
            # label_onehot_np = np.asarray(np.zeros((151,256,256)),dtype=np.uint8)

            # label_onehot_byte = np.unpackbits(label_onehot_bit, axis=None)[:label_onehot_np.size].reshape(label_onehot_np.shape).astype(np.uint8)



            
            # params = get_params(self.opt, label.size)
            # transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)

            label_np, label_foreground, foregrounds = self.get_onehot_box_tensor(fine_label_tensor)
            label_tensor = label_np.float()

            label_foreground[label_foreground!=0] = 1

            label_foreground = torch.from_numpy(label_foreground)


            retrival_labellist = []
            for label_back_tensor in retrival_label_list:

                label_back_tensor[:,label_foreground==1] = 0
                label_back_tensor[foregrounds] = label_tensor[foregrounds]
                retrival_labellist.append(label_back_tensor)

                # label_tensor = label_back_tensor
            retrival_labellist_tensor = torch.cat(retrival_labellist,dim=0)
            # label_save = label_foreground * 255
            # # label_save = fine_label_tensor.data.numpy()
            # label_save = label_save.astype(np.uint8)
            # # label_save = label_save.reshape(label_save.shape+(1,))
            # PIL.Image.fromarray(label_save, 'L').save('test_back1.png')

            # label_inmem = label_np.data.numpy().astype(np.uint8)
            # print(np.array_equal(label_inmem, label_onehot_byte))

        elif self.opt.box_unpair:
            label_path = self.label_paths[index]
            label_path_fine = self.label_paths_fine[index]
            label = Image.open(label_path_fine)
            params = get_params(self.opt, label.size)
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            fine_label_tensor = transform_label(label) * 255.0
            fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1),fine_label_tensor.size(2))

            # box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
            box_path = label_path.replace('.png','_boxes.pkl')
            label_np = pickle.load(open(box_path,'rb'))
            label_tensor = torch.from_numpy(label_np).float()

            # input image (real images fine)
            image_path_fine = self.image_paths_fine[index]
            assert self.paths_match(label_path_fine, image_path_fine), \
                "The label_path %s and image_path %s don't match." % \
                (label_path_fine, image_path_fine)
            image_fine = Image.open(image_path_fine)
            image_fine = image_fine.convert('RGB')

            transform_image = get_transform(self.opt, params)
            image_tensor_fine = transform_image(image_fine)


        else:
            label_path = self.label_paths[index]
            label = Image.open(label_path)
            params = get_params(self.opt, label.size)
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            label_tensor = transform_label(label) * 255.0
            fine_label_tensor = label_tensor

        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)
        if self.opt.box_unpair:
            input_dict = {'label': label_tensor,
                          'instance': instance_tensor,
                          'image': image_tensor,
                          'path': image_path,
                          'fine_label': fine_label_tensor,
                          'image_fine':image_tensor_fine,
                          }
        elif self.opt.retrival:
            input_dict = {'label': label_tensor,
                          'instance': instance_tensor,
                          'image': image_tensor,
                          'path': image_path,
                          'fine_label': fine_label_tensor,
                          'image_retrival': image_retrival_tensor,
                          }
        elif self.opt.retrival_memory:
            input_dict = {'label': label_tensor,
                          'instance': instance_tensor,
                          'image': image_tensor,
                          'path': image_path,
                          'fine_label': fine_label_tensor,
                          'retrival_label_list': retrival_labellist_tensor,
                          # 'image_retrival': image_retrival_tensor,
                          }

        else:
            input_dict = {'label': label_tensor,
                          'instance': instance_tensor,
                          'image': image_tensor,
                          'path': image_path,
                          'fine_label': fine_label_tensor,
                          }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc
