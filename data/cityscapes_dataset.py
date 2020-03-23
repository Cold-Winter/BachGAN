"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os.path
from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import PIL.Image
import pickle
import numpy as np
import copy

import torch
import cv2

from skimage import measure

from random import shuffle

import scipy.io as sio


class CityscapesDataset(Pix2pixDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(load_size=512)
        parser.set_defaults(crop_size=512)
        parser.set_defaults(display_winsize=512)
        #need to change to 36 when we use background as another class
        parser.set_defaults(label_nc=35)
        parser.set_defaults(aspect_ratio=2.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'


        label_train_dir = os.path.join(root, 'gtFine', 'train')
        label_paths_all_train = make_dataset(label_train_dir, recursive=True)
        label_paths_train = [p for p in label_paths_all_train if p.endswith('_labelIds.png')]
        
        image_dir_train = os.path.join(root, 'leftImg8bit', 'train')
        image_paths_all_train = make_dataset(image_dir_train, recursive=True)
        image_paths_train = [p for p in image_paths_all_train if p.endswith('.png')]




        label_dir = os.path.join(root, 'gtFine', phase)
        # gtCoarse
        # label_dir = os.path.join(root, 'gtCoarse', phase)
        label_paths_all = make_dataset(label_dir, recursive=True)
        # label_paths = [p for p in label_paths_all if p.endswith('_labelIds.png')]
        # foreground
        label_paths = [p for p in label_paths_all if p.endswith('_labelIds.png')]

        image_dir = os.path.join(root, 'leftImg8bit', phase)
        # image_paths = make_dataset(image_dir, recursive=True)

        image_paths_all = make_dataset(image_dir, recursive=True)
        image_paths = [p for p in image_paths_all if p.endswith('.png')]


        if not opt.no_instance:
            # instance_paths = [p for p in label_paths_all if p.endswith('_instanceIds.png')]
            # foreground
            instance_paths = [p for p in label_paths_all if p.endswith('_instanceIds.png')]
        else:
            instance_paths = [p for p in label_paths_all if p.endswith('_instanceIds.png')]

            # instance_paths = []

        return label_paths, image_paths, instance_paths, label_paths_train, image_paths_train
    def get_onehot_box_tensor(self, ins_label_tensor):
        ins_np = ins_label_tensor.data.cpu().numpy()[0]
        save_label = np.zeros(ins_np.shape) 
        label_onehot = np.zeros((1,35,) + ins_np.shape) 
        label_onehot_tensor = torch.from_numpy(label_onehot).float()
        ins_seq = np.unique(ins_np)
        foregrounds = [24,25,26,27,28,29,30,31,32,33]
        # foregrounds = [26]
        obj_count = 0
        for label_id in ins_seq:
            for obj in foregrounds:
                # if str(obj) in str(label_id):
                if str(label_id).startswith(str(obj)):
                    i, j = np.where(ins_np==label_id)
                    indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                                          np.arange(min(j), max(j) + 1),
                                          indexing='ij')
                    save_label[indices] = obj
                    # save_label[ins_np==label_id] = obj
                    save_label_batch = save_label.reshape((1,1,)+save_label.shape)
                    save_label_tensor = torch.from_numpy(save_label_batch).long()
                    label_onehot_tensor.scatter_(1, save_label_tensor, 1.0)
                    obj_count += 1
            # if obj_count >= 7:
            #     break

        label_onehot_tensor = label_onehot_tensor.squeeze(0)

        return label_onehot_tensor, save_label


    def get_onehot_box_tensor_flip(self, ins_label_tensor):
        ins_np = ins_label_tensor.data.cpu().numpy()[0]
        ins_np_flip = np.flip(ins_np ,1)
        save_label = np.zeros(ins_np.shape) 
        label_onehot = np.zeros((1,35,) + ins_np.shape) 
        label_onehot_tensor = torch.from_numpy(label_onehot).float()
        ins_seq = np.unique(ins_np)
        foregrounds = [24,25,26,27,28,29,30,31,32,33]
        # foregrounds = [26]
        obj_count = 0
        for label_id in ins_seq:
            for obj in foregrounds:
                # if str(obj) in str(label_id):
                if str(label_id).startswith(str(obj)):
                    if label_id == 26015 or label_id == 33000: #or  label_id == 26009
                    # if label_id == 26008 or label_id == 26010 or label_id == 26011:
                        i, j = np.where(ins_np_flip==label_id)
                        indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                                              np.arange(min(j), max(j) + 1),
                                              indexing='ij')
                        save_label[indices] = obj
                        # save_label[ins_np==label_id] = obj
                        save_label_batch = save_label.reshape((1,1,)+save_label.shape)
                        save_label_tensor = torch.from_numpy(save_label_batch).long()
                        label_onehot_tensor.scatter_(1, save_label_tensor, 1.0)
                        obj_count += 1
                    else:
                        i, j = np.where(ins_np==label_id)
                        indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                                              np.arange(min(j), max(j) + 1),
                                              indexing='ij')
                        save_label[indices] = obj
                        # save_label[ins_np==label_id] = obj
                        save_label_batch = save_label.reshape((1,1,)+save_label.shape)
                        save_label_tensor = torch.from_numpy(save_label_batch).long()
                        label_onehot_tensor.scatter_(1, save_label_tensor, 1.0)
                        obj_count += 1

            # if obj_count >= 7:
            #     break

        label_onehot_tensor = label_onehot_tensor.squeeze(0)

        return label_onehot_tensor, save_label

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

            box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
            label_np = pickle.load(open(box_path,'rb'))
            label_tensor = torch.from_numpy(label_np).float()
            # check save correct
            # box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
            # label_load = pickle.load(open(box_path,'rb'))
            # print(np.array_equal(label_np.data.numpy(),label_load))
            label_tensor = label_np.float()

        elif self.opt.retrival:
            label_path = self.label_paths[index]

            if self.opt.phase == 'test':
                image_pair_dict_all = pickle.load(open('./scripts/retrival_ious_city_halfset/retrival_img_pairs_halfset_val_all.pkl','rb'))
            elif self.opt.phase == 'train':
                image_pair_dict_all = pickle.load(open('./scripts/retrival_ious_city_halfset/retrival_img_pairs_halfset_train_all.pkl','rb'))
            
            # for img in image_pair_dict_all:
            #     if img in label_path:
            #         retrival_label_name = image_pair_dict_all[img]
            # for label_path_temp in self.label_paths_train:
            #     if retrival_label_name in label_path_temp:
            #         retrival_label_path = label_path_temp
            # retrival_img_name = retrival_label_path.split('/')[-1].split('_gtFine')[0]
            # for img_path_temp in self.image_paths_train:
            #     if retrival_img_name in img_path_temp:
            #         retrival_img_path = img_path_temp

            # assert self.paths_match(retrival_label_path, retrival_img_path), \
            #     "The label_path %s and image_path %s don't match." % \
            #     (retrival_label_path, retrival_img_path)

            # label_retrival = Image.open(retrival_label_path)
            # params = get_params(self.opt, label_retrival.size)
            # transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            # fine_label_retrival_tensor = transform_label(label_retrival) * 255.0

            # image_retrival = Image.open(retrival_img_path)
            # image_retrival = image_retrival.convert('RGB')
            # transform_image = get_transform(self.opt, params)
            # image_retrival_tensor = transform_image(image_retrival)

            # label_onehot = np.zeros((1,35,fine_label_retrival_tensor.size(1),fine_label_retrival_tensor.size(2))) 
            # label_back_tensor = torch.from_numpy(label_onehot).float()
            # fine_label_retrival_tensor = fine_label_retrival_tensor.unsqueeze(0).long()
            # label_back_tensor.scatter_(1, fine_label_retrival_tensor, 1.0)
            # label_back_tensor = label_back_tensor.squeeze(0)

            # label = Image.open(label_path)
            # # params = get_params(self.opt, label.size)
            
            # # transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            # fine_label_tensor = transform_label(label) * 255.0
            # fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1),fine_label_tensor.size(2))

            # instance_path = self.instance_paths[index]
            # instance = Image.open(instance_path)
            # if instance.mode == 'L':
            #     instance_tensor = transform_label(instance) * 255
            #     instance_tensor = instance_tensor.long()
            # else:
            #     instance_tensor = transform_label(instance)

            # label_np, label_foreground = self.get_onehot_box_tensor(instance_tensor)
            # label_tensor = label_np.float()

            # label_foreground[label_foreground!=0] = 1
            # # label_save = label_foreground * 255
            # # # label_save = fine_label_tensor.data.numpy()
            # # label_save = label_save.astype(np.uint8)
            # # # label_save = label_save.reshape(label_save.shape+(1,))
            # # PIL.Image.fromarray(label_save, 'L').save('test_back.png')
            # foregrounds = [24,25,26,27,28,29,30,31,32,33]
            

            # label_foreground = torch.from_numpy(label_foreground)
            # label_back_tensor[:,label_foreground==1] = 0
            # label_back_tensor[foregrounds] = label_tensor[foregrounds]

            # # test_temp = (label_back_tensor[1] * 255).data.numpy()
            # # PIL.Image.fromarray(test_temp.astype(np.uint8), 'L').save('test_back1.png')
            # label_tensor = label_back_tensor


            # for half set
            label = Image.open(label_path)
            params = get_params(self.opt, label.size)
            
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            fine_label_tensor = transform_label(label) * 255.0
            fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1),fine_label_tensor.size(2))

            label_name = label_path.split('/')[-1]
            if label_name in image_pair_dict_all:
                retrival_ious = image_pair_dict_all[label_name]

            else:
                print('no foreground data')
                if self.opt.phase == 'train':
                    retrival_ious = image_pair_dict_all['erfurt_000086_000019_gtFine_labelIds.png']
                elif self.opt.phase == 'test':
                    retrival_ious = image_pair_dict_all['frankfurt_000001_083852_gtFine_labelIds.png']

            query_ious_pairs = sorted(retrival_ious.items(), key=lambda d: d[1], reverse=True)[:6]


            retrival_label_list = []
            for it, query_image in enumerate(query_ious_pairs):
                re_label_name = query_image[0].split('/')[-1]
                if re_label_name in label_name:
                    continue 
                 
                for label_path_temp in self.label_paths_train:
                    if re_label_name in label_path_temp:
                        re_label_path = label_path_temp


                retrival_img_name = re_label_path.split('/')[-1].split('_gtFine')[0]
                for img_path_temp in self.image_paths_train:
                    if retrival_img_name in img_path_temp:
                        retrival_img_path = img_path_temp

                assert self.paths_match(re_label_path, retrival_img_path), \
                    "The label_path %s and image_path %s don't match." % \
                    (re_label_path, retrival_img_path)

                label_retrival = Image.open(re_label_path)

                transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                fine_label_retrival_tensor = transform_label(label_retrival) * 255.0

                image_retrival = Image.open(retrival_img_path)
                image_retrival = image_retrival.convert('RGB')
                transform_image = get_transform(self.opt, params)
                image_retrival_tensor = transform_image(image_retrival)

                label_onehot = np.zeros((1,35,fine_label_retrival_tensor.size(1),fine_label_retrival_tensor.size(2))) 
                label_back_tensor = torch.from_numpy(label_onehot).float()
                fine_label_retrival_tensor = fine_label_retrival_tensor.unsqueeze(0).long()
                label_back_tensor.scatter_(1, fine_label_retrival_tensor, 1.0)
                label_back_tensor = label_back_tensor.squeeze(0)
                retrival_label_list.append(label_back_tensor)
                if len(retrival_label_list) == 1:
                    break


            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

            label_np, label_foreground = self.get_onehot_box_tensor(instance_tensor)
            label_tensor = label_np.float()

            label_foreground[label_foreground!=0] = 1
            # label_save = label_foreground * 255
            # # label_save = fine_label_tensor.data.numpy()
            # label_save = label_save.astype(np.uint8)
            # # label_save = label_save.reshape(label_save.shape+(1,))
            # PIL.Image.fromarray(label_save, 'L').save('test_back.png')
            foregrounds = [24,25,26,27,28,29,30,31,32,33]

            
            # check save correct
            # box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
            # label_load = pickle.load(open(box_path,'rb'))
            # label_load_tensor = torch.from_numpy(label_load).float()

            # label_tensor = torch.cat((label_back_tensor, label_load_tensor), dim=0)
            # print(np.array_equal(label_np.data.numpy(),label_load))

            # print(label_np.shape)
            # label_tensor = torch.from_numpy(label_np).float()
        elif self.opt.retrival_memory:
            label_path = self.label_paths[index]

            if self.opt.phase == 'test':
                image_pair_dict_all = pickle.load(open('./scripts/retrival_ious_city_halfset/retrival_img_pairs_halfset_val_all.pkl','rb'))
            elif self.opt.phase == 'train':
                image_pair_dict_all = pickle.load(open('./scripts/retrival_ious_city_halfset/retrival_img_pairs_halfset_train_all.pkl','rb'))

            # if self.opt.phase == 'test':
            #     image_pair_dict_all = pickle.load(open('./scripts/retrival_ious_city_halfset/retrival_img_pairs_1-4set_val_all.pkl','rb'))
            # elif self.opt.phase == 'train':
            #     image_pair_dict_all = pickle.load(open('./scripts/retrival_ious_city_halfset/retrival_img_pairs_1-4set_train_all.pkl','rb'))

            label = Image.open(label_path)
            params = get_params(self.opt, label.size)
            
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            fine_label_tensor = transform_label(label) * 255.0
            fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1),fine_label_tensor.size(2))

            label_name = label_path.split('/')[-1]

            if label_name in image_pair_dict_all:
                retrival_ious = image_pair_dict_all[label_name]
            else:
                print('no foreground data')
                if self.opt.phase == 'train':
                    retrival_ious = image_pair_dict_all['erfurt_000086_000019_gtFine_labelIds.png']
                elif self.opt.phase == 'test':
                    retrival_ious = image_pair_dict_all['frankfurt_000001_083852_gtFine_labelIds.png']            
            # for img in image_pair_dict_all:
            #     if img in label_path:
            #         retrival_ious = image_pair_dict_all[img]
            query_ious_pairs = sorted(retrival_ious.items(), key=lambda d: d[1], reverse=True)[:6]

            retrival_label_list = []
            for it, query_image in enumerate(query_ious_pairs):
                re_label_name = query_image[0].split('/')[-1]
                if re_label_name in label_name:
                    continue 
                 
                for label_path_temp in self.label_paths_train:
                    if re_label_name in label_path_temp:
                        re_label_path = label_path_temp


                retrival_img_name = re_label_path.split('/')[-1].split('_gtFine')[0]
                for img_path_temp in self.image_paths_train:
                    if retrival_img_name in img_path_temp:
                        retrival_img_path = img_path_temp

                assert self.paths_match(re_label_path, retrival_img_path), \
                    "The label_path %s and image_path %s don't match." % \
                    (re_label_path, retrival_img_path)

                label_retrival = Image.open(re_label_path)

                transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
                fine_label_retrival_tensor = transform_label(label_retrival) * 255.0

                # image_retrival = Image.open(retrival_img_path)
                # image_retrival = image_retrival.convert('RGB')
                # transform_image = get_transform(self.opt, params)
                # image_retrival_tensor = transform_image(image_retrival)

                label_onehot = np.zeros((1,35,fine_label_retrival_tensor.size(1),fine_label_retrival_tensor.size(2))) 
                label_back_tensor = torch.from_numpy(label_onehot).float()
                fine_label_retrival_tensor = fine_label_retrival_tensor.unsqueeze(0).long()
                label_back_tensor.scatter_(1, fine_label_retrival_tensor, 1.0)
                label_back_tensor = label_back_tensor.squeeze(0)
                retrival_label_list.append(label_back_tensor)
                if len(retrival_label_list) == 3:
                    break


            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)
            #for rebuttal flip some objects
            label_np, label_foreground = self.get_onehot_box_tensor(instance_tensor)
            label_tensor = label_np.float()

            label_foreground[label_foreground!=0] = 1
            # label_save = label_foreground * 255
            # # label_save = fine_label_tensor.data.numpy()
            # label_save = label_save.astype(np.uint8)
            # # label_save = label_save.reshape(label_save.shape+(1,))
            # PIL.Image.fromarray(label_save, 'L').save('test_back.png')
            foregrounds = [24,25,26,27,28,29,30,31,32,33]
            

            label_foreground = torch.from_numpy(label_foreground)
            retrival_labellist = []
            for label_back_tensor in retrival_label_list:
                label_back_tensor[:,label_foreground==1] = 0
                label_back_tensor[foregrounds] = label_tensor[foregrounds]
                retrival_labellist.append(label_back_tensor)

            retrival_labellist_tensor = torch.cat(retrival_labellist,dim=0)

            # label_tensor = retrival_labellist_tensor[0]


            # label_tensor = label_back_tensor

            # test_temp = (label_back_tensor[1] * 255).data.numpy()
            # PIL.Image.fromarray(test_temp.astype(np.uint8), 'L').save('test_back1.png')
            
            # check save correct
            # box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
            # label_load = pickle.load(open(box_path,'rb'))
            # label_load_tensor = torch.from_numpy(label_load).float()

            # label_tensor = torch.cat((label_back_tensor, label_load_tensor), dim=0)
            # print(np.array_equal(label_np.data.numpy(),label_load))

            # print(label_np.shape)
            # label_tensor = torch.from_numpy(label_np).float()
        elif self.opt.box_unpair:
            label_path = self.label_paths[index]
            label_path_fine = self.label_paths_fine[index]
            label = Image.open(label_path_fine)
            params = get_params(self.opt, label.size)
            transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            fine_label_tensor = transform_label(label) * 255.0
            fine_label_tensor = fine_label_tensor.view(fine_label_tensor.size(1),fine_label_tensor.size(2))

            # box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
            box_path = label_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
            label_np = pickle.load(open(box_path,'rb'))
            label_tensor = torch.from_numpy(label_np).float()
            # instance_path = self.instance_paths[index]
            # instance = Image.open(instance_path)
            # if instance.mode == 'L':
            #     instance_tensor = transform_label(instance) * 255
            #     instance_tensor = instance_tensor.long()
            # else:
            #     instance_tensor = transform_label(instance)

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
            foregrounds = [24,25,26,27,28,29,30,31,32,33]
            # train foreground
            for obj in foregrounds:
                label_tensor[label_tensor == obj] = 0

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
                # instance_tensor = instance_tensor.long()

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

    def paths_match(self, path1, path2):
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)
        # compare the first 3 components, [city]_[id1]_[id2]
        return '_'.join(name1.split('_')[:3]) == \
            '_'.join(name2.split('_')[:3])

