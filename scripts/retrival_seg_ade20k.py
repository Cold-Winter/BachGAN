            # check each label_id
            # for label_id in range(35):
            #     label_box_class = label_np[label_id].data.numpy() * 255
            #     label_box_class = label_box_class.astype(np.uint8)
            #     if label_box_class.any():
            #         print(label_id)

            # label_save = label_np[24].data.numpy() * 255
            # # label_save = fine_label_tensor.data.numpy()
            # label_save = label_save.astype(np.uint8)
            # # label_save = label_save.reshape(label_save.shape+(1,))
            # PIL.Image.fromarray(label_save, 'L').save('test_back.png')
import os
import pickle
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import time
import sys
import re
import scipy.io as sio
import torch
import cv2

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset_rec(dir, images):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)


def make_dataset(dir, recursive=False, read_cache=False, write_cache=False):
    images = []

    if read_cache:
        possible_filelist = os.path.join(dir, 'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist, 'r') as f:
                images = f.read().splitlines()
                return images

    if recursive:
        make_dataset_rec(dir, images)
    else:
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

        for root, dnames, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    if write_cache:
        filelist_cache = os.path.join(dir, 'files.list')
        with open(filelist_cache, 'w') as f:
            for path in images:
                f.write("%s\n" % path)
            print('wrote filelist cache at %s' % filelist_cache)

    return images

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]

def natural_sort(items):
    items.sort(key=natural_keys)

root = '../datasets/ADEChallengeData2016/'
phase = 'train'

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

natural_sort(image_paths)
natural_sort(label_paths)


phase = 'val'

all_images = make_dataset(root, recursive=True, read_cache=False, write_cache=False)
image_paths_val = []
label_paths_val = []
for p in all_images:
    if '_%s_' % phase not in p:
        continue
    if p.endswith('.jpg'):
        image_paths_val.append(p)
    elif p.endswith('.png'):
        label_paths_val.append(p)

natural_sort(image_paths_val)

natural_sort(label_paths_val)

transform_list = []

def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)
new_h = new_w = 256

# method=Image.BICUBIC
method=Image.NEAREST
# comment for debug cityscapes and generate training data for seq_spade

osize = [new_h, new_w]
transform_list.append(transforms.Resize(osize, interpolation=method))

transform_list += [transforms.ToTensor()]

transforms_func = transforms.Compose(transform_list)



# label_mem = {}
# for lable_img_path in label_paths:
#       label = Image.open(lable_img_path)
#       label = transforms_func(label)
#       # print(label.size())
#       label_np = label[0].data.numpy() * 255
#       label_np = label_np-1
#       label_np = label_np.astype(np.uint8)
#       label_mem[lable_img_path] = label_np
#       # label_mem.append(label_np)
# pickle.dump(label_mem, open('label_mem_ade20k.pkl','wb'),-1)

label_mem = pickle.load(open('label_mem_ade20k.pkl','rb'))
print(image_paths_val)
# foregrounds = [24,25,26,27,28,29,30,31,32,33]

# get all paired image and backgrounds:
# image_pair_dict_all = {}
# for a,b,c in os.walk('./'):
#     for item in c:
#         # if '.pkl' in item and 'retrival' in item:
#         if ('.pkl' in item) and ('_val' in item):
#             print(item)
#             image_pair_dict = pickle.load(open(item,'rb'))
#             for image in image_pair_dict:
#                 image_pair_dict_all[image] = image_pair_dict[image]
# print(len(image_pair_dict_all))
# image_pair_dict_all = pickle.dump(image_pair_dict_all,open('retrival_img_pairs_val_all.pkl','wb'),-1)
def get_color_images():
    # image_pair_dict_all = pickle.load(open('retrival_img_pairs_val_all.pkl','rb'))
    # print(image_pair_dict_all)

    # get colored image
    for a,b,c in os.walk('/datadrive/yandong/SPADE/datasets/ADEChallengeData2016/retrival/validation/'):
        for item in c:
            img = item[:-4]
            print(img)
            ious = pickle.load(open(a+item,'rb'))
            image_iou_tuple = sorted(ious.items(), key=lambda d: d[1], reverse=True)[0]
            ret_img = image_iou_tuple[0].split('/')[-1][:-4]
            countimg = 0
            query = 0
            retrival_img = 0
            for image_all_path in image_paths_val:
                if img in image_all_path:
                    query = image_all_path
                    countimg += 1
            if countimg !=1:
                print('hehe')
            count_value = 0
            for image_all_path in image_paths:
                if ret_img in image_all_path:
                    retrival_img = image_all_path
                    count_value += 1
            if count_value !=1:
                print('hehe')
            query_colored = query.replace('_labelIds','_color')
            retrival_colored = retrival_img.replace('_labelIds','_color')
            img_name = query_colored.split('/')[-1]
            commandline = 'cp '+ retrival_colored + ' ./retrival_rgb_val_ade20k/'+img_name[:-4]+'_pair.jpg'
            commandline2 = 'cp '+ query_colored + ' ./retrival_rgb_val_ade20k/'
            print(commandline)
            print(commandline2)
            os.system(commandline)
            os.system(commandline2)
def get_labels_images():
    # image_pair_dict_all = pickle.load(open('retrival_img_pairs_val_all.pkl','rb'))
    # print(image_pair_dict_all)

    # get colored image
    for a,b,c in os.walk('/datadrive/yandong/SPADE/datasets/ADEChallengeData2016/retrival/validation/'):
        for item in c:
            img = item[:-4]
            print(img)
            ious = pickle.load(open(a+item,'rb'))
            image_iou_tuple = sorted(ious.items(), key=lambda d: d[1], reverse=True)[0]
            ret_img = image_iou_tuple[0].split('/')[-1][:-4]
            countimg = 0
            query = 0
            retrival_img = 0
            for image_all_path in label_paths_val:
                if img in image_all_path:
                    query = image_all_path
                    countimg += 1
            if countimg !=1:
                print('hehe')
            count_value = 0
            for image_all_path in label_paths:
                if ret_img in image_all_path:
                    retrival_img = image_all_path
                    count_value += 1
            if count_value !=1:
                print('hehe')
            query_colored = query.replace('_labelIds','_color')
            retrival_colored = retrival_img.replace('_labelIds','_color')
            img_name = query_colored.split('/')[-1]
            commandline = 'cp '+ retrival_colored + ' ./retrival_label_val_ade20k/'+img_name[:-4]+'_pair.png'
            commandline2 = 'cp '+ query_colored + ' ./retrival_label_val_ade20k/'
            print(commandline)
            print(commandline2)
            os.system(commandline)
            os.system(commandline2)

get_color_images()

def get_onehot_box_tensor(fine_label_tensor):
        label_np = fine_label_tensor.data.cpu().numpy()
        label_np = label_np-1
        # should be label[label == -1] = self.opt.label_nc unlabeled is 150
        # then the label_onehot = np.zeros((1,151,) + label_np.shape) + 150 --> 
        # ins_np = instance_tensor.data.cpu().numpy()[0]
        objects_stuff_mat = sio.loadmat('../../sceneparsing/objectSplit35-115.mat')
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

        for label_id in label_seq:
            for obj in object_list:
                if str(obj) in str(label_id):
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
                        save_label = np.zeros(label_np.shape)
                        save_label[indices] = obj
                        # save_label[label_np==label_id] = obj
                        save_label_batch = save_label.reshape((1,1,)+save_label.shape)
                        save_label_tensor = torch.from_numpy(save_label_batch).long()
                        label_onehot_tensor.scatter_(1, save_label_tensor, 1.0)
        label_onehot_tensor = label_onehot_tensor.squeeze(0)

        return label_onehot_tensor


# retrival accourding to segmentation of images in memory

# objects_stuff_mat = sio.loadmat('../../sceneparsing/objectSplit35-115.mat')
# objects_stuff_list = objects_stuff_mat['stuffOrobject']
# object_list = []
# for object_id in range(len(objects_stuff_list)):
#     if objects_stuff_list[object_id] == 2:
#         object_list.append(object_id)
# processid = int(sys.argv[1])
# print('processid: '+ str(processid))
# save_image_pair = './retrival_pkl_ade20k/retrival_img_pairs_validation'+str(processid)+'.pkl'
# print(save_image_pair)
# interval = 200
# image_pair_dict = {}
# no_foreground_list = []
# for i,label_img_path in enumerate(label_paths_val):
#       if i in range(processid*interval,(processid+1)*interval):
#             print('imageid: ',i)
#             img_name = label_img_path.split('/')[-1]
#             pkl_name = img_name.replace('.png','.pkl')
#             save_image_pkl_path = './retrival_pkl_ade20k/validation/' + pkl_name
#             if os.path.exists(save_image_pkl_path):
#                 continue
            
#             label = Image.open(label_img_path)
#             label = transforms_func(label)
#             # print(label.size())
#             label_load_tensor = label[0] * 255
#             label_ids_include = (label_load_tensor-1).unique()
#             all_back_check = False
#             for label_id in label_ids_include:
#                 label_id = int(label_id.item())
#                 if label_id in object_list:
#                     all_back_check = True
#             if not all_back_check:
#                 no_foreground_list.append(label_img_path)
#                 continue
#             if i % 100 == 0:
#                 print(no_foreground_list)
#             sys.stdout.flush()
#             # label_load = label_load.view(label_load.size(0),label_load.size(1))

#             label_load_onehot_tensor= get_onehot_box_tensor(label_load_tensor)
#             label_load = label_load_onehot_tensor.data.numpy()

#             ious = {}
#             start_time = time.time()
#             for seg_path in label_mem:
#                 if seg_path != label_img_path:
#                     intersection = 0
#                     union = 0
#                     for label_id in label_ids_include:
#                         label_id = int(label_id.item())
#                         if label_id in object_list:
#                             label_seg = np.zeros(label_load.shape[1:])
#                             label_seg[label_mem[seg_path]==label_id] = 1 
#                             true = label_load[label_id].sum()
#                             pred = label_seg.sum()  
#                             intersection_temp = np.sum(label_seg * label_load[label_id])
#                             union_temp =  true + pred - intersection_temp
#                             intersection += intersection_temp
#                             union += union_temp
#                     ious[seg_path] = intersection/union
#             print(label_img_path)
#             sys.stdout.flush()
#             img_name = label_img_path.split('/')[-1]

#             re_image_name = sorted(ious.items(), key=lambda d: d[1], reverse=True)[0][0].split('/')[-1]
#             print(sorted(ious.items(), key=lambda d: d[1], reverse=True)[0])
#             print(re_image_name)
#             sys.stdout.flush()
#             commandline = 'cp '+ sorted(ious.items(), key=lambda d: d[1], reverse=True)[0][0] + ' ./retrival_results/'+img_name[:-4]+'_pair.png'
#             commandline2 = 'cp '+ label_img_path + ' ./retrival_results_ade20k/'
#             image_pair_dict[img_name] = ious
#             # os.system(commandline)
#             # os.system(commandline2)
#             print(time.time()- start_time)
#             sys.stdout.flush()
            
#             pickle.dump(ious, open(save_image_pkl_path,'wb'),-1)
#       # print(ious)
# pickle.dump(image_pair_dict, open(save_image_pair,'wb'),-1)
# retrival accourding to bounding box of images in memory

      # print(label_mem[0].shape)




