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
import random

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

root = '../../datasets/cityscapes_box'
phase = 'train'
label_dir = os.path.join(root, 'gtFine', phase)

label_paths_all = make_dataset(label_dir, recursive=True)
label_paths = [p for p in label_paths_all if p.endswith('_labelIds.png')]
natural_sort(label_paths)


phase = 'val'
label_dir_val = os.path.join(root, 'gtFine', phase)

label_paths_all_val = make_dataset(label_dir_val, recursive=True)
label_paths_val = [p for p in label_paths_all_val if p.endswith('_labelIds.png')]
natural_sort(label_paths_val)

transform_list = []

def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)
w = 512
h = round(512 / 2.0)
method=Image.BICUBIC
# comment for debug cityscapes and generate training data for seq_spade
transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))
transform_list += [transforms.ToTensor()]

transforms_func = transforms.Compose(transform_list)

# label_mem = {}
# for lable_img_path in label_paths:
#       label = Image.open(lable_img_path)
#       label = transforms_func(label)
#       # print(label.size())
#       label_np = label[0].data.numpy() * 255
#       label_np = label_np.astype(np.uint8)
#       label_mem[lable_img_path] = label_np
#       # label_mem.append(label_np)
# pickle.dump(label_mem, open('label_mem.pkl','wb'),-1)


label_mem = pickle.load(open('label_mem.pkl','rb'))
np.random.seed(1121)
random.seed(1121) 
indices = np.random.permutation(len(label_mem))

indices = indices[0:(int(len(label_mem)/4))]
allkeys = list(label_mem.keys())
half_keys = random.sample(allkeys, int(len(label_mem)/4))
# print(half_keys[0:10])

label_mem = {k:label_mem[k] for k in half_keys}
print(len(label_mem))




foregrounds = [24,25,26,27,28,29,30,31,32,33]

def load_and_save_all_pkl():

    # get all paired image and backgrounds:
    image_pair_dict_all = {}
    for a,b,c in os.walk('./'):
        for item in c:
            if '.pkl' in item and 'val1-4set' in item:
            # if ('.pkl' in item) and ('_val' in item):
                print(item)
                image_pair_dict = pickle.load(open(item,'rb'))
                for image in image_pair_dict:
                    image_pair_dict_all[image] = image_pair_dict[image]
    print(len(image_pair_dict_all))
    image_pair_dict_all = pickle.dump(image_pair_dict_all,open('retrival_img_pairs_1-4set_val_all.pkl','wb'),-1)

def visulize_resuls():
    image_pair_dict_all = pickle.load(open('retrival_img_pairs_train_all.pkl','rb'))
    # print(image_pair_dict_all)
    # get colored image
    for img in image_pair_dict_all:
        ious = image_pair_dict_all[img] 
        # print(ious)
        re_image_pairs = sorted(ious.items(), key=lambda d: d[1], reverse=True)[:6]
        # print(img)
        # print(re_image_pairs)

        # print(re_image_pairs[0][0].split('/')[-1])
        # print(img)
        # if re_image_pairs[0][0].split('/')[-1] != img:
        #     print('bug in retrieval')
        for it, re_image in enumerate(re_image_pairs):
            re_image_name = re_image[0].split('/')[-1]
            if re_image_name in img:
                continue

            countimg = 0
            query = 0
            retrival_img = 0
            for image_all_path in label_paths:
                if img in image_all_path:
                    query = image_all_path
                    countimg += 1
            if countimg !=1:
                print('hehe1')
            count_value = 0
            for image_all_path in label_paths:
                if re_image_name in image_all_path:
                    retrival_img = image_all_path
                    count_value += 1
            if count_value !=1:
                print('hehe2')
            query_colored = query.replace('_labelIds','_color')
            retrival_colored = retrival_img.replace('_labelIds','_color')
            img_name = query_colored.split('/')[-1]
            commandline = 'cp '+ retrival_colored + ' ./retrival_colored_train/'+img_name[:-4]+'_'+str(it)+'_pair.png'
            commandline2 = 'cp '+ query_colored + ' ./retrival_colored_train/'
            # print(commandline)
            # print(commandline2)
            # os.system(commandline)
            # os.system(commandline2)


# # retrival accourding to segmentation of images in memory
def retirval_seg():
    print(len(label_paths))
    processid = int(sys.argv[1])
    print('processid: '+ str(processid))
    save_image_pair = 'retrival_img_pairs_val1-4set'+str(processid)+'.pkl'
    print(save_image_pair)
    interval = 50
    image_pair_dict = {}
    no_foreground_list = []
    for i,label_img_path in enumerate(label_paths_val):
        if i in range(processid*interval,(processid+1)*interval):
            print('imageid: ',i)
            label_img = Image.open(label_img_path)
            label_img = transforms_func(label_img)
            # print(label.size())
            label_load_tensor = label_img[0] * 255
            label_ids_include = (label_load_tensor).unique()
            all_back_check = False
            for label_id in label_ids_include:
                label_id = int(label_id.item())
                if label_id in foregrounds:
                    all_back_check = True
            if not all_back_check:
                no_foreground_list.append(label_img_path)
                continue
            if i % 100 == 0:
                print(no_foreground_list)
            box_path = label_img_path.replace('_gtFine_labelIds.png','_gtFine_boxes.pkl')
            label_load = pickle.load(open(box_path,'rb'))
            ious = {}
            start_time = time.time()
            for seg_path in label_mem:
                if seg_path.split('/')[-1] not in label_img_path:
                    intersection = 0
                    union = 0
                    for label_id in foregrounds:
                        if label_id not in label_ids_include:
                            continue
                        label_seg = np.zeros(label_load.shape[1:])
                        label_seg[label_mem[seg_path]==label_id] = 1 

                        # label_seg_query = np.zeros(label_load.shape[1:])
                        # label_seg_query[label_load_tensor==label_id] = 1 
                        true = label_load[label_id].sum()
                        # true = label_seg_query.sum()
                        pred = label_seg.sum() 
                        intersection_temp = np.sum(label_seg * label_load[label_id])
                        # intersection_temp = np.sum(label_seg * label_seg_query)
                        union_temp =  true + pred - intersection_temp
                        intersection += intersection_temp
                        union += union_temp
                    ious[seg_path] = intersection/union

            # print(ious)
            print(label_img_path)
            sys.stdout.flush()
            img_name = label_img_path.split('/')[-1]
            re_image_name = sorted(ious.items(), key=lambda d: d[1], reverse=True)[0][0].split('/')[-1]
            print(re_image_name)
            sys.stdout.flush()
            commandline = 'cp '+ sorted(ious.items(), key=lambda d: d[1], reverse=True)[0][0] + ' ./retrival_results/'+img_name[:-4]+'_pair.png'
            commandline2 = 'cp '+ label_img_path + ' ./retrival_results/'
            # print(ious)
            # image_pair_dict[img_name] = re_image_name
            image_pair_dict[img_name] = ious
            # os.system(commandline)
            # os.system(commandline2)
            print(time.time()- start_time)
            sys.stdout.flush()
      # print(ious)
    pickle.dump(image_pair_dict, open(save_image_pair,'wb'),-1)
# # retrival accourding to bounding box of images in memory

      # print(label_mem[0].shape)
# visulize_resuls()
# retirval_seg()
load_and_save_all_pkl()




