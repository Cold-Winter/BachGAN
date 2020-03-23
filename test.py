"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
# from models.pix2pix_model import Pix2PixModel


from util.visualizer import Visualizer
from util import html
import torch

opt = TestOptions().parse()

if opt.dual:
    from models.pix2pix_dualmodel import Pix2PixModel
elif opt.dual_segspade:
    from models.pix2pix_dual_segspademodel import Pix2PixModel
elif opt.box_unpair:
    from models.pix2pix_dualunpair import Pix2PixModel
else:
    from models.pix2pix_model import Pix2PixModel

dataloader = data.create_dataloader(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
# for i, data_i in enumerate(dataloader):
#     if i * opt.batchSize >= opt.how_many:
#         break
#     # print(data_i)
#     generated = model(data_i, mode='inference')

#     img_path = data_i['path']
#     for b in range(generated.shape[0]):
#         print('process image... %s' % img_path[b])
#         print(data_i['label'][b])
#         insset = set([])
#         data_np = data_i['label'][b].data.cpu().numpy()[0]
#         print(data_np.shape)
#         for row in range(255):
#             for column in range(255):
#                 insset.add(data_np[row,column])
#         print(insset)
#         print((data_i['label'][b] == 171).float()*data_i['label'][b])

#         print((data_i['label'][b] == 171).float()*data_i['label'][b].float())
#         #for ins in insset:
#         #    print(ins)
#         visuals = OrderedDict([('input_label', data_i['label'][b].float()*((data_i['label'][b]==182).float())),
#                                ('synthesized_image', generated[b])])
#         visualizer.save_images(webpage, visuals, img_path[b:b + 1])
#     if i == 1:
#         break

for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    img_path = data_i['path']
    # print(img_path)

    generated = model(data_i, mode='inference')

    for b in range(generated.shape[0]):
        print('process image... %s' % img_path[b])
        visuals = OrderedDict([('input_label', data_i['label'][b][0:35]),
                               ('synthesized_image', generated[b])])
        if opt.retrival_memory:
            visuals = OrderedDict([('input_label', data_i['retrival_label_list'][b][0:35]),
                               ('synthesized_image', generated[b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])
webpage.save()

# for i, data_i in enumerate(dataloader):
#     if i * opt.batchSize >= opt.how_many:
#         break

#     img_path = data_i['path']
#     # print(img_path)

# # frankfurt_000000_002963_leftImg8bit
#     # if '23769' in img_path[0]:
#     # if '3357' in img_path[0]:
#     if 'ADE_val_00000124' in img_path[0]:
#         print(img_path)
#         generated = model(data_i, mode='inference')

#         for b in range(generated.shape[0]):
#             print('process image... %s' % img_path[b])
#             visuals = OrderedDict([('input_label', data_i['label'][b][0:35]),
#                                    ('synthesized_image', generated[b])])
#             if opt.retrival_memory:
#                 visuals = OrderedDict([('input_label', data_i['retrival_label_list'][b][0:35]),
#                                    ('synthesized_image', generated[b])])
#             visualizer.save_images(webpage, visuals, img_path[b:b + 1])
#     else:
#         continue

webpage.save()

