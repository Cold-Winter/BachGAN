 def getitem(self, index):
    #load json for cityscape

    # Label Image
    label_path = self.label_paths[index]
    label = Image.open(label_path)
    params = get_params(self.opt, label.size)
    transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0


    label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

    # input image (real images)
    image_path = self.image_paths[index]
    assert self.paths_match(label_path, image_path), \
        "The label_path %s and image_path %s don't match." % \
        (label_path, image_path)
    image = Image.open(image_path)
    image = image.convert('RGB')

    transform_image = get_transform(self.opt, params,normalize=False)
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

    image_np = image_tensor.data.cpu().numpy()
    label_np = label_tensor.data.cpu().numpy()[0]
    ins_np = instance_tensor.data.cpu().numpy()[0]


    # background class is 34
    save_img = np.zeros(image_np.shape)
    # other as background
    save_label = np.zeros(label_np.shape) 
    save_instance = np.zeros(ins_np.shape) 

    # save_label = np.zeros(label_np.shape) + 34
    # save_instance = np.zeros(ins_np.shape) + 34

    label_seq = np.unique(label_np)

    # target_folder = 'cityscapes_seq_test'
    target_folder = 'cityscapes_box'
    seg_len = len(label_seq)/5
    for segid in range(5):
        seg_list = label_seq[int(segid*seg_len):int((segid+1)*seg_len)]
        for lable_id in seg_list:
            save_img[0][label_np==lable_id] = image_np[0][label_np==lable_id]
            save_img[1][label_np==lable_id] = image_np[1][label_np==lable_id]
            save_img[2][label_np==lable_id] = image_np[2][label_np==lable_id]

            save_label[label_np==lable_id] = label_np[label_np==lable_id]
            save_instance[label_np==lable_id] = ins_np[label_np==lable_id]
        # print(np.unique(save_label))
        # print(np.unique(save_instance))

        save_img_path = image_path.replace('cityscapes',target_folder)
        save_label_path = label_path.replace('cityscapes',target_folder)
        save_instance_path = instance_path.replace('cityscapes',target_folder)


        save_img_path = save_img_path.replace('_leftImg8bit','_'+str(segid)+'_leftImg8bit')
        save_label_path = save_label_path.replace('_gtFine_labelIds','_'+str(segid)+'_gtFine_labelIds')
        save_instance_path = save_instance_path.replace('_gtFine_instanceIds','_'+str(segid)+'_gtFine_instanceIds')

        save_img_trans = np.asarray(save_img*255.0, dtype=np.uint8).transpose(1,2,0)
        save_label = np.asarray(save_label, dtype=np.uint8)
        save_instance = np.asarray(save_instance, dtype=np.uint8)


        PIL.Image.fromarray(save_img_trans, 'RGB').save(save_img_path)
        PIL.Image.fromarray(save_label, 'L').save(save_label_path)
        PIL.Image.fromarray(save_instance, 'L').save(save_instance_path)


    # json_file = self.label_paths[index][:-13]+'_polygons.json'
    # annotation = Annotation()
    # annotation.fromJsonFile(json_file)
    # for obj in annotation.objects:
    #     label   = obj.label
    #     polygon_city = obj.polygon

    #     rrlist = []
    #     cclist = []
    #     for point in polygon_city:
    #         rrlist.append(point.x)
    #         cclist.append(point.y)
    #     # print(rrlist)
    #     # print(cclist)
    #     rr = np.asarray(rrlist)
    #     cc = np.asarray(cclist)
    #     rrall, ccall = polygon(rr, cc)
    #     print(instance_tensor[0][cc, rr])

    input_dict = {'label': label_tensor,
                  'instance': instance_tensor,
                  'image': image_tensor,
                  'path': image_path,
                  }

    # Give subclasses a chance to modify the final output
    self.postprocess(input_dict)

    return input_dict
