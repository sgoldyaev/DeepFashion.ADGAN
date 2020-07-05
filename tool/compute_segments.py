import os
import pandas as pd
import numpy as np
from PIL import Image

import mrcnn.config
import mrcnn

def computeR(img_folder, images, output_path):
    import multiprocessing
    training_process = multiprocessing.Process(target=compute, args=[img_folder, images, output_path])
    training_process.start()
    #get_message_from_training_process(...)
    training_process.join()

def compute(img_folder, images, output_path):
    import torchvision

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # set it to evaluation mode, as the model behaves differently
    # during training and during evaluation
    model.eval()

    inputs = []
    for img in images:
        image = Image.open(os.path.join(img_folder, img))
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        inputs.append(image_tensor)
        print('Input shape:', image_tensor.shape)

    # pass a list of (potentially different sized) tensors
    # to the model, in 0-1 range. The model will take care of
    # batching them together and normalizing
    outputs = model(inputs)
    # output is a list of dict, containing the postprocessed predictions

    print ('segments', len(outputs))
    for fileName, output in list(zip(images, outputs)):
        for name, segment in output.items():
            print (name)
        print ('labels:', len(output['labels']))
        print ('scores:', len(output['scores']))
        print ('masks:', len(output['masks']))
        if len(output['masks']) > 0 :
            segmentHeatMap = np.zeros(output['masks'][0].shape[1:])
            for i, mask in enumerate(output['masks'][output['scores']>0.5]):
                array  = (mask.data.numpy().squeeze() > 0.5).astype(np.uint8)
                segmentHeatMap = segmentHeatMap + array
            fSegm = os.path.join(output_path, '{}-segm.npy'.format(fileName))
            print (fSegm, ' segmentation heatmap')
            heatMap = np.uint8(segmentHeatMap)
            np.save(fSegm, heatMap)

            # fMap = os.path.join(output_path, '{}-map.jpg'.format(fileName))
            # gt_mask = Image.fromarray(np.uint8(255 * (heatMap/heatMap.max()))).convert('RGB').convert('RGB', (.3, .5, .3, 1))
            # gt_mask.save(fMap)
