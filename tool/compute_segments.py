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
        print ('scores:', len(output['scores']), ', high_score:', output['scores'] >0.7)
        print ('masks:', len(output['masks']))
        for i, mask in enumerate(output['masks'][output['scores']>0.7][0]):
            # f = Image.new('RGB', mask.shape)
            # Image.new("RGB", source.size)
            # data = np.zeros((source.shape[0], w, 3), dtype=np.uint8)
            # data[0:256, 0:256] = [255, 0, 0] # red patch in upper left
            array  = (mask.data.numpy().squeeze() > 0.5).astype(np.uint8)
            # print (np.mean(array), np.max(array))
            # img = Image.fromarray(array, '2')
            # img.save(os.path.join(output_path, '{}.bmp'.format(i)))

            f = os.path.join(output_path, '{}-segm.npy'.format(fileName))
            np.save(f, array)
