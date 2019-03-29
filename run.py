import numpy as np
import cv2
import torch
from src.crowd_count import CrowdCounter
from src import network

torch.set_grad_enabled(False)

def open_image(image_path, image_size_limit=None):
  image = cv2.imread(image_path, 0)
  image = image.astype(np.float32, copy=False)

  if image_size_limit:
    longer_dimension = np.max(image.shape)
    scaling = np.min((longer_dimension, image_size_limit)) / longer_dimension
    if scaling < 1:
      image = cv2.resize(
        image,
        (int(image.shape[1] * scaling), int(image.shape[0] * scaling)),
        interpolation=cv2.INTER_AREA
      )

  image = image.reshape((1, 1, image.shape[0], image.shape[1]))
  return image

def open_model(model_path):
  model = CrowdCounter()
  network.load_net(model_path, model)
  model.eval()
  return model


# image = open_image('./data/original/shanghaitech/part_A_final/test_data/images/IMG_3.jpg')
# image = open_image('./data/original/shanghaitech/part_B_final/test_data/images/IMG_170.jpg')
image = open_image('/Users/erichuang/Downloads/test.jpg', image_size_limit=1000)

# model = open_model('./final_models/cmtl_shtechA_204.h5')
model = open_model('./saved_models/cmtl_shtechA_1762.h5')
# model = open_model('./final_models/cmtl_shtechB_768.h5')
# model = open_model('./saved_models/cmtl_shtechB_732.h5')
density_map = model(image)

density_map = density_map.data.cpu().numpy()
crowd_count = np.sum(density_map).round().astype(int)
print(crowd_count)
