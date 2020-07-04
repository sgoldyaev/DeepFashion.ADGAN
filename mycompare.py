import numpy as np
import pandas as pd

from PIL import Image, ImageDraw
from tool import compute_coordinates, compute_segments, generate_pose_map_fashion
from util import util

img1 = Image.open('./deepfashion/my/02_7_additional_orig.jpg')
crp1 = img1.crop((0+40, 0, 256-40, 256))
crp1.save('./deepfashion/my/02_7_additional.jpg')

compute_coordinates.compute('./deepfashion/my/', ['02_7_additional.jpg'], './deepfashion/my_results/fashion-resize-annotation-my.csv')
compute_segments.compute('./deepfashion/my/', ['02_7_additional.jpg'], './deepfashion/my_results')
#generate_pose_map_fashion.compute_pose('./deepfashion/my/', './deepfashion/my_results/fashion-resize-annotation-my.csv', './deepfashion/my_results')

#02_7_additional.jpg: [36, 81, 80, 141, 202, 82, 148, 208, 196, -1, -1, 195, -1, -1, 28, 26, 32, 32]: [88, 89, 52, 41, 44, 124, 132, 148, 72, -1, -1, 118, -1, -1, 81, 97, 71, 108]
#02_7_additional.jpg: [36, 82, 81, 141, 202, 83, 148, 207, 195, -1, -1, 195, -1, -1, 27, 27, 30, 32]: [87, 88, 51, 41, 43, 122, 128, 144, 68, -1, -1, 114, -1, -1, 79, 96, 70, 107]

my_coords_Y = np.array([36, 81, 80, 141, 202, 82, 148, 208, 196, -1, -1, 195, -1, -1, 28, 26, 32, 32])
my_coords_X = np.array([88, 89, 52, 41, 44, 124, 132, 148, 72, -1, -1, 118, -1, -1, 81, 97, 71, 108])
my_coords = list(zip(my_coords_X, my_coords_Y))

# gt_coords_Y = np.array([36, 82, 81, 141, 202, 83, 148, 207, 195, -1, -1, 195, -1, -1, 27, 27, 30, 32])
# gt_coords_X = np.array([87, 88, 51, 41, 43, 122, 128, 144, 68, -1, -1, 114, -1, -1, 79, 96, 70, 107])
# gt_coords = list(zip(gt_coords_X, gt_coords_Y))

source = Image.open('./deepfashion/my/02_7_additional.jpg')

my_target = Image.new("RGB", source.size)
my_draw = ImageDraw.Draw(my_target)
my_draw.point((my_coords), fill='white')
my_target.save('./deepfashion/my_results/my_target.png', 'PNG')

# gt_target = Image.new("RGB", source.size)
# gt_draw = ImageDraw.Draw(gt_target)
# gt_draw.polygon((gt_coords), fill='black', outline='blue')
# gt_target.save('./deepfashion/my_results/gt_target.png', 'PNG')

my_npy = np.load('./deepfashion/my_results/02_7_additional.jpg.npy')
gt_npy = np.load('./deepfashion/gt_results/02_7_additional.npy')

gt_df = pd.DataFrame(data=gt_npy)
gt_df.to_csv ('./deepfashion/my_results/gt_npy.csv')

my_df = pd.DataFrame(data=my_npy)
my_df.to_csv ('./deepfashion/my_results/my_npy.csv')