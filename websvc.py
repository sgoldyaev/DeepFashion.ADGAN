import os
from PIL import Image
from flask import Flask
from test import run
from options.test_options import TestOptions

from tool import compute_coordinates, generate_pose_map_fashion
from util import util

app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'DeepFashion.MIPT! (by sgoldyaev for DLS2020)'

@app.route('/test')
def test_model():
  opt = TestOptions().parse()
  '''
  p1_file = '01_4_full.jpg'
  p2_file = '03_4_full.jpg'

  # input

  # crop and copy
  img1 = Image.open(os.path.join('./user_data', p1_file))
  crp1 = img1.crop((704, 0, 880, 256))
  crp1.save(os.path.join('./deepfashion/test', p1_file))

  img2 = Image.open(os.path.join('./user_data', p2_file))
  crp2 = img2.crop((704, 0, 880, 256))
  crp2.save(os.path.join('./deepfashion/test', p2_file))

  # coordinates
  compute_coordinates.computeR('./deepfashion/test/', [p1_file, p2_file], './deepfashion/fashion-resize-annotation-test-web.csv')

  # pose
  generate_pose_map_fashion.compute_pose('./deepfashion', './deepfashion/fashion-resize-annotation-test-web.csv', './deepfashion/testK')

  # pairs
  with open('./deepfashion/fashion-resize-pairs-test-web.csv', 'w') as pair_file:
      pair_file.write('from,to' + '\n')
      pair_file.write('{},{}'.format(p1_file, p2_file) + '\n')
  '''
  # run test
  opt.dataroot = './deepfashion/'
  opt.name = 'fashion_adgan_test'
  opt.model = 'adgan'
  opt.phase = 'test'
  opt.dataset_mode = 'keypoint'
  opt.norm = 'instance'
  opt.resize_or_crop = 'no'
  opt.gpu_ids = 0,
  opt.BP_input_nc = 18
  #opt.SP_input_nc = 18
  opt.which_model_netG = 'ADGen'
  opt.checkpoints_dir = './checkpoints'
  # opt.pairLst = './deepfashion/fashion-resize-pairs-test-web.csv'
  opt.pairLst = './deepfashion/fashion-resize-pairs-test-small.csv'
  opt.which_epoch = '80'
  opt.results_dir = './results'
  opt.nThreads = 1   # test code only supports nThreads = 1
  opt.batchSize = 1  # test code only supports batchSize = 1
  opt.serial_batches = True  # no shuffle
  opt.no_flip = True  # no flip

  printArgs(opt)
  run(opt)
  return 'success'

def printArgs(opt):
    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')

    # save to the disk
    expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
    util.mkdirs(expr_dir)
    file_name = os.path.join(expr_dir, 'opt.txt')
    with open(file_name, 'wt') as opt_file:
        opt_file.write('------------ Options -------------\n')
        for k, v in sorted(args.items()):
            opt_file.write('%s: %s\n' % (str(k), str(v)))
        opt_file.write('-------------- End ----------------\n')

if __name__ == '__main__':
  app.run()
