#Importing needed packages
import torch
import json
import argparse
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.transforms as transform
import matplotlib.pyplot as plt
import os

def array_to_tensor(name):
  data_list = torch.from_numpy(np.load(str(name), mmap_mode='r+'))
  return data_list

def normalization(t):

  mean1 = t[:,0].mean().item()
  mean2 = t[:,1].mean().item()
  mean3 = t[:,2].mean().item()

  std1 = t[:,0].std().item()
  std2 = t[:,1].std().item()
  std3 = t[:,2].std().item()

  return np.array([[mean1, mean2, mean3], [std1, std2, std3]])

def update(t1, t2):
  
  # find pixel means and stds for a given dataset
  # Use this for regularr training without transfer learning
  pristine = normalization(t1)
  noisy = normalization(t2)

  pr_trf = transform.Normalize(mean = pristine[0], std = pristine[1], inplace=True)
  no_trf = transform.Normalize(mean = noisy[0], std = noisy[1], inplace=True)

  ## for transfer learning we have to normalize new images to old means and stds 
  ## we just directly input these numbers here
  
  # pristine = [[0.0375126, 0.03326255, 0.06786563],[1.30893517, 1.02839041, 1.12307501]]
  # noisy = [[0.03749673, 0.03331003, 0.06772753],[1.37418461, 1.16330922, 1.19831419]]
  # pr_trf = transform.Normalize(mean = pristine[0], std = pristine[1], inplace=True)
  # no_trf = transform.Normalize(mean = noisy[0], std = noisy[1], inplace=True)
  
  for i in range(0, len(t1)-1):
      pr_trf(t1[i])
  
  for i in range(0, len(t2)-1):
      no_trf(t2[i])



if __name__ == '__main__':
  dir_path = os.path.abspath(os.path.dirname(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('param', type=str, default= dir_path + '/normalization_param.json', help='path to normalization hyperparameter', nargs='?', const=1,)
  args = parser.parse_args()
  param = json.load(open(args.param))

  print('beginning image normalization...')
  source = array_to_tensor(dir_path + param['directory'] + param['source'])
  target = array_to_tensor(dir_path + param['directory'] + param['target'])

  update(source, target)
  
  np.save(dir_path + param['directory'] + 'SimSim_SOURCE_X_Illustris2_pristine_normalized.npy', source)
  np.save(dir_path + param['directory'] + 'SimSim_TARGET_X_Illustris2_noisy_normalized.npy', target)
  print('normalization completed!')



"""
Distant galaxies z=2:
pristine: [[0.0375126  0.03326255 0.06786563]
 [1.30893517 1.02839041 1.12307501]] mean
noisy: [[0.03749673 0.03331003 0.06772753]
 [1.37418461 1.16330922 1.19831419]] std
Nearby galaxies z=0 (0.25 time window dusty images) and SDSS (postmergers):
pristine: [[0.60699224 0.63178635 0.56252038]
 [1.8455255  1.84083951 1.4652102 ]] mean
noisy: [[25.81263733 21.12583733 14.25884247]
 [35.08432007 29.94885445 19.8165493 ]] std
"""
