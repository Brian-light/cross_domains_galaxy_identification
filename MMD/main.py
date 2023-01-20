import numpy as np
import json, argparse, torch, os, sys
import torch.optim as optim
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, Dataset
from sklearn import metrics

from src.network import DeepMerge
from src.data import GalaxyData
from src.train import train
import src.test as tm

def main():
    # parse commandline args
    dir_path = os.path.abspath(os.path.dirname(__file__)) #universal main directory
    parser = argparse.ArgumentParser(description='PHYS 449 Project: DeepMerge II (MMD)')
    parser.add_argument('param', type=str, default= dir_path + '/param.json', help='path to hyperparameter json', nargs='?', const=1,)
    parser.add_argument('res_path', type=str, default=dir_path + '/results', help='results directory', nargs='?', const=1,)
    args = parser.parse_args()

    # moving model to cuda
    if torch.cuda.is_available():
        model = DeepMerge().cuda()

    #get main directory and extract hyperparameters
    print('Loading hyperparameters...')
    with open(args.param) as paramfile:
        params = json.load(paramfile)
        
    # make results directory if it doesn't exist
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)
        print('Created output directory:', args.res_path)

    #data loaded from files
    print('Loading data...')
    if params['data']['use_real_dataset'] == True:
        print('Dataset selected: real')
        source_data = GalaxyData(dir_path + params['data']['data directory'] + params['data']['sim_real_src_x_file'], dir_path + params['data']['data directory'] + params['data']['sim_real_src_y_file'])
        target_data = GalaxyData(dir_path + params['data']['data directory'] + params['data']['sim_real_tar_x_file'], dir_path + params['data']['data directory'] + params['data']['sim_real_tar_y_file'])
    else:
        print('Dataset selected: simulated')
        source_data = GalaxyData(dir_path + params['data']['data directory'] + params['data']['sim_sim_src_x_file'], dir_path + params['data']['data directory'] + params['data']['sim_sim_src_y_file'])
        target_data = GalaxyData(dir_path + params['data']['data directory'] + params['data']['sim_sim_tar_x_file'], dir_path + params['data']['data directory'] + params['data']['sim_sim_tar_y_file'])
    
    #varibles used to split data
    n_source_train = int(len(source_data) * params['data']['training ratio'])
    n_source_valid = int(len(source_data) * params['data']['valid ratio'])
    n_source_test = int(len(source_data) - n_source_valid - n_source_train)
    n_target_train = int(len(target_data) * params['data']['training ratio'])
    n_target_valid = int(len(target_data) * params['data']['valid ratio'])
    n_target_test = int(len(target_data) - n_target_valid - n_target_train)
    seed_split = params['data']['seed data split']

    #split data into train and test sets
    print('Splitting data into training, validation and test sets...')
    source_train, source_valid, source_test = torch.utils.data.random_split(source_data,[n_source_train, n_source_valid, n_source_test], generator=torch.Generator().manual_seed(seed_split))
    target_train, target_valid, target_test = torch.utils.data.random_split(target_data,[n_target_train, n_source_valid, n_target_test], generator=torch.Generator().manual_seed(seed_split))

    #create data loader for each dataset
    print('Creating data loaders...')
    batch_number = params['data']['batch_size']
    source_train_dl = DataLoader(source_train, batch_size=batch_number, shuffle= True, num_workers=1, drop_last= True, persistent_workers=True)
    source_valid_dl = DataLoader(source_valid, batch_size=batch_number, shuffle= False, num_workers=1, drop_last= True, persistent_workers=True)
    source_test_dl = DataLoader(source_test, batch_size=batch_number, shuffle= False, num_workers=1, drop_last= True, persistent_workers=True)
    target_train_dl = DataLoader(target_train, batch_size=batch_number, shuffle= True, num_workers=1, drop_last= True, persistent_workers=True)
    target_valid_dl = DataLoader(target_valid, batch_size=batch_number, shuffle= False, num_workers=1, drop_last= True, persistent_workers=True)
    target_test_dl = DataLoader(target_test, batch_size=batch_number, shuffle= False, num_workers=1, drop_last= True, persistent_workers=True)

    #setup DeepMerge neural network base on available devices
    print('Initializing neural network...')
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device type:', device_string)
    if params['data']['do_transfer_model'] == False:
        model = DeepMerge().to(torch.device(device_string))
    else:
        print('Loading pre-trained model from params.json...')
        model = torch.load(dir_path + params['data']['transfer_model'], map_location=device_string)
        print('Model loaded successfully.')
    
    print('-------------------------------------------------')
    
    #enable logging 
    sys.stdout = tm.Logger(args.res_path + '/log.txt')

    #execute training using epoch number and get losses for each epoch    
    num_epochs = params['exec']['num_epochs']
    training_mode = params['exec']['mode']   #training mode, 1 for MMD + fisher + EM, else just the regular
    if training_mode == 1:
        print("Training mode: MMD + fisher + EM")
    else:
        print("Training mode: MMD only")
    losses, model = train(model,
                   num_epochs,
                   training_mode,
                   source_train_dl,
                   source_valid_dl,
                   target_train_dl,
                   target_valid_dl,
                   use_scheduler=params['exec']['use_scheduler'])
    
    #save model to res path
    model_path = args.res_path + '/DeepMerge_net.pt'
    torch.save(model, model_path)
    print('Saved model in:', model_path)
    print('-------------------------------------------------')
    print('Validating model...')

    #perform test on source and target domain and obtain the final metrics
    s_results = tm.run_test(model, source_test_dl)
    t_results = tm.run_test(model, target_test_dl)
    print('Source domain results:')
    print('Accuracy: {:.2f}%    Balanced Accuracy: {:.2f}%    Precision: {:.2f}    Recall: {:.2f}    F1: {:.2f}    Brier_score: {:.2f}    AUC: {:.2f}'.format(s_results[0] * 100, s_results[1] * 100, s_results[2], s_results[3], s_results[4], s_results[5], s_results[6]))
    print('Target domain results:')
    print('Accuracy: {:.2f}%    Balanced Accuracy: {:.2f}%    Precision: {:.2f}    Recall: {:.2f}    F1: {:.2f}    Brier_score: {:.2f}    AUC: {:.2f}'.format(t_results[0] * 100, t_results[1] * 100, t_results[2], t_results[3], t_results[4], t_results[5], t_results[6]))
    
    #Stop logging and restore stdout to default
    sys.stdout.close()
    sys.stdout = sys.__stdout__

    #plot losses
    tm.plot_loss(training_mode, losses, args.res_path)

if __name__ == '__main__':
    main()