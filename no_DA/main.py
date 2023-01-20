import numpy as np
import json, argparse, torch, os
import torch.optim as optim
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import time

from src.network import DeepMerge
from src.data import Data

def setup_data(x, y, num_test_data, num_classes, batch_size):
    
    # create the test set with an equal amount of each class
    n = y.shape[0]
    train_x = np.empty((n - num_test_data, x.shape[1], x.shape[2], x.shape[3]))
    train_y = np.empty((n - num_test_data))
    test_x = np.empty((num_test_data, x.shape[1], x.shape[2], x.shape[3]))
    test_y = np.empty((num_test_data))
    
    num_per_class = int(num_test_data / num_classes)
    remaining_per_class = np.full((num_classes), num_per_class)
    extra = num_test_data - num_per_class * num_classes
    for i in range(extra):
        remaining_per_class[i] += 1
    
    i_train = 0
    i_test = 0
    indices = np.array(range(n))
    np.random.shuffle(indices)
    for i in indices:
        data_class = int(y[i])
        if remaining_per_class[data_class] > 0:
            test_x[i_test, :, :, :] = x[i, :, :, :]
            test_y[i_test] = y[i]
            i_test += 1
            remaining_per_class[data_class] -= 1
        else:
            train_x[i_train, :, :, :] = x[i, :, :, :]
            train_y[i_train] = y[i]
            i_train += 1
    
    assert sum(remaining_per_class) == 0
    
    trainset = Data(train_x, train_y, batch_size)
    testset = Data(test_x, test_y, batch_size)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainset, testset, trainloader, testloader
    
def plot_examples(data, out_path):
    print('Plotting example images...')
    data.shuffle()
    images, labels = data.get_batch(0)
    img = torchvision.utils.make_grid(images, nrow=30).cpu()
    plt.figure(dpi=300)
    plt.grid(False)
    plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    plt.savefig(out_path)
    plt.close()
    print('Exported example images to:', out_path)
    
def plot_loss(train_loss_vals, test_loss_vals, out_path):
    print('Plotting loss curves...')
    assert len(train_loss_vals)==len(test_loss_vals), 'Length mismatch between the curves'
    num_epochs = len(train_loss_vals)
    
    plt.figure(dpi=300)
    plt.plot(range(num_epochs), train_loss_vals, label= "Training loss", color="blue")
    plt.plot(range(num_epochs), test_loss_vals, label= "Test loss", color= "green")
    plt.legend()
    plt.savefig(out_path)
    plt.close()
    print('Exported plots of loss to:', out_path)
    
def train(net, trainset, testset, num_epochs, display_epochs, use_scheduler=False):
    print('Starting training...')
    t_start = time.time()
    
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.7,0.8), eps=1e-8, weight_decay=0.01, amsgrad=False)
    criterion = nn.CrossEntropyLoss()
    if use_scheduler == True:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=0.001,
                                                        steps_per_epoch=trainset.get_num_batches(),
                                                        epochs=num_epochs,
                                                        div_factor = 1,
                                                        final_div_factor = 10,
                                                        anneal_strategy = 'linear')
    
    train_loss_vals = []
    test_loss_vals = []
    
    for epoch in range(1, num_epochs + 1):
        train_loss_val = net.backprop(trainset, criterion, optimizer)
        train_loss_vals.append(train_loss_val)

        if use_scheduler == True:
            scheduler.step()

        test_loss_val = net.test(testset, criterion)
        test_loss_vals.append(test_loss_val)

        if epoch % display_epochs == 0:
            print('Epoch [{}/{}]'.format(epoch, num_epochs)+\
                  '\tTraining Loss: {:.4f}'.format(train_loss_val)+\
                  '\tTest Loss: {:.4f}'.format(test_loss_val)+\
                  '\tTraining Accuracy: {:.2f}%'.format(net.accuracy(trainset))+\
                  '\tTest Accuracy: {:.2f}%'.format(net.accuracy(testset)))
                
    print('Final training loss: {:.4f}'.format(train_loss_vals[-1]))
    print('Final test loss: {:.4f}'.format(test_loss_vals[-1]))
    
    t_end = time.time()
    print('Training time: {:.4f} seconds'.format(t_end - t_start))
    
    return train_loss_vals, test_loss_vals

def generate_report(train_loss, test_loss, overall_accuracy, accuracy_per_class, out_path):
    classes = ['Non-merger', 'Merger']
    
    with open(out_path, 'w') as f:
        f.write('Final test dataset loss:\t{:.4f}\n'.format(test_loss[-1]))
        f.write('Final test dataset accuracy:\t{:.2f}%\n\n'.format(overall_accuracy))
        
        f.write('Final test accuracy per class:\n')
        for i in range(len(accuracy_per_class)):
            f.write(classes[i] + ': {:.2f}%\n'.format(accuracy_per_class[i]))
        f.write('\n')
        
        f.write('Test dataset loss per epoch:\n')
        num_epochs = len(test_loss)
        for epoch in range(num_epochs):
            f.write('Epoch [{}/{}]'.format(epoch+1, num_epochs)+\
                    '\tTraining Loss: {:.4f}'.format(train_loss[epoch])+\
                    '\tTest Loss: {:.4f}\n'.format(test_loss[epoch]))
                
    print('Wrote test performance report to:', out_path)


#Just a seperate function for testing the target domain data using the trained model
def td_testing(td_set, model):
    print('Performing cross domain accuracy test...')
    model.eval()
    acc_lst = []

    #have to write the code out here because the network accuracy function have no batch support
    for i in range(td_set.get_num_batches()):
        print('.', end = '')
        inputs, labels = td_set.get_batch(i)
        
        with torch.no_grad():
            x, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            
            acc =  100.0 * correct / labels.size(0)

        acc_lst.append(acc)
    
    final_acc = sum(acc_lst) / len(acc_lst)

    print('\nCross Domain Accurracy:  {:.2f}%'.format(final_acc))


def main():
    # parse commandline args
    dir_path = os.path.abspath(os.path.dirname(__file__))
    parser = argparse.ArgumentParser(description='PHYS 449 Project: DeepMerge II (No Domain Adaptation)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('param', type=str, default= dir_path +'/param.json', help='path to hyperparameter json', nargs='?', const=1,)
    parser.add_argument('res_path', type=str, default=dir_path + '/results', help='results directory', nargs='?', const=1,)
    args = parser.parse_args()

    # hyperparameters from json file
    print('Loading hyperparameters...')
    with open(args.param) as paramfile:
        params = json.load(paramfile)
    print('Hyperparameters loaded!')
    
    # make results directory if it doesn't exist
    if not os.path.exists(args.res_path):
        os.makedirs(args.res_path)
        print('Created output directory:', args.res_path)
    
    np.random.seed(params['general']['seed'])
    print('Set random seed to', params['general']['seed'])
    
    # load and parse input data
    print('Loading and parsing data...')
    if params['data']['use_real_dataset'] == True:
        if params['data']['train_on_target'] == True:
            raw_x = np.load(dir_path + params['data']['path'] + params['data']['sim_real_tar_x_file'], mmap_mode= 'r+')
            raw_y = np.load(dir_path + params['data']['path'] + params['data']['sim_real_tar_y_file'], mmap_mode= 'r+')
        else:
            raw_x = np.load(dir_path + params['data']['path'] + params['data']['sim_real_src_x_file'], mmap_mode= 'r+')
            raw_y = np.load(dir_path + params['data']['path'] + params['data']['sim_real_src_y_file'], mmap_mode= 'r+')
    else:
        if params['data']['train_on_target'] == True:
            raw_x = np.load(dir_path + params['data']['path'] + params['data']['sim_sim_tar_x_file'], mmap_mode= 'r+')
            raw_y = np.load(dir_path + params['data']['path'] + params['data']['sim_sim_tar_y_file'], mmap_mode= 'r+')
        else:
            raw_x = np.load(dir_path + params['data']['path'] + params['data']['sim_sim_src_x_file'], mmap_mode= 'r+')
            raw_y = np.load(dir_path + params['data']['path'] + params['data']['sim_sim_src_y_file'], mmap_mode= 'r+')
    
    trainset, testset, trainloader, testloader = setup_data(raw_x,
                                                            raw_y,
                                                            int(params['data']['n_test_data']),
                                                            int(params['data']['num_classes']),
                                                            int(params['data']['batch_size']))
    print('Data loaded!')
    
    # show some of the images
    plot_examples(trainset, args.res_path + '/image_examples.png')
    
    # setup and train the NN
    print('Initializing neural network...')
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device type:', device_string)
    
    if params['data']['do_transfer_model'] == False:
        net = DeepMerge().to(torch.device(device_string))
    else:
        print('Loading pre-trained model from params.json...')
        net = torch.load(params['data']['transfer_model'])
        print('Model loaded successfully.')
    
    print('Network initialized with', sum(p.numel() for p in net.parameters()), 'parameters')
    
    print('-------------------------------------------------')
    
    train_loss, test_loss = train(net,
                                  trainset,
                                  testset,
                                  int(params['exec']['num_epochs']),
                                  params['exec']['display_epochs'],
                                  use_scheduler=params['exec']['use_scheduler'])

    print('-------------------------------------------------')

    net_path = args.res_path + '/no_DA_DeepMerge_net.pt'
    torch.save(net, net_path)
    print('Saved neural network in:', net_path)

    # plot results and generate report
    plot_loss(train_loss, test_loss, args.res_path + '/training_loss.png')
    overall_accuracy = net.accuracy(testset)
    test_accuracy_per_class = net.accuracy_verbose(testset, int(params['data']['num_classes']))
    generate_report(train_loss, test_loss, overall_accuracy, test_accuracy_per_class, args.res_path + '/report.txt')


    #For cross domain validation----------------------------------------------------------------------

    model = torch.load(net_path)

    #use previous variable to free up some memory
    raw_x = np.load(dir_path + params['data']['path'] + params['data']['sim_sim_tar_x_file'], mmap_mode= 'r+')
    raw_y = np.load(dir_path + params['data']['path'] + params['data']['sim_sim_tar_y_file'], mmap_mode= 'r+')
    testing_batch_size = params['data']['test_batch_size']

    td_set = Data(raw_x, raw_y, testing_batch_size)
    #data loader cause a lot of slow down and data module is not working with data loader atm
    #td_set_loader = torch.utils.data.DataLoader(td_set, batch_size=testing_batch_size, shuffle= False, num_workers=2)
    td_testing(td_set, model)


if __name__ == '__main__':
    main()
