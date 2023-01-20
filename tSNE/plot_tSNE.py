import numpy as np
import torch, argparse, sys, json
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os.path
import pandas as pd
from sklearn.metrics import confusion_matrix
from collections import deque
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

sys.path.append('../Adversarial')
from src.network import DeepMerge, AdversarialNetwork, AdvDeepMerge

class GalaxyData(Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path)
        self.x = torch.from_numpy(self.x).type(torch.float32)

        self.y = np.load(y_path)
        self.y = torch.from_numpy(self.y).type(torch.long)

        self.length = self.y.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.length

def plot_embedding(X, y, d, title=None, imgName=None, save_dir=None):
    '''
    Plot an embedding X with the class label y colored by the domain d.
    Args:
        param X: embedding
        param y: label
        param d: domain
        param title: title on the figure
        param imgName: the name of saving image
    Returns:
        tSNE plots of the embeddings
    '''
    fig_mode = 'save'

    if fig_mode is None:
        return

    # Normalization
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Plot figure
    plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    alpha_list = [.3, 1]

    #set opacity for domain> [source, target]
    alpha_list = [.3, 1]

    for i in range(X.shape[0]):
      
    	plt.scatter(X[i, 0], X[i, 1], marker='o', alpha= alpha_list[d[i]],
              color=plt.cm.bwr(y[i]/1.))

    plt.xticks([]), plt.yticks([])

    # If title is not given, we assign training_mode to the title.
    if title is not None:
        plt.title(title)
    else:
        plt.title('Blobs')

    if fig_mode == 'display':
        # Directly display if no folder provided.
        plt.show()

    if fig_mode == 'save':
        # Check if folder exist, otherwise need to create it.
        folder = os.path.abspath(save_dir)

        if not os.path.exists(folder):
            os.makedirs(folder)

        # Check extension in case.
        if not (imgName.endswith('.jpg') or imgName.endswith('.png') or imgName.endswith('.jpeg')):
            imgName = os.path.join(folder, imgName + '.pdf')

        print('Saving ' + imgName + ' ...')
        plt.savefig(imgName)
        plt.close()


def visualizePerformance(base_network, src_test_dataloader,
                         tgt_test_dataloader, batch_size, domain_classifier=None, num_of_samples=None, imgName=None, use_gpu=True, save_dir=None):
    """
    Evaluate the performance of dann and source only by visualization.
    Args:
        param feature_extractor: network used to extract feature from target samples
        param class_classifier: network used to predict labels
        param domain_classifier: network used to predict domain
        param source_dataloader: test dataloader of source domain
        param target_dataloader: test dataloader of target domain
        batch_size: batch size used in the main code 
        param num_of_samples: the number of samples (from train and test respectively) for t-sne
        param imgName: the name of saving image

    Returns:
        tSNE plots, ploted by calling plot_embedding function
    """

    # Setup the network
    base_network.eval()
    if domain_classifier is not None:
        domain_classifier.eval()

    # Randomly select samples from source domain and target domain.
    if num_of_samples is None:
        num_of_samples = batch_size
    else:
        assert len(src_test_dataloader) * num_of_samples, \
            'The number of samples can not bigger than dataset.' # NOT PRECISELY COMPUTATION

    # Collect source data.-- labeled with 0
    s_images, s_labels, s_tags = [], [], []
    for batch in src_test_dataloader:
        images, labels = batch

        if use_gpu:
            s_images.append(images.cuda())
        else:
            s_images.append(images)
        
        s_labels.append(labels)
        s_tags.append(torch.zeros((labels.size()[0])).type(torch.LongTensor))

        if len(s_images * batch_size) > num_of_samples:
            break

    s_images, s_labels, s_tags = torch.cat(s_images)[:num_of_samples], \
                                 torch.cat(s_labels)[:num_of_samples], torch.cat(s_tags)[:num_of_samples]

    # Collect test data.-- labeled with 1
    t_images, t_labels, t_tags = [], [], []
    for batch in tgt_test_dataloader:
        images, labels = batch

        if use_gpu:
            t_images.append(images.cuda())
        else:
            t_images.append(images)

        t_labels.append(labels)
        t_tags.append(torch.ones((labels.size()[0])).type(torch.LongTensor))

        if len(t_images * batch_size) > num_of_samples:
            break

    t_images, t_labels, t_tags = torch.cat(t_images)[:num_of_samples], \
                                 torch.cat(t_labels)[:num_of_samples], torch.cat(t_tags)[:num_of_samples]

    # Compute the embedding of target domain.
    embedding1, logits = base_network(s_images)
    embedding2, logits = base_network(t_images)

    tsne = TSNE(perplexity=36, metric= 'cosine', n_components=2, init='pca', n_iter=3000)

    if use_gpu:
        network_tsne = tsne.fit_transform(np.concatenate((embedding1.cpu().detach().numpy(),
                                                       embedding2.cpu().detach().numpy())))
    else:
        network_tsne = tsne.fit_transform(np.concatenate((embedding1.detach().numpy(),
                                                   embedding2.detach().numpy())))


    plot_embedding(network_tsne, np.concatenate((s_labels, t_labels)),
                         np.concatenate((s_tags, t_tags)), ' ', imgName, save_dir)

def main():
    parser = argparse.ArgumentParser(description='PHYS 449 Project: DeepMerge II (MMD)')
    parser.add_argument('param', type=str, default='tSNE_param.json', help='path to hyperparameter json', nargs='?', const=1,)
    parser.add_argument('plot_name', type=str, default='tSNE.png', help='filename of plot', nargs='?', const=1,)
    args = parser.parse_args()

    #get main directory and extract hyperparameters
    print('Loading hyperparameters...')
    dir_path = os.path.abspath(os.path.dirname(__file__)) #universal main directory
    with open(dir_path + "/" + args.param) as paramfile:
        params = json.load(paramfile)
        
    #data loaded from files
    print('Loading data...')
    if params['use_real_dataset'] == True:
        source_data = GalaxyData(params['data_path'] + params['sim_real_src_x_file'], params['data_path'] + params['sim_real_src_y_file'])
        target_data = GalaxyData(params['data_path'] + params['sim_real_tar_x_file'], params['data_path'] + params['sim_real_tar_y_file'])
    else:
        source_data = GalaxyData(params['data_path'] + params['sim_sim_src_x_file'], params['data_path'] + params['sim_sim_src_y_file'])
        target_data = GalaxyData(params['data_path'] + params['sim_sim_tar_x_file'], params['data_path'] + params['sim_sim_tar_y_file'])
    
    batch_size = params['batch_size']
    source_test_dl = DataLoader(source_data, batch_size=batch_size, shuffle= True, num_workers=1, drop_last= True, persistent_workers=True)
    target_test_dl = DataLoader(target_data, batch_size=batch_size, shuffle= True, num_workers=1, drop_last= True, persistent_workers=True)
    
    device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.load(params['model'], map_location=device_string)
    
    if params['is_adversarial'] == True:
        visualizePerformance(model.base_net,
                             source_test_dl,
                             target_test_dl,
                             batch_size,
                             model.adv_net,
                             params['num_samples'],
                             params['out_directory'] + args.plot_name,
                             torch.cuda.is_available(),
                             params['out_directory'])
    else:
        visualizePerformance(model,
                             source_test_dl,
                             target_test_dl,
                             batch_size,
                             None,
                             params['num_samples'],
                             params['out_directory'] + args.plot_name,
                             torch.cuda.is_available(),
                             params['out_directory'])


if __name__ == '__main__':
    main()

