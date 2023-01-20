import numpy as np
import torch
import time
import torch.nn as nn
from torch.nn.modules import loss
import torch.optim as optim
from torch.autograd import Variable
import src.loss as lm #lm for loss module, don't want to mistaken for actual loss
import src.test as tm #tm for test module
import copy


def lr_scheduler(optimizer, cur_epoch, initial_lr, cycle = 6, decay = 0.0002):
    lr = initial_lr - cur_epoch % cycle * decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
   

def train(model, grad_reverse_layer, optimizer, lambda_tl, num_epochs, training_mode, source_train_dl, source_valid_dl, target_train_dl, target_valid_dl, use_scheduler=False):
    print('Training model...')

    epoch_losses = [[], [], [], [], []] #classifier loss, transfer loss, fisher loss, em loss, total loss
    classifier_criterion = nn.CrossEntropyLoss()
    if use_scheduler == True:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=0.001,
                                                        steps_per_epoch=len(source_train_dl),
                                                        epochs= 8,
                                                        div_factor = 1,
                                                        final_div_factor = 10,
                                                        anneal_strategy = 'linear')
    
    fisher = lm.FisherTR(num_classes = 2, feat_dim = (32* 9 * 9))

    # keeps track of current epoch and target domain accuracy for early stopping condition
    best_epoch_num = 1
    best_accuracy = 0
    best_model = []

    for epoch in range(num_epochs):
        t_start = time.time() #time 2
        running_losses = [[], [], [], [], []]   #classifier loss, transfer loss, fisher loss, em loss, total loss

        dataloader_iterator = iter(target_train_dl) #serve no purpose other than to make the loop below functional
        model.train()   #training mode engaged
        

        for i, (x_source, y_source) in enumerate(source_train_dl):
            #the loop can get a batch from both dataloaders in the same iteration (ugly but works fine)
            try:
                x_target, y_target = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(target_train_dl)
                x_target, y_target = next(dataloader_iterator)
            
            #Just a wrapper for the tensors, supposely improve peformance (unsure), and configures cuda
            if torch.cuda.is_available(): 
                input_source, label_source, input_target = Variable(x_source).cuda(), Variable(y_source).cuda(), Variable(x_target).cuda()           
            else:
                input_source, label_source, input_target = Variable(x_source), Variable(y_source), Variable(x_target)

            optimizer.zero_grad()
            x = torch.cat((input_source, input_target), dim=0)  #concatenate source and targte training set
            source_batch_size = input_source.size(0)    #trainng set size of each domain 
            if torch.cuda.is_available():
                x = x.cuda()
                features, logits = model.cuda().base_net(x)
            else:
                features, logits = model.base_net(x)
            source_logits = logits.narrow(0, 0, source_batch_size)

            # domain classifier
            #model.adv_net.train(True)
            weight_adv = torch.ones(x.size(0))
            transfer_loss = lm.PADA(features, model.adv_net, grad_reverse_layer, weight_adv)
            adv_out, _ = model.adv_net(features.detach())
            adv_acc, adv_acc_source, adv_acc_target = tm.domain_cls_accuracy(adv_out)

            #losses and their corresponding lambdas
            classifier_loss = classifier_criterion(source_logits, label_source)
            fisher_loss = 0
            em_loss = 0
            lambda_w = 0.0001
            lambda_b = 1
            lambda_em = 0.0001

            #Training mode = 1, we add fisher and entropy minimization loss, else nothing
            if training_mode == 1:
                em_loss = lm.EntropyLoss(nn.Softmax(dim=1)(logits))

                fisher_loss, _, _, _ = fisher.forward(features.narrow(0, 0, int(x.size(0)/2)), label_source, inter_class = 'global', intra_loss_weight = lambda_w, inter_loss_weight = lambda_b)

            #final loss 
            total_loss = classifier_loss + lambda_tl * transfer_loss + lambda_em * em_loss + fisher_loss

            #accumulate losses each batch
            running_losses[0].append(classifier_loss.item())
            running_losses[1].append(lambda_tl * transfer_loss.item())
            if training_mode == 1:
                running_losses[2].append(fisher_loss.item())
                running_losses[3].append(em_loss.item() * lambda_em)
            running_losses[4].append(total_loss.item())

            total_loss.backward()
            optimizer.step()
            
            if use_scheduler == True:
                scheduler.step()
        
            
        #use validation set to get metrics for each epoch on source and target domain
        source_results = tm.run_test(model, source_valid_dl)
        target_results = tm.run_test(model, target_valid_dl)

        #accumulate losses each epoch
        epoch_losses[0].append(np.mean(running_losses[0]))
        epoch_losses[1].append(np.mean(running_losses[1]))
        if training_mode == 1:
            epoch_losses[2].append(np.mean(running_losses[2]))
            epoch_losses[3].append(np.mean(running_losses[3]))
        epoch_losses[4].append(np.mean(running_losses[4]))

        t_end = time.time() #time 2

        #Different printing mode for different training mode
        if training_mode == 1:
            print('Epoch [{}/{}]\tTime: {:.2f} seconds\t'.format(epoch + 1, num_epochs, t_end - t_start) +
                'Total Loss: {:.3f}\tClassifier Loss: {:.3f}\tTransfer Loss: {:.5f}\tFisher Loss: {:.3f}\tEM Loss: {:.3f}\t'.format(epoch_losses[4][epoch], 
                                                                                                                                    epoch_losses[0][epoch],
                                                                                                                                    epoch_losses[1][epoch],
                                                                                                                                    epoch_losses[2][epoch],
                                                                                                                                    epoch_losses[3][epoch]) +  
                'Source Domain Accuracy: {:.2f}%    Target Domain Accuracy: {:.2f}%'.format(source_results[0] * 100, target_results[0] * 100))
            #print(scheduler.get_last_lr())
        else:
            print('Epoch [{}/{}]\tTime: {:.2f} seconds\t'.format(epoch + 1, num_epochs, t_end - t_start) +
                'Total Loss: {:.3f}\tClassifier Loss: {:.3f}\tTransfer Loss: {:.5f}\t'.format(epoch_losses[4][epoch], 
                                                                                              epoch_losses[0][epoch],
                                                                                              epoch_losses[1][epoch]) +  
                'Source Domain Accuracy: {:.2f}%    Target Domain Accuracy: {:.2f}%'.format(source_results[0] * 100, target_results[0] * 100))
            #print(scheduler.get_last_lr())
        
        # applying early stop
        if  target_results[0] > best_accuracy:
            best_epoch_num = epoch
            best_accuracy = target_results[0]
            best_model = copy.deepcopy(model)

        if epoch - best_epoch_num >= 20:
            print("triggered early stopping")
            return epoch_losses, best_model

    print('Training complete')
    
    return epoch_losses, best_model