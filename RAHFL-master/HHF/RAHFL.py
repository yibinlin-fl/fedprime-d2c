import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import sys
sys.path.append("..")
from Dataset.utils import init_logs, partition_data, get_dataloader, init_nets, generate_public_data_indexs, mkdirs
from loss import SupConLoss, DCLLoss
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
from numpy import *
import random
import torch
import torch.backends.cudnn

#20clients
'''
Global Parameters
'''
Seed = 0
TrainBatchSize = 256
TestBatchSize = 512
Pretrain_Epoch = 40
CommunicationEpoch = 40
Pariticpant_Params = {
    'loss_funnction' : 'CE',
    'optimizer_name' : 'Adam',
    'learning_rate'  : 0.001
}
AugMix_Module = 'jsd' #['jsd', 'nojsd', None]
CL_Module = 'dcl' #['supcon', 'dcl', None]
Col_Module = 'asymhfl' #['hfl', 'asymhfl']
"""Corruption Setting"""
Private_Corrupt_Rate = 1 #[0, 1, 0.5, 0.2, 0.8]
Test_Corrupt_Rate = 1 #[0, 1]
Public_Corrupt_Rate = 0 #[0, 1]
"""Model Setting"""
Model_setting = 'hetero' #['hetero', 'homo']
Nets_Name_List = ['ResNet10','ResNet12','ShuffleNet','Mobilenetv2']
# Nets_Name_List = ['ResNet12','ResNet12','ResNet12','ResNet12']
N_Participants = len(Nets_Name_List)
"""Private Dataset Setting"""
Private_Dataset_Name = 'cifar10' #['cifar10']
Private_Dataset_Dir = '../Dataset/cifar_10_c'
Private_Data_Len = 10000
Private_Output_Channel = 10
Data_Partition = 'iid' #['iid', 'noniid']
Noniid_Dirichlet_Beta = 0 # 0 for iid
"""Public Dataset Setting"""
Public_Dataset_Name = 'cifar100'
Public_Dataset_Dir = '../Dataset/cifar_100_c'
Public_Dataset_Length = 5000
"""Model Save Setting"""
Model_Load_Dir = '../Network/Model_Storage_' + Model_setting + '_4client/' + Private_Dataset_Name + '_' + Data_Partition + '_' + str(Noniid_Dirichlet_Beta) + '/random_corrupt_' + str(Private_Corrupt_Rate)
Model_Save_Dir = '../Final_Model_Storage_' + Model_setting + '_4client/' + Private_Dataset_Name + '_' + Data_Partition + '_' + str(Noniid_Dirichlet_Beta) +'/random_corrupt_' + str(Private_Corrupt_Rate)
"""Matrix Update Times"""
Matrix_Update_Epoch = 1

def evaluate_network(network,dataloader,logger):
    network.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        logger.info('Test Accuracy of the model on the test images: {} %'.format(acc))
    return acc

def update_model_via_private_data(device,network,localtrain_epoch,private_dataloader,loss_function,optimizer_method,learning_rate,logger):
    if loss_function =='CE':
        criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    if optimizer_method =='Adam':
        optimizer = optim.Adam(network.parameters(),lr=learning_rate)
    if optimizer_method =='SGD':
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    participant_local_loss_batch_list = []
    for _ in range(localtrain_epoch):
        for batch_idx, (images, labels) in enumerate(private_dataloader):
            if AugMix_Module == 'jsd':
                #Augmix+JSD
                images_all = torch.cat([images[0], images[1], images[2]], 0).to(device)
                labels = labels.to(device)
                logits_all, _ = network(images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(logits_all, images[0].size(0))
                labels = labels.long()
                # Cross-entropy is only computed on clean images
                loss = criterion(logits_clean, labels)
                p_clean, p_aug1, p_aug2 = F.softmax(
                    logits_clean, dim=1), F.softmax(
                    logits_aug1, dim=1), F.softmax(
                    logits_aug2, dim=1)
                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                jsd_loss = (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                            F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
                loss += 12 * jsd_loss
                if CL_Module == 'supcon':
                    #-------------feature contrastive loss--------------
                    images_cont = torch.cat([images[0], images[1]], 0).to(device)
                    features = network.module.backbone(images_cont)
                    features = F.normalize(features, dim=1)
                    fclean, f1 = torch.split(features, images[0].size(0))
                    features = torch.cat([fclean.unsqueeze(1), f1.unsqueeze(1)], dim=1)
                    cont_loss_fn = SupConLoss(temperature=0.2, device=device)
                    cont_loss = cont_loss_fn(features, labels) #supcontrast
                    loss += cont_loss
                    #-------------feature contrastive loss--------------
                elif CL_Module == 'dcl':
                    #-------------feature contrastive dmm loss--------------
                    images_cont = torch.cat([images[0], images[1], images[3]], 0).to(device)
                    features = network.module.backbone(images_cont)
                    features = F.normalize(features, dim=1)
                    fclean1, f1, fclean2 = torch.split(features, images[0].size(0))
                    cont_supddm_loss_fn = DCLLoss(temperature=0.2, device=device, beta=1.0, ddm_temperature=0.2)
                    cont_loss = cont_supddm_loss_fn(original_feature=fclean1.unsqueeze(1), weak_feature=fclean2.unsqueeze(1), strong_feature=f1.unsqueeze(1), labels=labels) #supcontrast
                    loss += cont_loss
                    #-------------feature contrastive dmm loss--------------
            else:
                #---------------Original code------------------
                images = images.to(device)
                labels = labels.to(device)
                outputs, _ = network(images)
                labels = labels.long()
                loss = criterion(outputs, labels)
                #---------------Original code------------------
            optimizer.zero_grad()
            participant_local_loss_batch_list.append(loss.item())
            loss.backward()
            optimizer.step()
            logger.info('Private Train : [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(images[0]), len(private_dataloader.dataset),
                100. * batch_idx / len(private_dataloader), loss.item()))
    return network, participant_local_loss_batch_list

if __name__ =='__main__':
    logger = init_logs()
    logger.info("Random Seed and Server Config")
    seed = Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device_ids = [0,1,2,3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    logger.info("Initialize Participants' Data idxs")
    if Data_Partition == 'noniid':
        net_datanum_map = {}
        _, net_dataidx_map = partition_data(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,partition=Data_Partition,
                                        num_classes=Private_Output_Channel,num_users=N_Participants,corrupt_rate=Private_Corrupt_Rate,
                                        test_corrupt_rate=Test_Corrupt_Rate,dirichlet_beta=Noniid_Dirichlet_Beta)
    elif Data_Partition == 'iid':
        net_dataidx_map = {}
        for index in range(N_Participants):
            idxes = np.random.permutation(50000)
            idxes = idxes[0:Private_Data_Len]
            net_dataidx_map[index]= idxes
    logger.info(net_dataidx_map)

    net_list = init_nets(n_parties=N_Participants,nets_name_list=Nets_Name_List,num_classes=Private_Output_Channel)

    logger.info("Load Participants' Models")
    for i in range(N_Participants):
        network = net_list[i]
        network = nn.DataParallel(network, device_ids=device_ids).to(device)
        netname = Nets_Name_List[i]
        network.load_state_dict(torch.load(Model_Load_Dir + '/' + netname+'_'+str(i)+'.ckpt'))

    logger.info("Initialize Public Data Parameters")
    public_data_indexs = generate_public_data_indexs(dataset=Public_Dataset_Name,datadir=Public_Dataset_Dir,size=Public_Dataset_Length)
    public_train_dl, _, public_train_ds, _ = get_dataloader(dataset=Public_Dataset_Name,datadir=Public_Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                                                            dataidxs=public_data_indexs,corrupt_rate=Public_Corrupt_Rate,augmix_module=None)
    
    col_loss_list = []
    local_loss_list = []
    local_jsd_loss_list = []
    acc_list = []

    for epoch_index in range(CommunicationEpoch):
        logger.info("The "+str(epoch_index)+" th Communication Epoch")
        logger.info('Evaluate Models')
        acc_epoch_list = []
        if epoch_index % Matrix_Update_Epoch == 0:
            matrix_update_acc_list = []
        for participant_index in range(N_Participants):
            netname = Nets_Name_List[participant_index]
            private_dataset_dir = Private_Dataset_Dir
            _, test_dl, _, _= get_dataloader(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                                             test_corrupt_rate=Test_Corrupt_Rate)
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
            acc_epoch_list.append(accuracy)
            if epoch_index % Matrix_Update_Epoch == 0:
                matrix_update_acc_list.append(accuracy)
        acc_list.append(acc_epoch_list)
        accuracy_avg = sum(acc_epoch_list) / N_Participants
        logger.info('Average Test Accuracy of the models on the test images: {} %'.format(accuracy_avg))

        '''
        HHF
        '''
        for batch_idx, (images, _) in enumerate(public_train_dl):
            linear_output_list = []
            linear_output_target_list = []
            kl_loss_batch_list = []
            participant_kl_list = []
            '''
            Calculate Linear Output
            '''
            for participant_index in range(N_Participants):
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network.train()
                #---------------HFL-----------------
                images = images.to(device)
                logits, _ = network(x=images)
                p_logits = F.softmax(logits,dim =1)
                linear_output_target_list.append(p_logits.clone().detach())
                plog_logits = F.log_softmax(logits,dim=1)
                linear_output_list.append(plog_logits)
                #---------------HFL-----------------
            
            '''
            Update Participants' Models via KL Loss and Data Quality
            '''
            for participant_index in range(N_Participants):
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                network.train()
                criterion = nn.KLDivLoss(reduction='batchmean')
                criterion.to(device)
                optimizer = optim.Adam(network.parameters(), lr=Pariticpant_Params['learning_rate'])
                optimizer.zero_grad()
                loss = torch.tensor(0.0)
                #-----------------------------
                learn_from_client_num = 0
                total_weight = 0
                #-----------------------------
                if Col_Module == 'hfl':
                    for i in range(N_Participants):
                        if i != participant_index:
                    #-----------HFL----------------
                            weight_index = 1 / (N_Participants - 1)
                            loss_batch_sample = criterion(linear_output_list[participant_index], linear_output_target_list[i])
                            temp = weight_index * loss_batch_sample
                            loss = loss + temp
                    kl_loss_batch_list.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    #-----------HFL----------------
                elif Col_Module == 'asymhfl':
                    #---------learn from the clients better than me-------
                    for i in range(N_Participants):
                        if i != participant_index:
                            if matrix_update_acc_list[participant_index] <= matrix_update_acc_list[i]:
                                learn_from_client_num += 1
                                loss_batch_sample = criterion(linear_output_list[participant_index], linear_output_target_list[i])
                                loss = loss + loss_batch_sample
                    if learn_from_client_num != 0:
                        loss = loss / learn_from_client_num
                        kl_loss_batch_list.append(loss.item())
                        loss.backward()
                        optimizer.step()
                    #---------learn from the clients better than me-------
            col_loss_list.append(kl_loss_batch_list)

        '''
        Update Participants' Models via Private Data
        '''
        local_loss_batch_list = []
        for participant_index in range(N_Participants):
            network = net_list[participant_index]
            network = nn.DataParallel(network, device_ids=device_ids).to(device)
            network.train()
            private_dataidx = net_dataidx_map[participant_index]
            train_dl, _, train_ds, _= get_dataloader(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                                                    dataidxs=net_dataidx_map[participant_index],corrupt_rate=Private_Corrupt_Rate,augmix_module=AugMix_Module)
            localtrain_epoch = max(int(len(public_train_ds)/len(train_ds)),1)
            network,private_loss_batch_list = update_model_via_private_data(device=device,network=network,localtrain_epoch=localtrain_epoch,
                                                                            private_dataloader=train_dl,loss_function=Pariticpant_Params['loss_funnction'],
                                                                            optimizer_method=Pariticpant_Params['optimizer_name'],
                                                                            learning_rate=Pariticpant_Params['learning_rate'],logger=logger)
            mean_private_loss_batch = mean(private_loss_batch_list)
            local_loss_batch_list.append(mean_private_loss_batch)
        local_loss_list.append(local_loss_batch_list)

        """
        Evaluate Models in the final round
        """
        if epoch_index == CommunicationEpoch - 1:
            acc_epoch_list = []
            logger.info('Final Evaluate Models')
            for participant_index in range(N_Participants): 
                _, test_dl, _, _= get_dataloader(dataset=Private_Dataset_Name,datadir=Private_Dataset_Dir,train_bs=TrainBatchSize,test_bs=TestBatchSize,
                                                test_corrupt_rate=Test_Corrupt_Rate)
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                accuracy = evaluate_network(network=network, dataloader=test_dl, logger=logger)
                acc_epoch_list.append(accuracy)
            accuracy_avg = sum(acc_epoch_list) / N_Participants
            logger.info('Average Test Accuracy of the models on the test images: {} %'.format(accuracy_avg))

            logger.info('Save Models')
            mkdirs(Model_Save_Dir)
            for participant_index in range(N_Participants):
                netname = Nets_Name_List[participant_index]
                network = net_list[participant_index]
                network = nn.DataParallel(network, device_ids=device_ids).to(device)
                torch.save(network.state_dict(), Model_Save_Dir + '/' + netname+'_'+str(participant_index)+'.ckpt')
