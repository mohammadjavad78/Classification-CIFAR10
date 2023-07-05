import os
import torch
from torchvision import transforms
import torch.nn as nn
from dataloaders import  DataLoader
import matplotlib.pyplot as plt
from nets.model import ModelCNN
from utlis.Averagemeter import AverageMeter
import numpy as np






def load_model(ckpt_path, model, optimizer=None):
    """
    Loading a saved model and optimizer (from checkpoint)
    """
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])
    if (optimizer != None) & ("optimizer" in checkpoint.keys()):
        optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res



labelss=['Airplane','Automobile','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']






def test(
    train_loader,
    val_loader,
    test_loader,
    model,
    model_name,
    device,
    ckpt_path,
    load_saved_model
):
    fig,axes= plt.subplots(3,3,figsize=(10,10))

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()


    if os.path.exists(f"{ckpt_path}/ckpt_{model_name}.ckpt"):
        if load_saved_model:
            model, _ = load_model(
                ckpt_path=f"{ckpt_path}/ckpt_{model_name}.ckpt", model=model
            )
    
    top1_acc_train = AverageMeter()
    loss_avg_train = AverageMeter()
    top1_acc_val = AverageMeter()
    loss_avg_val = AverageMeter()
    top1_acc_test = AverageMeter()
    loss_avg_test = AverageMeter()

    model.eval()
    mode = "train"
    
    
    (images, labels)=train_loader.__getitem__()
    


    images = images.to(device)
    labels = labels.to(device)
    cri=[[0 for i in range(10)] for j in range(len(labels))]
    for i in range(len(labels)):
        cri[i][int(labels[i])]=1
    cri=torch.Tensor(cri).to(device)
    

    labels_pred = model(images)
    for i in range(3):
        dd=images[i].cpu().numpy()
        dd = (dd - dd.min()) / (dd.max() - dd.min())
        dd=np.rot90(dd)
        dd=np.rot90(dd)
        dd=np.rot90(dd)
        dd=np.fliplr(dd)
        axes[i][0].set_ylabel(f'actual: {labelss[int(labels[i])]} \n predict: {labelss[labels_pred[i].argmax()]}')
        axes[i][0].set_xticks([])
        axes[i][0].set_yticks([])
        axes[2][0].set_xlabel('train')
        axes[i][0].imshow(dd.transpose(0,2,1),aspect='auto')
    loss = criterion(labels_pred, cri)
    acc1 = accuracy(labels_pred, labels)
    top1_acc_train.update(acc1[0], images.size(0))
    loss_avg_train.update(loss.item(), images.size(0))

        


    model.eval()
    mode = "val"
    with torch.no_grad():
        (images, labels)=val_loader.__getitem__()
        
        images = images.to(device).float()
        labels = labels.to(device)
        labels_pred = model(images)
        for i in range(3):
            dd=images[i].cpu().numpy()
            dd = (dd - dd.min()) / (dd.max() - dd.min())
            dd=np.rot90(dd)
            dd=np.rot90(dd)
            dd=np.rot90(dd)
            dd=np.fliplr(dd)
            axes[i][1].set_ylabel(f'actual: {labelss[int(labels[i])]} \n predict: {labelss[labels_pred[i].argmax()]}')
            axes[i][1].set_xticks([])
            axes[i][1].set_yticks([])
            axes[2][1].set_xlabel('val')
            axes[i][1].imshow(dd.transpose(0,2,1),aspect='auto')
        cri=[[0 for i in range(10)] for j in range(len(labels))]
        for i in range(len(labels)):
            cri[i][int(labels[i])]=1
        cri=torch.Tensor(cri).to(device)
        loss = criterion(labels_pred, cri)
        acc1 = accuracy(labels_pred, labels)
        top1_acc_val.update(acc1[0], images.size(0))
        loss_avg_val.update(loss.item(), images.size(0))


        


    model.eval()
    mode = "test"
    with torch.no_grad():
        (images, labels)=test_loader.__getitem__()
        
        images = images.to(device).float()
        labels = labels.to(device)
        labels_pred = model(images)
        for i in range(3):
            dd=images[i].cpu().numpy()
            dd = (dd - dd.min()) / (dd.max() - dd.min())
            dd=np.rot90(dd)
            dd=np.rot90(dd)
            dd=np.rot90(dd)
            dd=np.fliplr(dd)
            axes[i][2].set_ylabel(f'actual: {labelss[int(labels[i])]} \n predict: {labelss[labels_pred[i].argmax()]}')
            axes[i][2].set_xticks([])
            axes[i][2].set_yticks([])
            axes[2][2].set_xlabel('test')
            
            axes[i][2].imshow(dd.transpose(0,2,1),aspect='auto')
        cri=[[0 for i in range(10)] for j in range(len(labels))]
        for i in range(len(labels)):
            cri[i][int(labels[i])]=1
        cri=torch.Tensor(cri).to(device)
        loss = criterion(labels_pred, cri)
        acc1 = accuracy(labels_pred, labels)
        top1_acc_test.update(acc1[0], images.size(0))
        loss_avg_test.update(loss.item(), images.size(0))


            
    train_loader.resetiter()
    val_loader.resetiter()
    test_loader.resetiter()

    fig.suptitle(f'{model_name}')
    plt.show()
    fig.savefig(f'{model_name}.png')
        
    return model



from utlis import Read_yaml


yml=Read_yaml.Getyaml()


model_name=yml['model_name']


batch_size = yml['batch_size']
epochs = yml['num_epochs']
learning_rate = yml['learning_rate']
gamma=yml['gamma']
step_size=yml['step_size']
ckpt_save_freq = yml['ckpt_save_freq']




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
custom_model = ModelCNN(model_name)

DIR = yml['dataset']

cifar_transforms_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])

cifar_transforms_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])

cifar_transforms_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.49139968, 0.48215827 ,0.44653124), (0.24703233,0.24348505,0.26158768))])
random_state=None
cifar_train_loader = DataLoader.CustomImageDataset(DIR,cifar_transforms_train,train=True,batch_size=3,shuffle=True,random_state=random_state)

cifar_val_loader = DataLoader.CustomImageDataset(DIR,cifar_transforms_val,train=True,val=True,batch_size=3,shuffle=True,random_state=random_state)

cifar_test_loader = DataLoader.CustomImageDataset(DIR,cifar_transforms_test,test=True,batch_size=3,shuffle=True,random_state=random_state)

trainer = test(
    train_loader=cifar_train_loader,
    val_loader=cifar_val_loader,
    test_loader=cifar_test_loader,
    model = custom_model,
    model_name=model_name,
    device=device,
    ckpt_path=yml['ckpt_path'],
    load_saved_model=True
)
