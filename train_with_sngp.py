import torch
import torchvision
import torch.nn as nn
import numpy as np
import json
import utils
import validate
import argparse
import models.densenet
import models.densenetsn
import models.resnet
import models.resnetsn
import models.sngp
import models.inception
import models.inceptionsn
import time
import dataloaders.datasetaug
import dataloaders.datasetnormal
import pdb
import os
from tqdm import tqdm
import pandas as pd
from tensorboardX import SummaryWriter
from torchsummary import summary
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
parser = argparse.ArgumentParser()
parser.add_argument("--config_path", default='/workspace/config/gtzan_resnet.json',type=str)

def xx_plot(prediction, labels,path): 
    
    prediction = np.array(prediction)
    labels = np.array(labels)
    
    confidence = np.max(prediction,axis=1)
    preds_labels = np.argmax(prediction,axis=1)

    accuracy = (preds_labels==labels.T).astype(np.int16)
    
    bins = 10 
    acc = [] 
    for i in range(bins): 
        part = np.where((confidence >= (i / bins)) & (confidence < ((i+1) / bins))) 
        part_acc = accuracy[part] 
        accBm = len(np.where(part_acc==1)[0])/len(part_acc) if len(part_acc) > 0 else 0 
        acc.append(accBm) 
    plt.figure() 
    x = np.arange(0,1,0.02) 
    barx = np.arange(1/(2*bins),1,1/bins) 
    ax = plt.gca() 
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1)) 
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.1)) 
    plt.xlim(-0.05,1) 
    plt.plot(x,x,color='k',linewidth=0.5,linestyle='--') 
    plt.bar(barx, acc, width=0.5/bins,color='#0000ff',label="Outputs") 
    plt.bar(barx,barx,width=0.5/bins,color='#fbd5d5', label='Gap',alpha=0.5) 
    plt.xlabel('Confidence') 
    plt.ylabel('Accuracy') 
    plt.legend() 
    plt.savefig(path) 

def ECE(py, y_test, n_bins=10):
    
    py = np.array(py)
    y_test = np.array(y_test)
    
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)

def mean_field_logits(logits,
                      covmat=None,
                      mean_field_factor=1.,
                      likelihood='logistic'):
    """Adjust the model logits so its softmax approximates the posterior mean [1].
    Arguments:
    logits: A float tensor of shape (batch_size, num_classes).
    covmat: A float tensor of shape (batch_size, batch_size). If None then it
      assumes the covmat is an identity matrix.
    mean_field_factor: The scale factor for mean-field approximation, used to
      adjust the influence of posterior variance in posterior mean
      approximation. If covmat=None then it is used as the scaling parameter for
      temperature scaling.
    likelihood: Likelihood for integration in Gaussian-approximated latent
      posterior.
    Returns:
    True or False if `pred` has a constant boolean value, None otherwise.
    """
    if likelihood not in ('logistic', 'binary_logistic', 'poisson'):
        raise ValueError(
            f'Likelihood" must be one of (\'logistic\', \'binary_logistic\', \'poisson\'), got {likelihood}.'
        )

    if mean_field_factor < 0:
        return logits

    # Compute standard deviation.
    if covmat is None:
        variances = 1.
    else:
        variances = torch.diagonal(covmat)

    # Compute scaling coefficient for mean-field approximation.
    if likelihood == 'poisson':
        logits_scale = torch.exp(- variances * mean_field_factor / 2.)
    else:
        logits_scale = torch.sqrt(1. + variances * mean_field_factor)

    # Cast logits_scale to compatible dimension.
    if len(logits.shape) > 1:
        logits_scale = torch.unsqueeze(logits_scale, dim=-1)

    return logits / logits_scale


def train(model, device, data_loader, optimizer, loss_fn):
    model.train()
    loss_avg = utils.RunningAverage()

    with tqdm(total=len(data_loader)) as t:
        for batch_idx, data in enumerate(data_loader):
            inputs = data[0].to(device)
            # print(inputs.shape)
            target = data[1].squeeze(1).to(device)
            # pdb.set_trace()
            outputs = model(inputs,update_cov=True)
            
            loss = loss_fn(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()
            
    return loss_avg()


def train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, split, scheduler=None):
    best_acc = 0.0

    for epoch in range(params.epochs):
        avg_loss = train(model, device, train_loader, optimizer, loss_fn)

        acc = validate.evaluate(model, device, val_loader)
        print("Epoch {}/{} Loss:{} Valid Acc:{}".format(epoch, params.epochs, avg_loss, acc))

        is_best = (acc > best_acc)
        if is_best:
            best_acc = acc
        if scheduler:
            scheduler.step()

        utils.save_checkpoint({"epoch": epoch + 1,
                               "model": model.state_dict(),
                               "optimizer": optimizer.state_dict()}, is_best, split, "{}".format(params.checkpoint_dir))
        
        writer.add_scalar("data{}/trainingLoss{}".format(params.dataset_name, split), avg_loss, epoch)
        writer.add_scalar("data{}/valLoss{}".format(params.dataset_name, split), acc, epoch)
        model.reset_cov()
    writer.close()


def test_sngp(model, device, test_loader):
    correct = 0
    total = 0
    model.eval()
    
    checkpoint = torch.load(os.path.join(params.checkpoint_dir,'last{}.pth.tar'.format(params.model)))
    
    model.load_state_dict(checkpoint["model"])
    y_real=torch.randn([0]).to(device)
    
    y_prediction=torch.randn([0]).to(device)
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs = data[0].to(device)
            target = data[1].squeeze(1).to(device)
            
            logit, cov = model(inputs,return_gp_cov=True, update_cov=False)
            logits = mean_field_logits(logit, cov, mean_field_factor=0.1)
            
            outputs = nn.functional.softmax(logits,dim=1).to(device)
            
            y_real=torch.cat([y_real,target],dim=0)
            
            y_prediction = torch.cat([y_prediction,outputs],dim=0)

            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        
        
        ece = ECE(y_prediction.cpu(),y_real.cpu()) 
        acc = 100*correct/total
        
        path = os.path.join(params.checkpoint_dir,'fig_{}_{}_ece.jpg'.format(params.dataset_name,params.model))
        fig = xx_plot(y_prediction.cpu().data.cpu(),y_real.data.cpu(),path)
        

        print('acc:{:.4f}'.format(acc))
        print('ece:{:.4f}'.format(ece))
        

        pd.DataFrame(np.array(y_prediction.cpu())).to_csv(os.path.join(params.checkpoint_dir,'{}_{}_result.csv'.format(params.dataset_name,params.model)))
        pd.DataFrame(np.array(y_real.cpu())).to_csv(os.path.join(params.checkpoint_dir,'{}_{}_label.csv'.format(params.dataset_name,params.model)))


if __name__ == "__main__":
    args = parser.parse_args()
    params = utils.Params(args.config_path)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    num_classes = 50 if params.dataset_name=="ESC" else 10


    if params.dataaug:
        train_loader = dataloaders.datasetaug.fetch_dataloader( "{}training128mel.pkl".format(params.data_dir), params.dataset_name, params.batch_size, params.num_workers, 'train')
        val_loader = dataloaders.datasetaug.fetch_dataloader("{}validation128mel.pkl".format(params.data_dir), params.dataset_name, params.batch_size, params.num_workers, 'validation')
        test_loader = dataloaders.datasetaug.fetch_dataloader("{}test128mel.pkl".format(params.data_dir), params.dataset_name, params.batch_size, params.num_workers, 'test')
    else:
        train_loader = dataloaders.datasetnormal.fetch_dataloader( "{}training128mel.pkl".format(params.data_dir), params.dataset_name, params.batch_size, params.num_workers)
        val_loader = dataloaders.datasetnormal.fetch_dataloader("{}validation128mel.pkl".format(params.data_dir), params.dataset_name, params.batch_size, params.num_workers)
        test_loader = dataloaders.datasetnormal.fetch_dataloader("{}test128mel.pkl".format(params.data_dir), params.dataset_name, params.batch_size, params.num_workers, 'test')
   
    writer = SummaryWriter(comment=params.dataset_name)
    if params.model=="densenetsn":
        backbone = models.densenetsn.densenet201(num_classes=num_classes).to(device) 
        model = models.sngp.SNGP(backbone,hidden_size=1920, num_classes=num_classes,epochs=params.epochs, device=device).to(device)   
    elif params.model=="resnetsn":
        backbone = models.resnetsn.resnet50(num_classes=num_classes).to(device) 
        model = models.sngp.SNGP(backbone,hidden_size=2048, num_classes=num_classes,epochs=params.epochs, device=device).to(device)
    elif params.model=="inceptionsn":
        backbone = models.inceptionsn.inception_v3(num_classes=num_classes, aux_logits=False).to(device) 
        model = models.sngp.SNGP(backbone,hidden_size=2048, num_classes=num_classes,epochs=params.epochs, device=device).to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)

    if params.scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    else:
        scheduler = None
    
    print('--------------start training-----------------')
    train_and_evaluate(model, device, train_loader, val_loader, optimizer, loss_fn, writer, params, params.model, scheduler)
    print('--------------start testing-----------------')
    test_sngp(model, device, test_loader)

