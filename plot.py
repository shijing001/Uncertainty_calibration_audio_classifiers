
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def xx_plot(confidence, accuracy): 
    confidence = np.array(confidence)
    accuracy = np.array(accuracy)

    path = "D:/document"
    
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
    plt.bar(barx, acc, width=0.5/bins,color='#84B0FD',label="Outputs") 
    plt.bar(barx,barx,width=0.5/bins,color='#DFA8A4', label='Except',alpha=0.5) 
    plt.xlabel('Confidence') 
    plt.ylabel('Accuracy') 
    plt.title('ECE-20') 
    plt.legend() 
    plt.savefig(path) 

def ye_plot(data, acc):

    plt.figure(figsize=(32, 32))
    fig, ax1 = plt.subplots()
    ax1.hist(data[np.where(acc == 1)],
             bins=np.linspace(0, 1, num=51),
             label="success",
             density=True)
    ax1.hist(data[np.where(acc == 0)],
             bins=np.linspace(0, 1, num=51),
             label="fail",
             alpha=0.5,
             density=True)
    ax1.legend()
    plt.show()
labels = pd.read_csv('D:/document/Audio-Classification/checkpoint_dir/label.csv',header=0,index_col=0)
predict = pd.read_csv('D:/document/Audio-Classification/checkpoint_dir/result.csv',header=0,index_col=0)

predicted = np.max(predict.values,axis=1)
preds_labels = np.argmax(predict.values,axis=1)
acc = (preds_labels==labels.values.reshape(200,)).astype(np.int16)

xx_plot(predicted, acc)

