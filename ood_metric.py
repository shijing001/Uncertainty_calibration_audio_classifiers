from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd
import os
import numpy as np
import pdb

dataset_name = 'ESC'
data_dir = "/workspace/esc_focal_checkpoint_dir/"
ood_dataset_name ='GTZAN'
ood_data_dir = "/workspace/esc_gtzan/"
model = 'densenet'



for i in range(5):
    logit = pd.read_csv(os.path.join(data_dir,'{}_{}_result_{}.csv'.format(dataset_name,model,i)),header=0,index_col=0)
    ood_logit = pd.read_csv(os.path.join(ood_data_dir,'{}_{}_result_{}.csv'.format(ood_dataset_name,model,i)),header=0,index_col=0)
    # pdb.set_trace()
    all_logit = np.concatenate((logit,ood_logit),axis=0)
    predict = np.max(all_logit,axis=1)
    target_label = np.append(np.ones(len(logit)),np.zeros(len(ood_logit)))
    roc_auc = roc_auc_score(target_label, predict)
    # Data to plot precision - recall curve
    precision, recall, _ = precision_recall_curve(target_label, predict)
    # Use AUC function to calculate the area under the curve of precision recall curve
    aupr = auc(recall, precision)
    # print('-------start print-------------')
    print(roc_auc)
    print(aupr)
