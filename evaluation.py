import numpy as np
from regex import F
from sklearn.metrics import roc_auc_score
import pickle
import os

def get_f1(tp, fp, fn):
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2.0 * recall * precision / (recall + precision)

def evaluate(prediction, gt_inliner, threshold):
    
    y = np.greater(prediction, threshold)   # positive
    
    gt_outlier = np.logical_not(gt_inliner)
    
    tp = np.sum(np.logical_and(y,gt_inliner))
    fp = np.sum(np.logical_and(y,gt_outlier))
    tn = np.sum(np.logical_and(np.logical_not(y),gt_outlier))
    fn = np.sum(np.logical_and(np.logical_not(y),gt_inliner))
    total_count = tp + tn + fp + fn
    
    # accuracy
    
    accuracy = 100 * (tp + tn) / total_count    
    
    # f1
    f1 = get_f1(tp,fp,fn)
    
    #AUC
    
    try:
        auc = roc_auc_score(gt_inliner,prediction)  
    except:
        auc = 0
        
    X1 = [x[1] for x in zip(gt_inliner,prediction) if x[0]]
    Y1 = [x[1] for x in zip(gt_inliner,prediction) if not x[0]]
    
    minP = min(prediction) - 1
    maxP = max(prediction) + 1
    
    # FPR at TPR 95
    
    fpr95 = 0.0
    clothest_tpr = 1.0 
    dist_tpr = 1.0
    for threshold in np.arange(minP,maxP,0.2):
        tpr = np.sum(np.greater_equal(X1,threshold)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1,threshold)) / np.float(len(Y1))
        if abs(tpr - 0.95) < dist_tpr:
            dist_tpr = abs(tpr - 0.95)
            clothest_tpr = tpr
            fpr95 = fpr
            
    # Detection error
    error = 1.0
    for threshold in np.arange(minP,maxP,0.2):
        tpr = np.sum(np.less(X1,threshold)) / np.float(len(X1))
        fpr = np.sum(np.greater_equal(Y1,threshold)) / np.float(len(Y1))
        error = np.minimum(error, (tpr + fpr) / 2.0)    # should weighted average?
        
    # AUPR IN
    auprin = 0.0
    recallTemp = 1.0
    for threshold in np.arange(minP,maxP,0.2):
        tp = np.sum(np.greater_equal(X1,threshold))
        fp = np.sum(np.greater_equal(Y1,threshold))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(X1))
        auprin += (recallTemp - recall) * precision
        recallTemp = recall
    auprin += recall * precision
    
    # AUPR OUT
    minP, maxP = -maxP, -minP
    X1 = [-x for x in X1]
    Y1 = [-x for x in Y1]
    auprout = 0.0
    recallTemp = 1.0
    for threshold in np.arange(minP, maxP, 0.2):
        tp = np.sum(np.greater_equal(Y1, threshold))
        fp = np.sum(np.greater_equal(X1, threshold))
        if tp + fp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / np.float(len(Y1))
        auprout += (recallTemp - recall) * precision
        recallTemp = recall
    auprout += recall * precision    
    
    with open(os.path.join("results.txt"), "a") as file:
        file.write(
            "# retrained \n"
            "Accuracy: %f\n F1: %f\n AUC: %f\nfpr95: %f"
            "\nDetection: %f\nauprin: %f\nauprout: %f\n\n" %
            (accuracy, f1, auc, fpr95, error, auprin, auprout))   
        
    return dict(auc=auc, f1=f1, fpr95=fpr95, error=error, auprin=auprin, auprout=auprout)