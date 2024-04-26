import numpy as np
from sklearn.metrics import average_precision_score,confusion_matrix,accuracy_score
from sklearn.metrics import roc_auc_score
def metric(y_pred,y_true,confidence=0.5):
    try:
        ap = average_precision_score(y_true,y_pred)
    except:
        print(f'Maybe only one class in t_pred. So we will set the ap to -1.')
        ap = -1
        
        
    try:
        roc = roc_auc_score(y_true, y_pred)
    except:
        print(f'We set the roc to -1 too.')
        roc = -1
        
    y_pred = np.where(np.array(y_pred) >= confidence,1,0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    acc = (tp + tn) / (tp + tn + fp + fn)


    return acc,fnr,fpr,ap,roc


def csl_metric(y_pred,y_true):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    return fnr,fpr,acc
    