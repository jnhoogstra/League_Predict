# functions file for reusing functions accross notebooks

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


# 
def ScoreModel(model, X, y):
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    recall = recall_score(y, preds)
    precision = precision_score(y, preds)
    rockout = roc_auc_score(y, preds)
    
    print("Accuracy:  ", acc)
    print("F1 Score:  ", f1)
    print("Recall:    ", recall)
    print("Precision: ", precision)
    print("ROC_AUC:   ", rockout)

    
def FeatureImp(model, X):
    feature = list(zip(X.columns, 100*(np.round(boost_model_new.feature_importances_, 4))))
    return feature
