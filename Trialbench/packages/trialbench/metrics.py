from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(y_true, y_pred):
    return {
        'auc_macro': roc_auc_score(y_true, y_pred, average='macro'),
        'auc_weighted': roc_auc_score(y_score=y_pred, y_true=y_true, average='weighted'),
    }