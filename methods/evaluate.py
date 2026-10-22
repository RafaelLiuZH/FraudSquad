import numpy as np
import sklearn.metrics as skme

def top_pred(prob, top_ratio=0.05):
    pred = np.zeros_like(prob)
    topn = int(len(prob)*top_ratio)
    pred[np.argsort(prob)[-topn:]] = 1
    return pred
    
def evaluate(probs, labels, top_ratio=0.05, score_thre=None):
    auc = skme.roc_auc_score(y_score=probs, y_true=labels)
    ap = skme.average_precision_score(y_score=probs, y_true=labels)
    if top_ratio:
        preds = top_pred(probs, top_ratio)
    elif score_thre:
        preds = probs>score_thre
    precision = skme.precision_score(y_pred=preds, y_true=labels)
    recall = skme.recall_score(y_pred=preds, y_true=labels)
    f1 = skme.f1_score(y_pred=preds, y_true=labels)
    return auc, ap, precision, recall, f1

def print_latex_df(result_df, file=None):    
    latex_df = result_df.copy()
    latex_df[['Precision', 'Recall', 'AP','AUC']] = latex_df[['Precision', 'Recall', 'AP','AUC']].round(4)*100
    del latex_df['AP']
    if file:
        with open(file, 'w') as f:
            f.write(latex_df[['Node Emb.', 'Engineered features', 'Precision', 'Recall','AUC']].to_latex( float_format="%.2f"))
    else:
        print(latex_df[['Node Emb.', 'Engineered features', 'Precision', 'Recall','AUC']].to_latex(float_format="%.2f"))
    return latex_df