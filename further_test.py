from pandas import read_csv
from sklearn.metrics import accuracy_score

# ======================================================================================================================
# Further tests on B's prediction
B_df = pd.read_csv('./B.csv')
y_true_B = B_df['label']
y_pred_B = B_df['predict_label']
y_score_B = B_df['score']
print(accuracy_score())

# ======================================================================================================================
# Further tests on B's prediction
D_df = pd.read_csv('./D.csv')
y_true_D = D_df['label']
y_pred_D = D_df['predict_label']
y_score_D = D_df['score']