import numpy as np

def perturb(df, col_list, mean, std_fix = 0, std_spec = 0, seed=None):
    # df must be pandas df, not np array
    df = df.copy()
    if seed is not None:
        np.random.seed(seed)
    for col in col_list:
        if std_spec:
            df[col] = df[col] + np.random.normal(mean, np.std(df[col])*std_spec, len(df))
        else:
            df[col] = df[col] + np.random.normal(mean, std_fix, len(df))
    return df


def PGI(data, model, col_list, mean, std_fix = 0, std_spec = 0, seed=None):
    data_perturb = perturb(data, col_list, mean, std_fix, std_spec, seed)
    y_pred = model.predict(data)
    y_pred_perturb = model.predict(data_perturb)
    return np.mean(np.abs(y_pred_perturb - y_pred))


from sklearn.metrics import auc
import matplotlib.pyplot as plt

# def auc_PGI(feature_list, X_train, model, mean = 0, std_fix = 0, std_spec = 0, seed=None, label = False):
#     PGI_list = []
#     for k in range(len(feature_list)):
#         p = PGI(X_train, model, feature_list[:k], mean, std_fix = std_fix, std_spec = std_spec, seed=None)
#         PGI_list.append(p)
#     plt.plot(PGI_list, la)
#     plt.xlabel('Number of Features')
#     #plt.xticks(range(len(PGI_list)), feature_list, rotation=90)
#     plt.xticks(range(len(PGI_list)), list(range(1,19)))

#     print(auc(range(len(PGI_list)), PGI_list))
#     if label:
#         plt.legend()

def auc_PGI(feature_list, data, model, mean = 0, std_fix = 0, std_spec = 0, seed=None, label = None, color = None):
    PGI_list = []
    for k in range(len(feature_list)):
        p = PGI(data, model, feature_list[:k], mean, std_fix = std_fix, std_spec = std_spec, seed=None)
        PGI_list.append(p)
    a = (auc(range(len(PGI_list)), PGI_list))
    if color:
        plt.plot(PGI_list, label = "{} (AUC: {:.3f})".format(label, a), color = color)
    else:
        plt.plot(PGI_list, label = "{} (AUC: {:.3f})".format(label, a))
    plt.xlabel('Number of Features')
    plt.xticks(range(len(PGI_list)), list(range(1,19)))
    if label:
        plt.legend()




