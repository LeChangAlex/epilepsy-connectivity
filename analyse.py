import mne
import os
import matplotlib.pyplot as plt
import numpy as np
# import spectrum
import pandas as pd
import scipy
import scipy.io
from tqdm import tqdm
# import torch
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn import linear_model
import sklearn.metrics
from sklearn import svm 
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import PrecisionRecallDisplay

# ======= plotting =========
class PCAModel:
    # TODO: fit k hyperparameter, compute auprc for those


    def __init__(self, model, k=5):
        self.model = model
        self.pca_model = PCA(n_components=k)

    def fit(self, X, y):

        self.pca_model.fit(X)
        embs = self.pca_model.transform(X)
        self.model.fit(embs, y)

    def predict(self, X):
        embs = self.pca_model.transform(X)
        return self.model.predict(embs)

def plot_matrices(mat):

    for i in range(5):
        for j in range(5):
            
            # plot a freq_band - freq_band matrix
            plt.matshow(mat[:, :, i, j])
            plt.show()
            


def plot_flow(mat, ch_names, flow_direction="out"):

    n_channels = mat.shape[0]
    
    fig, axs = plt.subplots(5, 5)
    fig.set_figheight(20)
    fig.set_figwidth(20)

    

    for i in range(5):
        for j in range(5):

            # outflow -> axis=0 sums ij for all j aka entire row
            axis = 0 if flow_direction == "out" else "in"
            outflow_arr = np.sum(mat[:, :, i, j], axis=0)

            axs[i][j].bar(np.arange(n_channels), outflow_arr)

            top_idx =(-outflow_arr).argsort()[:10]

            for k in top_idx:
                axs[i][j].text(k, outflow_arr[k], ch_names[k])

    plt.xticks(rotation=90)
    plt.show()



def plot_patient_flow(mat, ch_names, flow_direction="out"):

    """
    mat: n_seizure x n_channels x n_channels x 5 x 5 array
    """
    n_channels = mat.shape[0]
    axis = 1 if flow_direction == "out" else 2

    

    for i in range(5):
        for j in range(5):

            # outflow -> axis=0 sums ij for all j aka entire row
            outflow_arr = np.sum(mat[:, :, :, i, j], axis=axis)

            inflow_arr = np.sum(mat[:, :, :, i, j], axis=axis)



            plt.boxplot(outflow_arr)

            top_idx =(-outflow_arr).argsort()[:10]

            # for k in top_idx:
            #     plt.text(k, outflow_arr[k], ch_names[k])

            plt.xticks(rotation=90)
            plt.show()


def scatter_flow(X, y, z_electrode=None, cluster="pca", p=0): # , z_patient):

    if not z_electrode:
        z_electrode = np.ones_like(y)
    # fig, axs = plt.subplots()
    # fig.set_figheight(10)
    # fig.set_figwidth(10)

    print("fitting PCA...")
    if cluster == "pca":
        pca= PCA(n_components=2)
        emb_X = pca.fit_transform(X)
    elif cluster == "tsne":
        tsne = TSNE(n_components=2, perplexity=10, metric="euclidean")
        emb_X = tsne.fit_transform(X)


    for z in np.unique(z_electrode):
        # plt.scatter(emb_X[z_electrode == z, 0], emb_X[z_electrode == z, 1], label=z, s=10, marker=r'.')
        plt.scatter(emb_X[np.logical_and(z_electrode == z, y == 0), 0], emb_X[np.logical_and(z_electrode == z, y == 0), 1], label=z, s=10, marker=r'x')
    
    plt.scatter(emb_X[y == 1, 0], emb_X[y == 1, 1], label="soc", s=20, marker=r'.', c="red")
    plt.title("patient: {}".format(p))
    plt.legend()
    plt.xticks(rotation=90)
    plt.show()


def plot_electrode_bar(preds, y, z):
    unique_z = np.unique(z)
    print(len(preds))
    for i in range(len(unique_z)):
        
        normal = preds[np.logical_and(z == unique_z[i], y == 0)]
        soc = preds[np.logical_and(z == unique_z[i], y == 1)]

        print(np.arange(len(normal)).shape, normal.shape)
        # plt.bar(np.arange(len(normal)), normal[:, 1], color="b")
        # plt.bar(np.arange(len(normal), len(normal) + len(soc)), soc[:, 1], color="r")
        plt.bar(np.arange(len(normal)), normal, color="b")
        plt.bar(np.arange(len(normal), len(normal) + len(soc)), soc, color="r")

        plt.show()
        print(normal)


        


def classify_flow(X, y, z, zs, n_test=1, shuffle=False, model_type="lg", model_pca=False, k=5):
    
    # model = svm.SVC(kernel='linear', degree=2,  C=1, random_state=0)
    if model_type == "svm":
        model = svm.SVC(kernel='linear', C=1, random_state=0)
    elif model_type == "lg":
        model = linear_model.LogisticRegression(penalty="l1", C=1, random_state=0, max_iter=1000, solver="saga")
    elif model_type == "nn":
        model = MLPClassifier(hidden_layer_sizes=(100), max_iter=2000, random_state=0)
    
    if model_pca:
        model = PCAModel(model, k=k)

    test_preds = []
    
    if shuffle:
        X, y = sklearn.utils.shuffle(X, y)

    unique_z = sorted(np.unique(z))

    for i in tqdm(range(0, len(unique_z))):

        train_idx = z != unique_z[i]


        X_train = X[train_idx]
        y_train = y[train_idx]

        X_test = X[z == unique_z[i]]
        y_test = y[z == unique_z[i]]

        model.fit(X_train, y_train)

        # y_pred = model.predict_proba(X_test)
        y_pred = model.predict(X_test)

        test_preds.append(y_pred)

        print(sklearn.metrics.classification_report(y_test, y_pred))
        auprc = sklearn.metrics.average_precision_score(y_test, y_pred)
        print("auprc: ", auprc)
            
        # print(model.coef_)

    test_preds = np.concatenate(test_preds)

    # average predictions by channel and patient
    # test_preds, y, _ = patient_mean(test_preds, y, z, zs, mean=True)

    # plot_electrode_bar(test_preds, y, z)

    # test_preds = test_preds >= 0.5
    auprc = sklearn.metrics.average_precision_score(y, test_preds)
    print(sklearn.metrics.classification_report(y, test_preds))
    print("auprc: ", auprc)
            

    return test_preds, y

def extract_flows(fn, seizure_onset_channels, bad_channels, preprocess=True, normalize=True, add_freqs=False):

    connectivity_mat = scipy.io.loadmat("results/{}".format(fn))["bspec_matrix"]
    ch_names = pd.read_csv("channel_names/{}.csv".format(fn[:-8]))["0"].tolist()
    

    bad_idx = []
    for ch in bad_channels:
        if ch != "":
            bad_idx.append(ch_names.index(ch))

    connectivity_mat = np.delete(connectivity_mat, bad_idx, 0)
    connectivity_mat = np.delete(connectivity_mat, bad_idx, 1)
    
    # print(ch_names)

    ch_names = list(np.delete(ch_names, bad_idx))
    
    # print(ch_names)
    
    outflow = np.sum(connectivity_mat, axis=0)
    inflow = np.sum(connectivity_mat, axis=1)

    if add_freqs:
        outflow = np.sum(outflow, axis=-1)
        inflow = np.sum(inflow, axis=-1)
    
         
    outflow = outflow.reshape(outflow.shape[0], -1)
    inflow = inflow.reshape(inflow.shape[0], -1)
    inflow_sum = np.sum(inflow, axis=-1)[..., np.newaxis]
    outflow_sum = np.sum(outflow, axis=-1)[..., np.newaxis]
    X = np.concatenate((inflow, outflow, inflow_sum, outflow_sum), axis=1)

    soc_idx = []
    for soc in seizure_onset_channels:
        if soc in ch_names:
            soc_idx.append(ch_names.index(soc))


    y = np.zeros(outflow.shape[0])
    y[soc_idx] = 1

    # if preprocess:
    #     print(X.shape)
    #     # p1 = np.percentile
    #     # p99


       

    return X, y


def my_replace(s):
    res = []
    for ch in s:
        if ch in ['́']:
            continue
        res.append(ch)

    return "".join(res)


def patient_mean(X, y, zp, zs, mean=False):
    unique_zp = np.unique(zp)

    out_X = []
    out_y = []
    out_z = []

    for i in range(len(unique_zp)):
        unique_zs = np.unique(zs)

        X_p = X[zp == unique_zp[i]] # choose X of patient
        zs_p = zs[zp == unique_zp[i]] # choose electrodes of patient
        y_p = y[zp == unique_zp[i]]

        means_X = []
        means_y = []

        unique_zs_p = np.unique(zs_p)
        for j in range(len(unique_zs_p)):
            if mean:
                mean_X = np.mean(X_p[zs_p == j], axis=0)
            else:
                mean_X = X_p[zs_p == j] - np.mean(X_p[zs_p == j], axis=0)

                mean_X = mean_X / (np.std(mean_X, axis=0) + 1e-3)

            mean_y = np.mean(y_p[zs_p == j])



            means_X.append(mean_X)
            means_y.append(mean_y)


        out_X.append(np.array(means_X))
        out_y.append(np.array(means_y))
        out_z.append(np.ones(len(means_X)) * unique_zp[i])

    
    out_X = np.concatenate(out_X)
    out_y = np.concatenate(out_y)
    out_z = np.concatenate(out_z)

    return out_X, out_y, out_z


def patient_normalize(X, zp):
    """
    normalize each electrode of X within each patient
    """
    unique_zp = np.unique(zp)
    all_X = []

    for i in range(len(unique_zp)):
        X_p = X[zp == unique_zp[i]]
        X_p = X_p - np.mean(X_p, axis=0)
        X_p = X_p / np.std(X_p, axis=0)

        all_X.append(X_p)

    all_X = np.concatenate(all_X)
    return all_X

if __name__ == "__main__":

    results_dir = "results_timelapse"

    group = "patient"
    # group = "seizure"

    normalize=False
    preprocess=False

    shuffle=False
    add_freqs=False
    model_pca = False
    k = 10

    cluster="pca"
    # model_type = "nn"
    # model_type = "svm"
    model_type = "lg"

    metadata = pd.read_csv("metadata.csv")
    metadata.fillna("", inplace=True)

    metadata["bad_channels"] = metadata["bad_channels"].apply(lambda x: x.strip(",").split(","))
    metadata["seizure_onset_channels"] = metadata["seizure_onset_channels"].apply(lambda x: x.strip(",").split(","))
    metadata["resected_channels"] = metadata["resected_channels"].apply(lambda x: x.strip(",").split(","))
    

    # metadata["long_fn"] = metadata["long_fn"].apply(lambda x: x.split("/")[-1], 1)
    # metadata["long_fn"] = metadata["long_fn"].apply(lambda x: x.split("/")[-1], 1)


    a = os.listdir(results_dir)[2].split(".")[0]
    # print(a, a in list(metadata["long_fn"]), "--------------------")


    all_X = []
    all_y = []
    all_zp = []
    all_ze = []
    all_zs = []

    n = 0
    for i in range(len(metadata["long_fn"])):
        
        lfn = metadata["long_fn"][i]
        for j in range(len(os.listdir(results_dir))):
            fn = os.listdir(results_dir)[j]
            fixed_fn = my_replace(fn)

            if lfn != "" and lfn in fixed_fn:


                # label_channels = list(metadata["seizure_onset_channels"][i])
                label_channels = list(metadata["resected_channels"][i])
                bad_channels = list(metadata["bad_channels"][i])
                
                if not label_channels[0]:
                    continue

                X, y = extract_flows(fn, label_channels, bad_channels, preprocess=preprocess, normalize=normalize, add_freqs=add_freqs)
    
                z_patient = np.ones(X.shape[0]) * metadata["patient_n"][i]
                z_electrode = np.arange(X.shape[0])
                z_seizure = np.ones(X.shape[0]) * n

                #classify_flow(X, y)


                all_X.append(X)
                all_y.append(y)
                all_zp.append(z_patient)
                all_ze.append(z_electrode)
                all_zs.append(z_seizure)

                n += 1


    all_X = np.concatenate(all_X)
    all_y = np.concatenate(all_y)
    all_zp = np.concatenate(all_zp)
    all_ze = np.concatenate(all_ze)
    all_zs = np.concatenate(all_zs)


    # Calculate AUPRC per freq coupling and patient




    # p95 = np.percentile(all_X, q=95, axis=0)
    # for i in range(len(p95)):
    #     all_X[all_X[:, i] > p95[i], :] > p95[i]

    # average electrode for each patient
    all_X, all_y, all_zp = patient_mean(all_X, all_y, all_zp, all_ze, mean=True)

    if normalize:
        all_X = patient_normalize(all_X, all_zp)


    unique_patient = np.unique(all_zp)

    for p in range(len(unique_patient)):
        auprc_list = []

        yp = all_y[all_zp == unique_patient[p]]
        Xp = all_X[all_zp == unique_patient[p]]

        
        for i in range(all_X.shape[1]):

            Xpi = all_X[all_zp == unique_patient[p], i]
            # print(yp, Xpi)
            auprc = sklearn.metrics.average_precision_score(yp, Xpi)

            auprc_list.append((unique_patient[p], i, auprc))
        
        mat = np.array(auprc_list[:-2])[:, 2].reshape(5, -1)
        
        # f, (ax1, ax2) = plt.subplots(2, 1)
        # mat_in = ax1.matshow(mat[:, :5], vmin=0, vmax=1, cmap='seismic', label="inflow")
        # mat_out = ax2.matshow(mat[:, 5:], vmin=0, vmax=1, cmap='seismic', label="outflow")
        # plt.colorbar(mat_in)
        # plt.title("patient {}".format(unique_patient[p]), loc="left")
        # plt.show()

        scatter_flow(Xp, yp, cluster=cluster, p=unique_patient[p])

        # s_list = sorted(auprc_list, key=lambda x: -x[2])
        # for i in range(len(s_list)):
        #     f1 = s_list[i][1] // 5
        #     f2 = s_list[i][1] % 5

        #     if i < 5:
        #         e = s_list[i][1]
        #         display = PrecisionRecallDisplay.from_predictions(yp, Xp[:, e], name="patient {} f1:{} f2:{} auprc:{}".format(
        #             unique_patient[p], f1, f2, s_list[i][2]))
        #         # display.plot()
        #     print(s_list[i], f1, f2)

        

    classify_flow(all_X, all_y, all_zp, all_ze, n_test=200, shuffle=False, model_type=model_type, model_pca=model_pca)


    # dim reduct visualization
    # scatter_flow(all_X, all_y, all_zp, cluster="pca")
    # scatter_flow(all_X, all_y, all_zp, cluster="tsne")

    # TODO: each patient has clustered seizures, 
    # see how patients can generalize to others
    # clip and preprocess outflow/inflow p1 p99


    # group seizures by patient
    patient_seizure_d = {}
    for fn in os.listdir("results"):
        p = fn.split("_")[0]
        
        if p not in patient_seizure_d:
            patient_seizure_d[p] = []
        patient_seizure_d[p].append(fn)



    # for fn in sorted(os.listdir("channel_names")):
    #     print(fn)
    #     if fn[0] == ".":
    #         continue
    #     mat = scipy.io.loadmat("results/{}.npy.mat".format(fn[:-4]))["bspec_matrix"]
    #     ch_names = pd.read_csv("channel_names/{}".format(fn))["0"].tolist()


        
    #     plot_patient_flow(mat, ch_names)
    #     # plot_flow(mat, ch_names)
        









