from sklearn.mixture import GaussianMixture
from kneed import KneeLocator
import numpy as np
from NN_model import SCAMP
import torch

'''
PARAMETERS
model : SCAMP model loaded from .pt file
input : copy number list
n_components : if known, the number of components. Set to None to let kneed find
one_thresh : if n_components is None, uses this to determine when the knee is at 1 
max_components : if n_components is None, checks up to these components
genes : gene name
decision_rule : scAmp NN threshold for classifying ecDNA status
kneedle_coeff : higher means kneedle tends to smaller distributions

RETURNS
Boolean, the predicted ecDNA status
'''
def GMM_ecDNA_predict(model, input, 
                      n_components = None, 
                      one_thresh = 1.15, 
                      max_components = 15,
                      genes = np.array(['GENE']), 
                      decision_rule = 0.5,
                      kneedle_coeff = 1.3
                      ) :
    
    input = np.array(input).reshape(-1, 1)
    if n_components is None :
        bic_scores = []
        aic_scores = []
        ks = []
        for k in range(1, max_components):
            gmm = GaussianMixture(n_components=k, random_state=0)
            gmm.fit(input)

            bic_scores.append(gmm.bic(input))
            aic_scores.append(gmm.aic(input))
            ks.append(k)

        if bic_scores[0]/np.min(bic_scores) < one_thresh :
            n_components = 1
            print(f"Number of distributions (Kneed): {n_components}")

        else :

            bic_scaled = (bic_scores - (np.min(bic_scores))) / np.ptp(bic_scores)
            bic_scaled = bic_scaled ** kneedle_coeff
            # bic_scaled = np.log(np.array(bic_scores) - np.min(bic_scores) + 1e-8)
            
            kneedle = KneeLocator(
                ks,
                bic_scaled,
                curve="convex",      # or "concave" depending on your curve
                direction="decreasing"
            )

            n_components = kneedle.knee
            print(f"Number of distributions (Kneed): {n_components}")

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(input)

    for i in range(gmm.n_components):
        mean = gmm.means_[i, 0]
        var = gmm.covariances_[i, 0, 0]
        # print(f"Mean: {mean}, Var: {var}")

        samples = np.random.normal(mean, np.sqrt(var), size=1000).tolist()
        if NN_ecDNA_predict(model, samples, genes, decision_rule) :
            return True

    return False



'''
PARAMETERS
model : SCAMP model loaded from .pt file
input : copy number list
genes : gene name
decision_rule : threshold for classifying ecDNA status
log_results : Whether or not to print results

RETURNS
Boolean, the predicted ecDNA status
'''
def NN_ecDNA_predict(model, input, genes = np.array(['GENE']), decision_rule = 0.5, log_results = False) :
    input = np.array(input).reshape(-1, 1)

    X, genes_pass_filter = model.prepare_copy_numbers(
        input,
        genes,
        min_copy_number=2.0,
        max_percentile=99.0,
        filter_copy_number=2.5,
    )


    if len(genes_pass_filter) == 0 :
        if log_results :
            print("No genes passed filter")
        return False

    # Just get the second to last layer
    _x = model.forward(torch.Tensor(X))
    _x = _x[0]
    ecDNA_pred = _x[1]/(_x[0] + _x[1])
    if log_results :
        print(f"Node 1: {_x[0]}, Node 2: {_x[1]}, prediction: {ecDNA_pred > decision_rule}")
    return (ecDNA_pred > decision_rule).item()


'''
PARAMETERS
model : SCAMP model loaded from .pt file
input : copy number list
genes : gene name
k : number of neighbors to consider
ecDNA_percentage_thresh : threshold for ecDNA positive cells classifying ecDNA status
decision_rule : scAmp NN threshold for classifying ecDNA status

RETURNS
Boolean, the predicted ecDNA status
''' 
def KNN_ecDNA_predict(model, input, genes = np.array(['GENE']), k = 100, ecDNA_percentage_thresh = 0.001,decision_rule = 0.5) :
    arr = sorted(input)
    
    n = len(arr)
    out = []

    window_size = k + 1  # include self

    for i in range(n):
        target = arr[i]

        l, r = i - 1, i + 1
        count = 1

        while count < window_size:
            if l < 0:
                r += 1
            elif r >= n:
                l -= 1
            else:
                if abs(arr[l] - target) <= abs(arr[r] - target):
                    l -= 1
                else:
                    r += 1
            count += 1

        subset = np.array(arr[l+1:r])
      
        if np.mean(subset) > 2.5 and np.var(subset) > 10 :
            out.append(NN_ecDNA_predict(model, subset.tolist(), genes, decision_rule))

    print(f'Number of cells ecDNA positive: {sum(out)}')

    return sum(out) > (ecDNA_percentage_thresh * n)

'''
PARAMETERS
input : copy number list
k : number of neighbors to consider
mean_thresh : threshold needed to call a cell ecDNA positive
var_thresh : threshold needed to call a cell ecDNA positive
ecDNA_percentage_thresh : threshold for ecDNA positive cells classifying ecDNA status

RETURNS
Boolean, the predicted ecDNA status
''' 
# Just uses a cutoff, rather than running the scamp model
def _BASIC_KNN_ecDNA_predict(input, k = 100, mean_thresh = 5, var_thresh = 100, ecDNA_percentage_thresh = 0.01) :
    arr = sorted(input)
    
    n = len(arr)
    out = []

    window_size = k + 1  # include self

    for i in range(n):
        target = arr[i]

        l, r = i - 1, i + 1

        S = target
        Q = target * target
        count = 1

        while count < window_size:
            if l < 0:
                val = arr[r]
                r += 1
            elif r >= n:
                val = arr[l]
                l -= 1
            else:
                if abs(arr[l] - target) <= abs(arr[r] - target):
                    val = arr[l]
                    l -= 1
                else:
                    val = arr[r]
                    r += 1

            S += val
            Q += val * val
            count += 1

        mean = S / window_size
        var = (Q / window_size - mean * mean)**2

        out.append(int((mean > mean_thresh) and (var > var_thresh)))

    return sum(out) > (ecDNA_percentage_thresh * n)





'''
PARAMETERS
input : copy number list
n_components : if known, the number of components. Set to None to let kneed find
one_thresh : if n_components is None, uses this to determine when the knee is at 1 
max_components : if n_components is None, checks up to these components
var_thresh : threshold needed for variance to classify as ecDNA
mean_thresh : threshold needed for mean to classify as ecDNA

RETURNS
Boolean, the predicted ecDNA status
'''
def _BASIC_GMM_ecDNA_predict(input, 
                      n_components = None, 
                      one_thresh = 1.15, 
                      max_components = 15,
                      var_thresh = 100,
                      mean_thresh = 5
                      ) :
    
    input = np.array(input).reshape(-1, 1)
    if n_components is None :
        bic_scores = []
        aic_scores = []
        ks = []
        for k in range(1, max_components):
            gmm = GaussianMixture(n_components=k, random_state=0)
            gmm.fit(input)

            bic_scores.append(gmm.bic(input))
            aic_scores.append(gmm.aic(input))
            ks.append(k)

        # import matplotlib.pyplot as plt
        # fig, ax1 = plt.subplots(figsize=(7,5))

        # # BIC (left axis)
        # ax1.plot(ks, bic_scores, marker='o', color='tab:blue', label='BIC')
        # ax1.set_xlabel("Number of components (K)")
        # ax1.set_ylabel("BIC", color='tab:blue')
        # ax1.tick_params(axis='y', labelcolor='tab:blue')
        # plt.show()
        # plt.close()

        if bic_scores[0]/np.min(bic_scores) < one_thresh :
            n_components = 1
            print(f"Number of distributions (Kneed): {n_components}")

        else :

            bic_scaled = (bic_scores - np.min(bic_scores)) / np.ptp(bic_scores)
            # bic_scaled = np.log(np.array(bic_scores) - np.min(bic_scores) + 1e-8)
            
            kneedle = KneeLocator(
                ks,
                bic_scaled,
                curve="convex",      # or "concave" depending on your curve
                direction="decreasing"
            )

            n_components = kneedle.knee
            print(f"Number of distributions (Kneed): {n_components}")

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(input)


    for i in range(gmm.n_components):
        mean = gmm.means_[i, 0]
        var = gmm.covariances_[i, 0, 0]
        weight = gmm.weights_[i]

        # Super simple predictor
        if var >= var_thresh and mean >= mean_thresh :
            return True
    
    return False