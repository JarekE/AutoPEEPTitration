import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split

def comp_clf(data, target):
    data = np.concatenate(data, axis=0 )
    target = np.concatenate(target, axis=0 )
    
    # Load and split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.1, random_state=42, shuffle = False)

    # Initializing Classifiers
    clf1 = KNeighborsClassifier(n_neighbors = 15, weights = 'uniform')
    clf2 = RandomForestClassifier(random_state=1, n_estimators=100)
    clf3 = GaussianNB()
    clf4 = SVC(kernel='rbf', gamma = 0.7)

    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(10,8))

    labels = ['k-Nearest-Neighbor', 'Random Forest', 'Naive Bayes', 'SVM']

    for clf, lab, grd in zip([clf1, clf2, clf3, clf4],
                            labels,
                            itertools.product([0, 1], repeat=2)):

        clf.fit(X_train, y_train.ravel())
        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=data, y=target.ravel().astype(np.int_), clf=clf, legend=2)
        #plt.subplots_adjust(bottom=0.8, right=0.8, top=0.9)
        plt.title(lab)
        if lab == 'Naive Bayes' or lab == 'SVM':
            plt.xlabel('PEEP [mbar]')
        plt.ylabel('Compliance [mL/mbar]')

    plt.show()