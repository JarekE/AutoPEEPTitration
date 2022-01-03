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
from sklearn import metrics
import config

def comp_clf(data, target):
    data = np.concatenate(data, axis=0 )
    target = np.concatenate(target, axis=0 )
    
    # Load and split data
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42, shuffle = True)

    # Initializing Classifiers
    clf1 = KNeighborsClassifier(n_neighbors = 15, weights = 'uniform')
    clf2 = RandomForestClassifier(random_state=1, n_estimators=100)
    clf3 = GaussianNB()
    clf4 = SVC(kernel='rbf', gamma = 0.7)

    labels = ['k-Nearest-Neighbor', 'Random Forest', 'Naive Bayes', 'SVM']
    clfs = [clf1, clf2, clf3, clf4]

    # Fitting Classifiers
    for clf, lab in zip(clfs, labels):
        clf.fit(X_train, y_train.ravel())
        # Evaluating the performance
        pred_results(clf, X_test, y_test, lab)

    # Plotting decision boundaries in 2D
    if not config.grad:
        plot_clfs(clfs, data, target, labels)
    
  
def pred_results(clf, X_test, y_test, label):
    # Accuracy
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test.ravel(), y_pred))

    # Confusion Matrix
    matrix = metrics.plot_confusion_matrix(clf, X_test, y_test,
                                    cmap=plt.cm.Blues,
                                    normalize='true')
    plt.title('Confusion matrix for ' + str(label))
    plt.show()


def plot_clfs(clfs, data, target, labels):
    gs = gridspec.GridSpec(2, 2)

    fig = plt.figure(figsize=(10,8))

    for clf, lab, grd in zip(clfs,
                            labels,
                            itertools.product([0, 1], repeat=2)):

        ax = plt.subplot(gs[grd[0], grd[1]])
        fig = plot_decision_regions(X=data, y=target.ravel().astype(np.int_), clf=clf, legend=2)

        plt.title(lab)
        if lab == 'Naive Bayes' or lab == 'SVM':
            plt.xlabel('PEEP [mbar]')
        plt.ylabel('Compliance [mL/mbar]')

    plt.show()