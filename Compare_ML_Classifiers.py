import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import matplotlib.gridspec as gridspec
import itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
from sklearn import metrics
import config


def comp_clf(X_test, y_test, X_val, y_val, X_train, y_train):

    # Initializing Classifiers
    clf1 = KNeighborsClassifier(n_neighbors=15, weights='uniform')
    clf2 = RandomForestClassifier(random_state=1, n_estimators=100)
    clf3 = GaussianNB()
    clf4 = SVC(kernel='rbf', gamma=0.7)

    labels = ['k-Nearest-Neighbor', 'Random Forest', 'Naive Bayes', 'SVM']
    clfs = [clf1, clf2, clf3, clf4]

    # Fitting Classifiers
    for clf, lab in zip(clfs, labels):
        clf.fit(X_train, y_train.ravel())
        # Evaluating the performance
        pred_results(clf, X_test, y_test, lab)

    # Plotting decision boundaries in 2D
    if not config.grad:
        plot_clfs(clfs, X_test, y_test, X_train, y_train, labels)
    else:
        raise Exception("Check your gradient settings")


def pred_results(clf, X_test, y_test, label):
    # Accuracy
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test.ravel(), y_pred))

    # Confusion Matrix
    metrics.plot_confusion_matrix(clf, X_test, y_test,
                                           cmap=plt.cm.Blues,
                                           normalize='true')
    plt.title('Confusion matrix for ' + str(label))
    plt.show()


def plot_clfs(clfs, X_test, y_test, X_train, y_train, labels):
    data = np.concatenate((X_train, X_test), axis=0)
    target = np.concatenate((y_train, y_test), axis=0)

    gs = gridspec.GridSpec(2, 2)

    plt.figure(figsize=(10, 8))

    for clf, lab, grd in zip(clfs,
                             labels,
                             itertools.product([0, 1], repeat=2)):

        plt.subplot(gs[grd[0], grd[1]])
        plot_decision_regions(X=data, y=target.ravel().astype(np.int_), clf=clf, legend=2)

        plt.title(lab)
        if lab == 'Naive Bayes' or lab == 'SVM':
            plt.xlabel('PEEP [mbar]')
        plt.ylabel('Compliance [mL/mbar]')

    plt.show()