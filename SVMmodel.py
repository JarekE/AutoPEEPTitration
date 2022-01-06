import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import pyplot as plt
import config
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mlxtend.plotting import plot_decision_regions


def plot_different_SVCs(X_train, y_train):
    h = .02
    C = 1.0  # SVM regularization parameter
    svc = SVC(kernel='linear', C=C).fit(X_train, y_train)
    rbf_svc = SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
    poly_svc = SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
    lin_svc = LinearSVC(C=C).fit(X_train, y_train)

    # create a mesh to plot in
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # title for the plots
    titles = ['SVC with linear kernel',
              'LinearSVC (linear kernel)',
              'SVC with RBF kernel',
              'SVC with polynomial (degree 3) kernel']

    for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        plt.subplot(2, 2, i + 1)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

        # Plot also the training points
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)
        plt.xlabel('PEEP')
        plt.ylabel('Compliance')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.xticks(())
        plt.yticks(())
        plt.title(titles[i])

    plt.show()


def plot_linear(X_train, y_train, X_test, y_test, data, target):
    clf = SVC(kernel='linear', gamma=0.7)  # Linear Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train.ravel())

    # Plot data with classifier line
    w = clf.coef_[0]
    # print(w)
    a = -w[0] / w[1]

    xx = np.linspace(0, 25)  # draw classifier line from x=0 to x=25
    yy = a * xx - clf.intercept_[0] / w[1]

    h0 = plt.plot(xx, yy, 'k-', label="Decision boundary")  # 'k-' dashed line in black

    plt.scatter(data[:, 0], data[:, 1], c=target.ravel())
    plt.legend()
    plt.xlabel('PEEP [mbar]')
    plt.ylabel('Compliance [mL/mbar]')
    plt.title('Linear SVM')
    plt.show()

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test.ravel(), y_pred))

    # Confusion Matrix
    matrix = metrics.plot_confusion_matrix(clf, X_test, y_test,
                                           cmap=plt.cm.Blues,
                                           normalize='true')
    plt.title('Confusion matrix for RBF SVM')
    plt.show()


def plot_rbf(X_train, y_train, X_test, y_test, data):
    clf = SVC(kernel='rbf', gamma=0.7)  # Radial Basis Function Kernel

    # Train the model using the training sets
    clf.fit(X_train, y_train.ravel())

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)

    h = .02
    # create a mesh to plot in
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.coolwarm)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('PEEP')
    plt.ylabel('Compliance')
    plt.show()

    print("Accuracy:", metrics.accuracy_score(y_test.ravel(), y_pred))
    # print("Precision:",metrics.precision_score(y_test, y_pred))
    # print("Recall:",metrics.recall_score(y_test, y_pred))

    print(metrics.confusion_matrix(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred))


def plot_rbf2(X_train, y_train, X_test, y_test, data):
    # Create the SVM
    svm = SVC(random_state=42, kernel='rbf', gamma=1)

    # Fit the data to the SVM classifier
    svm = svm.fit(X_train, y_train.ravel())

    # Evaluate by means of a confusion matrix
    matrix = metrics.plot_confusion_matrix(svm, X_test, y_test,
                                           cmap=plt.cm.Blues,
                                           normalize='true')
    plt.title('Confusion matrix for RBF SVM')
    plt.show()

    # Generate predictions
    y_pred = svm.predict(X_test)

    # Evaluate by means of accuracy
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy}')

    # Plot decision boundary
    plot_decision_regions(X_test, y_test[:, 0].astype(np.int_), clf=svm, legend=2)
    plt.xlabel('PEEP [mbar]')
    plt.ylabel('Compliance [mL/mbar]')
    plt.show()


def plot_3D_data(X_train, y_train, X_test, y_test, data):
    # Fit the data with an svm
    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train.ravel())

    y_pred = svc.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test.ravel(), y_pred))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train.ravel())
    ax.set_xlabel('PEEP [mbar]')
    ax.set_ylabel('Compliance [mL/mbar]')
    ax.set_zlabel('Gradient')
    plt.show()


def rbf_3D(X_train, y_train, X_test, y_test, data):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # creating scatter plot
    ax.scatter3D(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap=plt.cm.Paired)

    clf2 = SVC(kernel='rbf', C=100)
    clf2.fit(X_train, y_train)

    z = lambda x, y: (-clf2.intercept_[0] - clf2.coef_[0][0] * x - clf2.coef_[0][1] * y) / clf2.coef_[0][2]

    ax = plt.gca(projection='3d')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    ### from here i don't know what to do ###
    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    zz = np.linspace(zlim[0], zlim[1], 50)
    XX, YY, ZZ = np.meshgrid(xx, yy, zz)
    xyz = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
    Z = clf2.decision_function(xyz).reshape(XX.shape)

    # find isosurface with marching cubes
    dx = xx[1] - xx[0]
    dy = yy[1] - yy[0]
    dz = zz[1] - zz[0]
    verts, faces, _, _ = measure.marching_cubes(Z, 0, spacing=(1, 1, 1), step_size=2)
    verts *= np.array([dx, dy, dz])
    verts -= np.array([xlim[0], ylim[0], zlim[0]])

    # add as Poly3DCollection
    mesh = Poly3DCollection(verts[faces])
    mesh.set_facecolor('g')
    mesh.set_edgecolor('none')
    mesh.set_alpha(0.3)
    ax.add_collection3d(mesh)
    ax.view_init(20, -45)
    ax.set_xlabel('PEEP')
    ax.set_ylabel('Compliance')
    ax.set_zlabel('Gradient')
    plt.show()

    # Accuracy
    y_pred = clf2.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test.ravel(), y_pred))

    # Confusion Matrix
    matrix = metrics.plot_confusion_matrix(clf2, X_test, y_test,
                                           cmap=plt.cm.Blues,
                                           normalize='true')
    plt.title('Confusion matrix for RBF SVM')
    plt.show()


def svm(data, target):
    data = np.concatenate(data, axis=0)
    target = np.concatenate(target, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

    if config.grad:
        # plot_3D_data(X_train, y_train, X_test, y_test, data)
        rbf_3D(X_train, y_train, X_test, y_test, data)
    else:
        plot_rbf(X_train, y_train, X_test, y_test, data)
        # plot_different_SVCs(X_train, y_train)
        # plot_rbf2(X_train, y_train, X_test, y_test, data)
        # plot_linear(X_train, y_train, X_test, y_test, data, target)

    return