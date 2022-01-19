from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import metrics
import config
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA

n_neighbors = 15


def kNN(X_test, y_test, X_val, y_val, X_train, y_train):

    """
    # Transform 3D data into 2D with help of PCA (Principal Component Analysis)
    clf = KNeighborsClassifier(n_neighbors, weights = 'uniform')
    pca = PCA(n_components = 2)
    X_train2 = pca.fit_transform(X_train)
    clf.fit(X_train2, y_train)
    plot_decision_regions(X_train2, y_train.ravel().astype(np.int_), clf=clf, legend=2)
    """

    # Create classifier
    clf = KNeighborsClassifier(n_neighbors, weights='uniform')
    clf.fit(X_train, y_train.ravel())

    # Plot results
    if not config.grad:
        # plot_with_plotly(data, X_train, y_train, X_test, y_test, clf)
        plot_with_matplotlib(clf, data, target)
    else:
        # data_3D(clf, data, target)
        print('Set config.grad = False, if you want to plot data (and not use the gradient)')

    # Predict results
    pred_results(clf, X_test, y_test)


def plot_with_plotly(data, X_train, y_train, X_test, y_test, clf):
    mesh_size = .02
    margin = 0.25

    # Create a mesh grid on which we will run our model
    x_min, x_max = data[:, 0].min() - margin, data[:, 0].max() + margin
    y_min, y_max = data[:, 1].min() - margin, data[:, 1].max() + margin
    xrange = np.arange(x_min, x_max, mesh_size)
    yrange = np.arange(y_min, y_max, mesh_size)
    xx, yy = np.meshgrid(xrange, yrange)

    # Create classifier, run predictions on grid
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    trace_specs = [
        [X_train, y_train, 0, 'Train', 'square'],
        [X_train, y_train, 1, 'Train', 'circle'],
        [X_test, y_test, 0, 'Test', 'square-dot'],
        [X_test, y_test, 1, 'Test', 'circle-dot']
    ]

    fig = go.Figure(data=[
        go.Scatter(
            x=X[y.ravel() == label, 0], y=X[y.ravel() == label, 1],
            name=f'{split} Split, Label {label}',
            mode='markers', marker_symbol=marker
        )
        for X, y, label, split, marker in trace_specs
    ])
    fig.update_traces(
        marker_size=12, marker_line_width=1.5,
        marker_color="lightyellow"
    )

    fig.add_trace(
        go.Contour(
            x=xrange,
            y=yrange,
            z=Z,
            showscale=False,
            colorscale='RdBu',
            opacity=0.4,
            name='Score',
            # hoverinfo='skip',
            hovertemplate=(
                'PEEP: %{x} <br>'
                'Compliance: %{y} <br>')
        )
    )
    fig.show()


def plot_with_matplotlib(clf, data, target):
    plt.xlabel("PEEP")
    plt.ylabel("Compliance")
    plt.title("k Nearest Neighbor (k = %i)" % (n_neighbors))
    plot_decision_regions(data, target.ravel().astype(np.int_), clf=clf, legend=2)


def data_3D(clf, data, target):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    # creating scatter plot
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=target, cmap=plt.cm.Paired)

    z = lambda x, y: (-clf.intercept_[0] - clf.coef_[0][0] * x - clf.coef_[0][1] * y) / clf.coef_[0][2]

    ax = plt.gca(projection='3d')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    xx = np.linspace(xlim[0], xlim[1], 50)
    yy = np.linspace(ylim[0], ylim[1], 50)
    zz = np.linspace(zlim[0], zlim[1], 50)
    XX, YY, ZZ = np.meshgrid(xx, yy, zz)
    xyz = np.vstack([XX.ravel(), YY.ravel(), ZZ.ravel()]).T
    Z = clf.decision_function(xyz).reshape(XX.shape)

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


def pred_results(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    print("Accuracy:", metrics.accuracy_score(y_test.ravel(), y_pred))

    # Confusion Matrix
    matrix = metrics.plot_confusion_matrix(clf, X_test, y_test,
                                           cmap=plt.cm.Blues,
                                           normalize='true')
    plt.title('Confusion matrix for kNN')
    plt.show()