import kernels
from dataset import load_musk_dataset
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def test_svm():
    clf1 = SVC(kernel='rbf', gamma=1)
    clf2 = SVC(kernel=kernels.SVMKernel(kernel=kernels.RBFKernel(1)))
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    for x in (X_train, X_test, y_train, y_test):
        print(x.shape)
    for i, clf in enumerate((clf1, clf2)):
        print(['Original RBF kernel', 'Custom RBF kernel'][i])
        clf.fit(X_train, y_train)
        print(clf.score(X_test, y_test))

def test_datasets():
    musk = load_musk_dataset()
    print(len(musk[0].X))
    print(len(musk[1].X))
