import sklearn.svm as svm
from seedtrick.kernels import SVMKernel

class CustomKernelSVC(svm.SVC):
    def __init__(self, kernel, embed=True):
        self.svm_kernel = SVMKernel(kernel) if embed else kernel
        svm.SVC.__init__(self, kernel='precomputed')
        self.X_fit = None

    def fit(self, X, y):
        self.X_fit = X
        tmp = self.svm_kernel(X, X)
        svm.SVC.fit(self, tmp, y)

    def predict(self, X):
        tmp = self.svm_kernel(X, self.X_fit)
        return svm.SVC.predict(self, tmp)

    def score(self, X, y):
        return (self.predict(X) == y).mean()
