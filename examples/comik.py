import numpy as np
from seedtrick.svm import CoMIK

from sklearn.model_selection import train_test_split

def _fasta_reader(path):
    ret = list()
    with open(path) as f:
        s = ''
        for l in f:
            if l.startswith('>'):
                if s == '':
                    continue
                ret.append(s)
                s = ''
            else:
                s += l.strip().upper()
        ret.append(s)
    return ret

if __name__ == '__main__':
    DIR = '/media/robin/DATA/comik/sample_data/simulated_dataset1'
    pos_examples = _fasta_reader(DIR + '/pos.fasta')
    neg_examples = _fasta_reader(DIR + '/neg.fasta')
    X = np.array(pos_examples + neg_examples)
    y = np.ones(len(X))
    y[len(pos_examples):] = -1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    print(y_train)
    print(y_test)
    comik = CoMIK(3, nb_exp_points=5)
    comik.fit(X_train, y_train)
    print('Accuracy: {}%'.format(100*comik.score(X_test, y_test)))
