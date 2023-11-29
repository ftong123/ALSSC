import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from joblib import Parallel, delayed
from sklearn.utils.fixes import _joblib_parallel_args


def _train(x, y, index, model_pool):
    cur_x = x[:, index, :]
    model_pool[index].fit(cur_x, y)


def _predict(x, index, result, model_pool):
    cur_x = x[:, index, :]
    prf = model_pool[index]
    result[index, :, :] = prf.predict_proba(cur_x)


class Layer(object):

    def __init__(self, n_tree=5, n_class=10):
        setattr(self, 'n_tree', int(n_tree))
        setattr(self, 'n_class', int(n_class))
        self.Classifiers = []

    def train(self, tr_x, tr_y, layer):
        tr_prob = self.cascade_layer(layer, tr_x, tr_y)

    def predict(self, tt_x, layer):
        tt_prob = self.cascade_layer(layer, tt_x)
        tt_prob = np.mean(tt_prob, axis=0)  # average class probability vectors generated from multiscale features
        return tt_prob

    def cascade_layer(self, layer, x, y=None):
        scale_num = np.shape(x)[1]
        sample_num = np.shape(x)[0]
        pre_prob = []
        if y is not None:
            print('Adding/Training Layer, n_layer={}'.format(layer+1))
            model_pool = []
            for irf in range(scale_num):
                prf = ExtraTreesClassifier(n_estimators=self.n_tree)
                model_pool.append(prf)
            Parallel(n_jobs=-1, **_joblib_parallel_args(prefer="threads", require="sharedmem"))(
                delayed(_train)(x, y, index, model_pool) for index in range(scale_num))
            self.Classifiers.append(model_pool)
        elif y is None:
            pre_prob = np.zeros((scale_num, sample_num, self.n_class), dtype=np.float32)
            model_pool = self.Classifiers[layer]
            Parallel(n_jobs=-1, **_joblib_parallel_args(prefer="threads", require="sharedmem"))(
                delayed(_predict)(x, index, pre_prob, model_pool) for index in range(scale_num))

        return pre_prob
