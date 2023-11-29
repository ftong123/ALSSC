import numpy as np
import time
from scipy.io import loadmat
import imageio
from joblib import Parallel, delayed
import cv2.ximgproc
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.utils.fixes import _joblib_parallel_args
from sklearn.decomposition import PCA
from forest_classification import Layer
import util

""" assign hyperparameters """
max_superpixel_size = 19   # the size of the patch
initial_training_num = 5   # initial number of training samples for every class
rounds_AL = 10             # the active learning rounds
layer_num = rounds_AL + 1  # number of layers in ALSSC
learn_num = 10             # number of samples selected by active learning in each layer
ntrees_in_EFs = 20         # number of trees used in each EF


time_start = time.time()  # get the start time

""" Load data and ground truth """
image = loadmat(r'data\images\Salinas\Salinas_corrected.mat')
imageGT = loadmat(r'data\images\Salinas\Salinas_gt.mat')
image_input = image["salinas_corrected"]
labels = imageGT["salinas_gt"]
[row, col, dim] = image_input.shape
class_num = np.max(labels)

""" Load multiscale segmentation maps """
scale_num = int((max_superpixel_size-1)/2)  # number of segmentation maps
segmentations = []
for i in range(scale_num):
    seg_name = r'data\segmentation_maps\Salinas\Salinas_seg_' + str(2 * i + 3) + 'X' + str(2 * i + 3) + '.mat'
    cur_seg = loadmat(seg_name)
    segmentations.append(cur_seg["labels"])


""" Conduct PCA to get the first PCA component (used in guided filtering) """
image_input = image_input.reshape((row * col, dim))
pca_breast = PCA(n_components=1)
first_pca = pca_breast.fit_transform(image_input)
first_pca = first_pca.reshape((row, col))
first_pca = first_pca.astype(np.float32)

""" image standardization """
scaler = preprocessing.StandardScaler().fit(image_input)
Image_input = scaler.transform(image_input)
Image_input = Image_input.astype(np.float32)


""" extract multiscale features from multiscale superpixels """
multiscale_feat = []
for i in range(scale_num):
    cur_seg = segmentations[i]
    superpixel_num = np.max(cur_seg) + 1
    seg_fea = np.zeros((superpixel_num, dim))
    Parallel(n_jobs=-1, **_joblib_parallel_args(prefer="threads", require="sharedmem"))\
            (delayed(util.extract_superpixel_feature)(index, seg_fea, cur_seg, col, image_input)
             for index in range(superpixel_num))
    multiscale_feat.append(seg_fea)


""" get 1_D indexes and labels for all samples in the dataset """
class_length = []  # number of samples for each class
sample_label = []  # label for all samples
sample_index = []  # 1-dimensional indexes for all samples
class_length, sample_label, sample_index = util.fetch_index_label(class_num, labels, col)
index_image = np.array(range(0, row * col))  # 1_D indexes for all pixels in the image


""" set numbers of initial training samples for each class """
training_per_class = np.zeros(class_num)
training_per_class = training_per_class.astype(int)
for i in range(class_num):
    training_per_class[i] = 5


""" randomly divide all sample into training and testing sets """
index_train, tr_lab, index_test, tt_lab = util.divide_train_test(class_length, class_num, sample_label,
                                                                 sample_index, training_per_class)

""" Multiple layers training with active learning """
train_prob = []  # class probability vectors for training samples
test_prob = []   # class probability vectors for testing samples
all_prob = []    # class probability vectors for all pixels
ALSSC_layers = Layer(n_tree=ntrees_in_EFs, n_class=class_num)

for layer in range(layer_num):
    # training stage #
    cur_tr_num = len(tr_lab)
    tr_feat = np.zeros((cur_tr_num, scale_num, dim), dtype=np.float32)
    Parallel(n_jobs=-1, **_joblib_parallel_args(prefer="threads", require="sharedmem"))(
        delayed(util.fetch_superpixel_feature)(index, tr_feat, index_train, cur_tr_num, col,
                                               segmentations[index],
                                               multiscale_feat[index]) for index in range(scale_num))

    # concatenate multiscale features with class probability vectors inherited from the preceding layer
    if layer > 0:
        tr_feat = util.feature_concatenate(scale_num, tr_feat, train_prob)
    ALSSC_layers.train(tr_feat, tr_lab, layer)  # conduct training in current layer

    # testing stage #
    img_feat = np.zeros((row*col, scale_num, dim), dtype=np.float32)
    Parallel(n_jobs=-1, **_joblib_parallel_args(prefer="threads", require="sharedmem"))(
        delayed(util.fetch_superpixel_feature)(index, img_feat, index_image, row*col, col,
                                               segmentations[index],
                                               multiscale_feat[index]) for index in range(scale_num))

    # concatenate multiscale features with class probability vectors inherited from the preceding layer
    if layer > 0:
        img_feat = util.feature_concatenate(scale_num, img_feat, all_prob)
    all_prob = ALSSC_layers.predict(img_feat, layer)

    # guided filtering for optimizing classification performance #
    for k in range(class_num):
        cur_prob_map = all_prob[:, k]
        cur_prob_map = cur_prob_map.reshape((row, col))
        cur_prob_map = cv2.ximgproc.guidedFilter(first_pca, cur_prob_map, int(max_superpixel_size / 4), 0.1)
        all_prob[:, k] = cur_prob_map.reshape((row * col))

    # active learning to enrich training set with most informative samples in testing set#
    train_prob = all_prob[index_train]
    test_prob = all_prob[index_test]
    if layer < layer_num - 1:
        # select most informative samples from testing set based in Breaking Ties (BT)
        sort_prob = np.sort(test_prob, axis=1)
        bt_value = sort_prob[:, -1] - sort_prob[:, -2]
        sort_index = np.argsort(bt_value)
        select_index = sort_index[0:learn_num]

        # add selected samples to training set
        tr_lab = np.append(tr_lab, tt_lab[select_index], axis=0)
        train_prob = np.append(train_prob, test_prob[select_index, :], axis=0)
        index_train = np.append(index_train, index_test[select_index])

        # remove selected samples from testing set
        tt_lab = np.delete(tt_lab, select_index, axis=0)
        index_test = np.delete(index_test, select_index)

predict_lab = np.argmax(test_prob, axis=1)  # get predicted label for all rest testing samples

"""print classification time cost """
time_end = time.time()
timeConsumed = time_end - time_start
print("time cost is: {}s".format(timeConsumed))

"""classification metrics evaluation """
accuracy = accuracy_score(y_true=tt_lab, y_pred=predict_lab)
kappa = cohen_kappa_score(tt_lab, predict_lab)
acc_for_each_class = metrics.recall_score(tt_lab, predict_lab, average=None)
average_accuracy = np.mean(acc_for_each_class)
print("Overall Accuracy is: {}".format(accuracy))
print("Average Accuracy is: {}".format(average_accuracy))
print("Kappa is: {}".format(kappa))
print("CA is: ")
for i in range(len(acc_for_each_class)):
    print(acc_for_each_class[i])


""" save the classification map """
color_table = np.zeros((16, 3), dtype=np.uint8)
color_table[0] = [140, 67, 46]
color_table[1] = [0, 0, 255]
color_table[2] = [255, 100, 0]
color_table[3] = [0, 255, 123]
color_table[4] = [164, 75, 155]
color_table[5] = [101, 174, 255]
color_table[6] = [118, 254, 172]
color_table[7] = [60, 91, 112]
color_table[8] = [255, 255, 0]
color_table[9] = [255, 255, 125]
color_table[10] = [255, 0, 255]
color_table[11] = [100, 0, 255]
color_table[12] = [0, 172, 254]
color_table[13] = [0, 255, 0]
color_table[14] = [175, 175, 80]
color_table[15] = [101, 193, 60]

classification_map = np.zeros((row, col, 3), dtype=np.uint8)  # RGB classification map

# fill training labels
for i in range(len(index_train)):
    cur_Index = index_train[i]
    cur_label = tr_lab[i]
    cur_y = np.mod(cur_Index, col)
    cur_x = int((cur_Index - cur_y) / col)
    cur_color = color_table[cur_label]
    classification_map[cur_x, cur_y] = cur_color

# fill predicted labels
for i in range(len(index_test)):
    cur_Index = index_test[i]
    cur_label = predict_lab[i]
    cur_y = np.mod(cur_Index, col)
    cur_x = int((cur_Index - cur_y) / col)
    cur_color = color_table[cur_label]
    classification_map[cur_x, cur_y] = cur_color

save_path = r'classification_map\Salinas\ALSSC_Salinas_Classification_Map.png'
imageio.imwrite(save_path, classification_map)








