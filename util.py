from sklearn import preprocessing
import numpy as np
import numba as nb


@nb.jit(nopython=True)
def fetch_data(index, num, img, new_img):
    for i in range(num):
        new_img[i] = img[index[i], :]
    return new_img


def fetch_index_label(class_num, labels, col):
    class_length = []
    sample_label = []
    sample_index = []
    for i in range(class_num):
        index_x, index_y = np.where(labels == i + 1)
        index_ii = index_x * col + index_y
        class_length.append(len(index_ii))
        class_ii = np.ones((len(index_ii))) * (i + 1)
        class_ii = class_ii.astype(int)
        sample_label.extend(class_ii)
        sample_index.extend(index_ii)
    class_length = np.array(class_length)
    sample_label = np.array(sample_label)
    sample_index = np.array(sample_index)
    return class_length, sample_label, sample_index


def extract_superpixel_feature(superpixel_index, fea, seg_map, width, img):
    cur_x, cur_y = np.where(seg_map == superpixel_index)
    cur_index = cur_x * width + cur_y
    superpixel_fea = img[cur_index, :]
    fea[superpixel_index] = np.mean(superpixel_fea, axis=0)


def fetch_superpixel_feature(index, data, all_index, length, col, seg_map, fea):
    for i in range(length):
        cur_index = all_index[i]
        cur_y = np.mod(cur_index, col)
        cur_x = int((cur_index-cur_y)/col)
        cur_superpixel = seg_map[cur_x, cur_y]
        data[i, index, :] = fea[cur_superpixel]


def divide_train_test(class_length, class_num, sample_label, sample_index, training_per_class):
    tr_lab = []
    tt_lab = []
    index_train = []
    index_test = []
    for i in range(class_num):
        cur_sample_num = class_length[i]
        cur_sequence = np.random.permutation(cur_sample_num)   # get random arrangement for each class
        cur_class_index = np.where(sample_label == i + 1)
        cur_class_index = cur_class_index[0]
        shuffled_index = cur_class_index[cur_sequence]  # shuffle the indexes for each class
        cur_tr_index = shuffled_index[0:training_per_class[i]]
        cur_sample_num = len(shuffled_index)
        index_train.extend(sample_index[cur_tr_index])
        cur_tt_index = shuffled_index[training_per_class[i]:cur_sample_num]
        index_test.extend(sample_index[cur_tt_index])

        tt_temp = np.ones((cur_sample_num - training_per_class[i])) * (i + 1)
        tt_temp = tt_temp.astype(int)
        tt_lab.extend(tt_temp)

        tr_temp = np.ones((training_per_class[i])) * (i + 1)
        tr_temp = tr_temp.astype(int)
        tr_lab.extend(tr_temp)

    tr_lab = np.array(tr_lab) - 1
    tt_lab = np.array(tt_lab) - 1
    index_train = np.array(index_train)
    index_test = np.array(index_test)
    return index_train, tr_lab, index_test, tt_lab


def feature_concatenate(scale_num, multiscale_feat, prob):
    no_class = np.shape(prob)[1]
    item_num = np.shape(prob)[0]
    add_feat = np.repeat(prob.reshape(item_num, 1, no_class), scale_num, axis=1)
    add_feat = add_feat.astype(np.float32)
    feat_arr = np.concatenate([add_feat, multiscale_feat], axis=2)
    return feat_arr
