import os
import pickle
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import torch.nn.functional as functional
from baselines.conan import *
import models.units as units
from models.transformer import Get_embd, Generator, Discriminator
# from models.Dipole import Get_embd, Generator, Discriminator
# from models.RNN import Get_embd, Generator, Discriminator
from baselines.meta_model import MLP, convert_meta_data, train_embd_dis_meta, upsample
import matplotlib.pyplot as plt
import numpy as np

import pickle
print(pickle.format_version)
print('0')

def compute_embd(data, embd_model, generator_model_original=None):
    embd = []
    batch_ori = int(np.ceil(float(len(data[0])) / float(batch_size_ori)))
    for idx in range(batch_ori):
        diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, \
        labels = get_batch_data_single_label(data, options, batch_size_ori, idx)
        h = embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths)
        if generator_model_original==None:
            h_new = h.detach().numpy()
            embd.append(h_new)
        else:
            h_synthetic = generator_model_original(h).detach().numpy()
            embd.append(h_synthetic)

    embd_list = np.concatenate(embd, axis=0)

    return embd_list

device = torch.device("cpu")
batch_size = 128
batch_size_ori = 512
batch_size_meta = 100
batch_size_gan = 64
maxcode = 50
maxlen = 100
dropout_rate = 0.5
L2_reg = 1e-3
log_eps = 1e-8
n_labels = 2  # binary classification
visit_size = 64  # size of input embedding
hidden_size = 128  # size of hidden layer
gamma = 0.0  # setting for Focal Loss, when it's zero, it's equal to standard cross loss
layer = 1  # layer of Transformer
meta_lr = 1e-5
normal_lr = 1e-4
lr = 1e-2
gan_meta_epoch = 400
meta_interval = 1

model_file = eval('Get_embd')
disease = 'hes'  # name of the sample data set, you can place you own data set by following the same setting

path = '../../dataset/' + disease + '_dataset/'
training_file = path + disease + '.train'
validation_file = path + disease + '.valid'
testing_file = path + disease + '.test'


dict_file = path + disease + '.record_code_dict'
code2id = pickle.load(open(dict_file, 'rb'))
n_diagnosis_codes = len(pickle.load(open(dict_file, 'rb'))) + 1

options = locals().copy()

# build
embd_model_original = Get_embd(options)
generator_model_original = Generator(options)
embd_model = Get_embd(options)
generator_model = Generator(options)
discriminator_model = Discriminator(options)
meta_net = MLP()

# load model
epoch = 219
temp_path = './saved_models/'+disease+'_saved_models/'
original_parameters_embd_file = temp_path + 'embd_model.temp'
original_parameters_generator_file =temp_path + 'generator_model.temp'
best_parameters_embd_file = temp_path + 'embd_model.' + str(epoch)
best_parameters_generator_file =temp_path + 'generator_model.' + str(epoch)
best_parameters_discriminator_file = temp_path + 'discriminator_model.' + str(epoch)
best_parameters_meta_net_file = temp_path + 'meta_net.' + str(epoch)

embd_model_original.load_state_dict(torch.load(original_parameters_embd_file, map_location=torch.device('cpu')))
embd_model.load_state_dict(torch.load(best_parameters_embd_file, map_location=torch.device('cpu')))
generator_model_original.load_state_dict(torch.load(original_parameters_generator_file, map_location=torch.device('cpu')))
generator_model.load_state_dict(torch.load(best_parameters_generator_file, map_location=torch.device('cpu')))
discriminator_model.load_state_dict(torch.load(best_parameters_discriminator_file, map_location=torch.device('cpu')))
meta_net.load_state_dict(torch.load(best_parameters_meta_net_file, map_location=torch.device('cpu')))
embd_model.eval()
generator_model.eval()
discriminator_model.eval()
meta_net.eval()

# load data
train, validate, test = units.load_data(training_file, validation_file, testing_file)
# load generated data
sample = np.array(pickle.load(open(temp_path + 'sample_train_neg.pickle', 'rb')))
print('0')

# compute embedding
train_pos = units.select_data(train, pos_flag=1)
train_neg = units.select_data(train, pos_flag=0)
pos_num = train_pos[0].shape[-1]
print('pos_num: {}'.format(pos_num))
neg_num = train_neg[0].shape[-1]
print('neg_num: {}'.format(neg_num))
train_pos_embd = compute_embd(train_pos, embd_model)
train_neg_embd = compute_embd(train_neg, embd_model)
sample_embd = compute_embd(sample, embd_model_original, generator_model_original=generator_model_original)

X = []
X.append(train_pos_embd)
X.append(train_neg_embd)
X.append(sample_embd)
X = np.concatenate(X, axis=0)
with open('X.pickle', 'wb') as f:
    pickle.dump(X, f)

# X_embedded = TSNE(n_components=2, perplexity=50).fit_transform(X)
# with open('X_embedded.pickle', 'wb') as f:
#     pickle.dump(X_embedded, f)
#
# pos_num = train_pos[0].shape[-1]
# neg_num = train_neg[0].shape[-1]
# pos_embd = X_embedded[0:pos_num, :]
# neg_embd = X_embedded[pos_num:pos_num+neg_num, :]
# synthetic_embd = X_embedded[pos_num+neg_num:, :]
# embd_list = [neg_embd, pos_embd, synthetic_embd]
# colors = ['blue', 'red', 'gold']
# labels = ['Unlabeled sample', 'Real positive sample', 'Generated positive sample']
#
# fig, ax = plt.subplots()
# for temp, c, label in zip(embd_list, colors, labels):
#     ax.scatter(temp[:, 0], temp[:, 1], color=c, label=label)
#
# ax.legend(loc='upper right')
# plt.show()

# X = []
# X.append(train_pos_embd)
# selected_idx = random.sample(range(0, len(train_neg_embd)), len(train_pos_embd))
# train_neg_embd_selected = train_neg_embd[selected_idx]
# X.append(train_neg_embd_selected)
# X.append(sample_embd)
# X = np.concatenate(X, axis=0)
#
# X_embedded = TSNE(n_components=2, perplexity=30).fit_transform(X)
# with open('X_embedded.pickle', 'wb') as f:
#     pickle.dump(X_embedded, f)
#
# pos_num = train_pos[0].shape[-1]
# neg_num_selected = len(selected_idx)
# pos_embd = X_embedded[0:pos_num, :]
# neg_embd = X_embedded[pos_num:pos_num+neg_num_selected, :]
# synthetic_embd = X_embedded[pos_num+neg_num_selected:, :]
# embd_list = [neg_embd, pos_embd, synthetic_embd]
# colors = ['blue', 'red', 'gold']
# labels = ['Unlabeled sample', 'Real positive sample', 'Generated positive sample']
#
# fig, ax = plt.subplots()
# for temp, c, label in zip(embd_list, colors, labels):
#     ax.scatter(temp[:, 0], temp[:, 1], color=c, label=label)
#
# ax.legend(loc='upper right')
# plt.show()
