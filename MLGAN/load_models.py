import os
import pickle
import time
import random
import torch.nn.functional as functional
from baselines.conan import *
import models.units as units
from models.transformer import Get_embd, Generator, Discriminator
# from models.Dipole import Get_embd, Generator, Discriminator
# from models.RNN import Get_embd, Generator, Discriminator
from baselines.meta_model import MLP, convert_meta_data, train_embd_dis_meta, upsample
import matplotlib.pyplot as plt
import numpy as np

def compute_weight(data, embd_model, generator_model, discriminator_model, meta_net, true_flag=True):
    weight = []
    batch_ori = int(np.ceil(float(len(data[0])) / float(batch_size_ori)))
    for idx in range(batch_ori):
        diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, \
        labels = get_batch_data_single_label(data, options, batch_size_ori, idx)
        h = embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths)
        if true_flag:
            temp_outputs = discriminator_model(h)
            loss_vector = functional.cross_entropy(temp_outputs, labels.long(), reduction='none')
            pseudo_loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))
            pseudo_weight = meta_net(pseudo_loss_vector_reshape.data)
            temp = pseudo_weight.squeeze(-1).data.tolist()
            weight.extend(temp)
        else:
            cur_batch_size = labels.size(0)
            h_new = generator_model(h)
            temp_outputs = discriminator_model(h_new)
            labels_new = torch.tensor([1]*cur_batch_size, dtype=torch.int64)
            loss_vector = functional.cross_entropy(temp_outputs, labels_new.long(), reduction='none')
            pseudo_loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))
            pseudo_weight = meta_net(pseudo_loss_vector_reshape.data)
            temp = pseudo_weight.squeeze(-1).data.tolist()
            weight.extend(temp)

    return weight

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
pretrain_epoch = 20
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
embd_model = Get_embd(options)
generator_model = Generator(options)
discriminator_model = Discriminator(options)
meta_net = MLP()

# load model
epoch = 936
temp_path = './saved_models/'+disease+'_saved_models/'
best_parameters_embd_file = temp_path + 'embd_model.' + str(epoch)
best_parameters_generator_file =temp_path + 'generator_model.' + str(epoch)
best_parameters_discriminator_file = temp_path + 'discriminator_model.' + str(epoch)
best_parameters_meta_net_file = temp_path + 'meta_net.' + str(epoch)

embd_model.load_state_dict(torch.load(best_parameters_embd_file, map_location=torch.device('cpu')))
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

# compute weight
train_pos = units.select_data(train, pos_flag=1)
train_neg = units.select_data(train, pos_flag=0)
train_pos_weight = compute_weight(train_pos, embd_model, generator_model, discriminator_model, meta_net)
train_neg_weight = compute_weight(train_neg, embd_model, generator_model, discriminator_model, meta_net)
sample_weight = compute_weight(sample, embd_model, generator_model, discriminator_model, meta_net, true_flag=False)


fig, ax = plt.subplots()
all_weight = []
all_weight.extend(train_neg_weight)
all_weight.extend(train_pos_weight)
all_weight.extend(sample_weight)
all_heights, all_bins = np.histogram(all_weight, bins=20)
a_heights, a_bins = np.histogram(train_pos_weight, bins=all_bins)
b_heights, b_bins = np.histogram(train_neg_weight, bins=all_bins)
c_heights, c_bins = np.histogram(sample_weight, bins=all_bins)
width = (all_bins[1] - all_bins[0])/4


ax.bar(a_bins[:-1], a_heights, width=width, log=True, facecolor='red', label='real positive sample')
ax.bar(b_bins[:-1]+width, b_heights, width=width, log=True, facecolor='blue', label='real negative sample')
ax.bar(c_bins[:-1]+width+width, c_heights, width=width, log=True, facecolor='gold', label='generated positive sample')
ax.legend(['Real positive sample', 'Unlabeled sample', 'Generated positive sample'], loc = 'upper right')
plt.ylabel('Numbers')
plt.xlabel('Weight')
plt.show()

print(b_bins[:-1])
print(b_heights)
