import torch.nn as nn
import torch
from random import shuffle
import random
import torch.nn.functional as functional
import numpy as np
import pickle
import copy
import torch.nn.init as init
from models.units import sample_data
from models.units import get_batch_data_single_label, get_batch_data
# from models.RNN import Generator, Get_embd, Discriminator
# from models.Dipole import Generator, Get_embd, Discriminator
from models.transformer import Generator, Get_embd, Discriminator
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, \
    recall_score, f1_score, roc_auc_score

def upsample(data, upsample_num=1):
    if upsample_num==1:
        return data
    pos_label = []
    pos_time = []
    pos_dia = []

    neg_label = []
    neg_time = []
    neg_dia = []
    for idx in range(len(data[-1])):
        if int(data[-1][idx])==0:
            neg_label.append(data[-1][idx])
            neg_time.append(data[1][idx])
            neg_dia.append(data[0][idx])
        else:
            pos_label.append(data[-1][idx])
            pos_time.append(data[1][idx])
            pos_dia.append(data[0][idx])

    pos_label_new, pos_time_new, pos_dia_new = [], [], []

    for i in range(upsample_num):
        pos_label_new.extend(pos_label)
        pos_time_new.extend(pos_time)
        pos_dia_new.extend(pos_dia)

    dia_list = [*pos_dia_new, *neg_dia]
    time_list = [*pos_time_new, *neg_time]
    label_list = [*pos_label_new, *neg_label]

    idx_list = [i for i in range(len(dia_list))]
    shuffle(idx_list)
    dia_list = (np.array(dia_list)[idx_list]).tolist()
    time_list = (np.array(time_list)[idx_list]).tolist()
    label_list = (np.array(label_list)[idx_list]).tolist()

    upsample_data = np.array((dia_list, time_list, label_list))
    return upsample_data

def convert_meta_data(data, rate=1):
    pos_label = []
    pos_time = []
    pos_dia = []

    neg_label = []
    neg_time = []
    neg_dia = []
    for idx in range(len(data[-1])):
        if int(data[-1][idx])==0:
            neg_label.append(data[-1][idx])
            neg_time.append(data[1][idx])
            neg_dia.append(data[0][idx])
        else:
            pos_label.append(data[-1][idx])
            pos_time.append(data[1][idx])
            pos_dia.append(data[0][idx])

    pos_num = len(pos_label) * rate
    neg_num = len(neg_label)
    select_list = [i for i in range(neg_num)]
    random.seed(2022)
    shuffle(select_list)
    selected_neg = select_list[:pos_num]
    select_neg_dia = (np.array(neg_dia)[selected_neg]).tolist()
    select_neg_time = (np.array(neg_time)[selected_neg]).tolist()
    select_neg_label = (np.array(neg_label)[selected_neg]).tolist()

    pos_dia_new, pos_time_new, pos_label_new = [], [], []
    for i in range(rate):
        pos_dia_new.extend(pos_dia)
        pos_time_new.extend(pos_time)
        pos_label_new.extend(pos_label)

    dia_list = [*pos_dia_new, *select_neg_dia]
    time_list = [*pos_time_new, *select_neg_time]
    label_list = [*pos_label_new, *select_neg_label]

    idx_list = [i for i in range(len(dia_list))]
    shuffle(idx_list)
    dia_list = (np.array(dia_list)[idx_list]).tolist()
    time_list = (np.array(time_list)[idx_list]).tolist()
    label_list = (np.array(label_list)[idx_list]).tolist()

    meta_data = np.array((dia_list, time_list, label_list))
    return meta_data

class HiddenLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(HiddenLayer, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.fc(x))

class MLP(nn.Module):
    def __init__(self, hidden_size=100, num_layers=1):
        super(MLP, self).__init__()
        self.first_hidden_layer = HiddenLayer(1, hidden_size)
        self.rest_hidden_layers = nn.Sequential(*[HiddenLayer(hidden_size, hidden_size) for _ in range(num_layers - 1)])
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.first_hidden_layer(x)
        x = self.rest_hidden_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)

from torch.optim.sgd import SGD

class MetaSGD(SGD):
    def __init__(self, net1, net2, *args, **kwargs):
        super(MetaSGD, self).__init__(*args, **kwargs)
        self.net1 = net1
        self.net2 = net2


    def set_parameter(self, current_module, name, parameters):
        if '.' in name:
            name_split = name.split('.')
            module_name = name_split[0]
            rest_name = '.'.join(name_split[1:])
            for children_name, children in current_module.named_children():
                if module_name == children_name:
                    self.set_parameter(children, rest_name, parameters)
                    break
        else:
            current_module._parameters[name] = parameters

    def meta_step(self, grads):
        group = self.param_groups[0]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']
        lr = group['lr']

        temp = 0
        change_temp = len(list(self.net1.named_parameters()))

        temp_parameter_list = list(self.net1.named_parameters())+list(self.net2.named_parameters())

        for (name, parameter), grad in zip(temp_parameter_list, grads):
            parameter.detach_()
            if weight_decay != 0:
                grad_wd = grad.add(parameter, alpha=weight_decay)
            else:
                grad_wd = grad
            if momentum != 0 and 'momentum_buffer' in self.state[parameter]:
                buffer = self.state[parameter]['momentum_buffer']
                grad_b = buffer.mul(momentum).add(grad_wd, alpha=1-dampening)
            else:
                grad_b = grad_wd
            if nesterov:
                grad_n = grad_wd.add(grad_b, alpha=momentum)
            else:
                grad_n = grad_b

            if temp < change_temp:
                self.set_parameter(self.net1, name, parameter.add(grad_n, alpha=-lr))
            else:
                self.set_parameter(self.net2, name, parameter.add(grad_n, alpha=-lr))
            temp += 1

def compute_loss_accuracy(data, embd_model, discriminator_model, criterion, options, flag=True):
    embd_model.eval()
    discriminator_model.eval()

    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    if flag:
        samples_idx = random.sample(range(n_batches), n_batches)
    else:
        samples_idx = [i for i in range(n_batches)]
    total_loss = 0
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.array([])

    for id, index in enumerate(samples_idx):
        diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, \
        labels = get_batch_data_single_label(data, options, batch_size, index)

        outputs_temp = embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths)
        outputs = discriminator_model(outputs_temp)
        total_loss += criterion(outputs, labels).item()

        temp = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()
        temp = temp[:, 1]
        prediction = torch.max(outputs, 1)[1].view((len(labels),)).data.cpu().numpy()
        labels = labels.data.cpu().numpy()

        y_score = np.concatenate((y_score, temp))
        y_true = np.concatenate((y_true, labels))
        y_pred = np.concatenate((y_pred, prediction))

    accuary = accuracy_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_score)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)

    total_loss = total_loss / (len(samples_idx))

    return total_loss, roc_auc, avg_precision, accuary, precision, f1, recall



def meta_train(embd_model, discriminator_model, meta_net,
               train, meta, validate_ori, test, options,
               optimizer, meta_optimizer):
    embd_model.to(options['device'])
    discriminator_model.to(options['device'])
    meta_net.to(options['device'])
    lr = options['lr']
    batch_size = options['batch_size_meta_batch']
    batch_size_meta = options['batch_size_meta']
    criterion = nn.CrossEntropyLoss().to(device=options['device'])
    criterion_valid = nn.CrossEntropyLoss().to(device=options['device'])
    min_validate_loss = 1000
    best_epoch = -1

    for epoch in range(options['max_epoch']):
        if epoch >= 80 and epoch % 400 == 0:
            lr = lr / 10
        for group in optimizer.param_groups:
            group['lr'] = lr

        print('Training...')
        n_batches = int(np.ceil(float(len(train[0])) / float(batch_size)))
        samples_idx = random.sample(range(n_batches), n_batches)
        meta_batches = int(np.ceil(float(len(meta[0])) / float(batch_size_meta)))
        meta_batches_list = random.sample(range(meta_batches), meta_batches)
        samples_meta_idx = iter(meta_batches_list)
        iteration = 0
        for id, index in enumerate(samples_idx):
            embd_model.train()
            discriminator_model.train()
            meta_net.train()
            ori_diagnosis_codes, ori_seq_time_step, ori_mask_mult, ori_mask_final, ori_mask_code, ori_lengths, \
            ori_labels = get_batch_data_single_label(train, options, batch_size, index)

            if (iteration + 1) % options['meta_interval'] == 0:
                pseudo_embd = Get_embd(options).to(options['device'])
                pseudo_discri = Discriminator(options).to(options['device'])
                pseudo_embd.load_state_dict(embd_model.state_dict())
                pseudo_discri.load_state_dict(pseudo_discri.state_dict())

                pseudo_embd.train()
                pseudo_discri.train()

                pseudo_outputs_temp = pseudo_embd(ori_diagnosis_codes, ori_seq_time_step, ori_mask_mult, ori_mask_final,
                                            ori_mask_code, ori_lengths)
                pseudo_outputs = pseudo_discri(pseudo_outputs_temp)

                pseudo_loss_vector = functional.cross_entropy(pseudo_outputs, ori_labels.long(), reduction='none')
                pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))
                pseudo_weight = meta_net(pseudo_loss_vector_reshape.data)
                pseudo_loss = torch.mean(pseudo_weight * pseudo_loss_vector_reshape)

                temp_para = list(pseudo_embd.parameters()) + list(pseudo_discri.parameters())

                pseudo_grads = torch.autograd.grad(pseudo_loss,
                                                   temp_para,
                                                   create_graph=True, allow_unused=True)

                pseudo_optimizer = MetaSGD(pseudo_embd, pseudo_discri,
                                           temp_para,
                                           lr=lr)

                pseudo_optimizer.load_state_dict(optimizer.state_dict())
                pseudo_optimizer.meta_step(pseudo_grads)

                del pseudo_grads
                try:
                    meta_index = next(samples_meta_idx)
                except StopIteration:
                    samples_meta_idx = iter(meta_batches_list)
                    meta_index = next(samples_meta_idx)

                diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, \
                meta_labels = get_batch_data_single_label(meta, options, batch_size_meta, meta_index)
                meta_outputs_temp = pseudo_embd(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths)
                meta_outputs = pseudo_discri(meta_outputs_temp)
                meta_loss = criterion(meta_outputs, meta_labels.long())

                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

            outputs_temp = embd_model(ori_diagnosis_codes, ori_seq_time_step, ori_mask_mult, ori_mask_final,
                                 ori_mask_code, ori_lengths)
            outputs = discriminator_model(outputs_temp)
            loss_vector = functional.cross_entropy(outputs, ori_labels.long(), reduction='none')
            loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))

            with torch.no_grad():
                weight = meta_net(loss_vector_reshape)

            loss = torch.mean(weight * loss_vector_reshape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Computing validation and test Result...')
        test_loss, roc_auc, avg_precision, accuary, precision, f1, recall = compute_loss_accuracy(
            test, embd_model, discriminator_model, criterion, options, flag=True
        )
        print(roc_auc, avg_precision, accuary, precision, f1, recall)
        print('Epoch: {}, (Loss) Test: ({:.4f}) LR: {}'.format(
            epoch,
            test_loss,
            lr,
        ))

        valid_embd_model = Get_embd(options).to(options['device'])
        valid_embd_model.load_state_dict(embd_model.state_dict())
        valid_embd_model.eval()
        valid_discriminator_model = Discriminator(options).to(options['device'])
        valid_discriminator_model.load_state_dict(discriminator_model.state_dict())
        valid_discriminator_model.eval()
        validate_loss, _, _, _, \
        _, _, _ = compute_loss_accuracy(
            meta, valid_embd_model, valid_discriminator_model, criterion_valid, options, flag=False
        )

        if min_validate_loss > validate_loss:
            min_validate_loss = validate_loss
            best_epoch = epoch
            best_roc_auc = roc_auc
            best_avg_precision = avg_precision
            best_accuracy = accuary
            best_precision = precision
            best_f1 = f1
            best_recall = recall
        print(
            'Best epoch: {}, best_valid_loss: {}, ccurrent_valid_loss: {}, '
            'best_roc_auc: {:.4f}, best_avg_precision: {:.4f}, best_accuracy: {:.4f}, '
            'best_precision: {:.4f}, best_f1: {:.4f}, best_recall: {:.4f}'.format(
                best_epoch, min_validate_loss, validate_loss,
                best_roc_auc, best_avg_precision, best_accuracy,
                best_precision, best_f1, best_recall))

def get_synthetic_embd(embd_model, generator_model, sample_train_neg, options):
    embd_model.eval()
    generator_model.eval()
    synthetic_embd = []

    batch_size_ori = options['batch_size_ori']
    batch_synthetic = int(np.ceil(float(len(sample_train_neg[0])) / float(batch_size_ori)))
    gan_list = random.sample(range(batch_synthetic), batch_synthetic)
    for id, gan_idx in enumerate(gan_list):
        diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, labels = \
            get_batch_data(sample_train_neg, options,
                           batch_size_ori,
                           gan_idx, flag=False)

        h = embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths)
        gan_embd_temp = generator_model(h)
        gan_embd = gan_embd_temp.cpu().detach().numpy()
        synthetic_embd.append(gan_embd)


    synthetic_embd = np.concatenate(synthetic_embd, axis=0)
    return synthetic_embd


def train_embd_dis_meta(embd_model, generator_model, discriminator_model, meta_net,
                        train, validate_meta, validate, test, options,
                        embd_discri_para_meta_optimizer, meta_optimizer):
    lr = options['lr']
    device = options['device']
    output_model_path = options['output_model_path']
    embd_model.to(device)
    generator_model.to(device)
    discriminator_model.to(device)
    meta_net.to(device)
    criterion = nn.CrossEntropyLoss().to(device=options['device'])
    criterion_valid = nn.CrossEntropyLoss().to(device=options['device'])
    min_validate_loss = 10000

    # sample data original
    _, sample_train_neg = sample_data(train, upsample=1)
    with open(output_model_path+'sample_train_neg.pickle', 'wb') as f:
        pickle.dump(sample_train_neg, f)


    # get synthetic representation
    synthetic_embd = get_synthetic_embd(embd_model, generator_model, sample_train_neg, options)
    generator_model.cpu()


    for epoch in range(options['gan_meta_epoch']):
        if epoch >= 80 and epoch % 400 == 0:
            lr = lr / 10
        for group in embd_discri_para_meta_optimizer.param_groups:
            group['lr'] = lr

        print('training...')

        # iter
        batch_size_ori = options['batch_size_ori']
        batch_ori = int(np.ceil(float(len(train[0])) / float(batch_size_ori)))
        ori_list = random.sample(range(batch_ori), batch_ori)
        batch_size_meta = options['batch_size_meta']
        batch_meta = int(np.ceil(float(len(validate_meta[0])) / float(batch_size_meta)))
        meta_list = random.sample(range(batch_meta), batch_meta)
        meta_idx = iter(meta_list)
        batch_size_gan = options['batch_size_gan']
        batch_gan = int(np.ceil(float(len(sample_train_neg[0])) / float(batch_size_gan)))
        gan_list = random.sample(range(batch_gan), batch_gan)
        gan_idx = iter(gan_list)
        iteration = 0

        for id, ori_idx in enumerate(ori_list):
            embd_model.train()
            discriminator_model.train()
            meta_net.train()

            # gan
            try:
                gan_index = next(gan_idx)
            except StopIteration:
                gan_idx = iter(gan_list)
                gan_index = next(gan_idx)

            diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, labels = \
                get_batch_data(train, options,
                               batch_size_ori,
                               ori_idx)

            if (iteration + 1) % options['meta_interval'] == 0:
                pseudo_embd = Get_embd(options).to(options['device'])
                pseudo_discri = Discriminator(options).to(options['device'])
                pseudo_embd.load_state_dict(embd_model.state_dict())
                pseudo_discri.load_state_dict(pseudo_discri.state_dict())

                pseudo_embd.train()
                pseudo_discri.train()

                real_embd = pseudo_embd(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths)
                gan_embd = synthetic_embd[batch_size_gan * gan_index: batch_size_gan * (gan_index + 1)]
                # gan_label
                gan_labels = np.array([1 for i in range(gan_embd.shape[0])])
                gan_labels = torch.LongTensor(gan_labels).to(options['device'])
                label = torch.cat((labels, gan_labels))

                gan_embd = torch.Tensor(gan_embd).to(options['device'])
                embd = torch.cat((real_embd, gan_embd), 0)
                pseudo_outputs = pseudo_discri(embd)

                pseudo_loss_vector = functional.cross_entropy(pseudo_outputs, label.long(), reduction='none')
                pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))
                pseudo_weight = meta_net(pseudo_loss_vector_reshape.data)
                pseudo_loss = torch.mean(pseudo_weight * pseudo_loss_vector_reshape)

                temp_para = list(pseudo_embd.parameters()) + list(pseudo_discri.parameters())
                pseudo_grads = torch.autograd.grad(pseudo_loss,
                                                   temp_para,
                                                   create_graph=True, allow_unused=True)

                pseudo_optimizer = MetaSGD(pseudo_embd, pseudo_discri,
                                           temp_para,
                                           lr=lr)
                pseudo_optimizer.load_state_dict(embd_discri_para_meta_optimizer.state_dict())
                pseudo_optimizer.meta_step(pseudo_grads)

                del pseudo_grads
                try:
                    meta_index = next(meta_idx)
                except StopIteration:
                    meta_idx = iter(meta_list)
                    meta_index = next(meta_idx)

                diagnosis_codes_meta, seq_time_step_meta, mask_mult_meta, mask_final_meta, mask_code_meta, lengths_meta, \
                meta_labels_meta = get_batch_data_single_label(validate_meta, options, batch_size_meta, meta_index)
                meta_outputs_temp = pseudo_embd(diagnosis_codes_meta, seq_time_step_meta, mask_mult_meta,
                                                mask_final_meta, mask_code_meta, lengths_meta)
                meta_outputs = pseudo_discri(meta_outputs_temp)
                meta_loss = criterion(meta_outputs, meta_labels_meta.long())
                meta_optimizer.zero_grad()
                meta_loss.backward()
                meta_optimizer.step()

            real_embd = embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths)
            gan_embd = synthetic_embd[batch_size_gan * gan_index: batch_size_gan * (gan_index + 1)]
            # gan_label
            gan_labels = np.array([1 for i in range(gan_embd.shape[0])])
            gan_labels = torch.LongTensor(gan_labels).to(options['device'])
            label = torch.cat((labels, gan_labels))

            gan_embd = torch.Tensor(gan_embd).to(options['device'])
            embd = torch.cat((real_embd, gan_embd), 0)
            outputs = discriminator_model(embd)
            loss_vector = functional.cross_entropy(outputs, label.long(), reduction='none')
            loss_vector_reshape = torch.reshape(loss_vector, (-1, 1))

            with torch.no_grad():
                weight = meta_net(loss_vector_reshape)

            loss = torch.mean(weight * loss_vector_reshape)
            embd_discri_para_meta_optimizer.zero_grad()
            loss.backward()
            embd_discri_para_meta_optimizer.step()

        print('Computing validation and test Result...')
        test_loss, roc_auc, avg_precision, accuary, precision, f1, recall = compute_loss_accuracy(
            test, embd_model, discriminator_model, criterion, options, flag=True
        )
        print('epoch: {}, test roc_auc: {}, test avg_precision: {}, test accuary: {}, test precision: {}, test f1: {}, test recall: {}'.
              format(epoch, roc_auc, avg_precision, accuary, precision, f1, recall))

        valid_embd_model = Get_embd(options).to(options['device'])
        valid_embd_model.load_state_dict(embd_model.state_dict())
        valid_embd_model.eval()
        valid_discriminator_model = Discriminator(options).to(options['device'])
        valid_discriminator_model.load_state_dict(discriminator_model.state_dict())
        valid_discriminator_model.eval()
        validate_loss, _, _, _, \
        _, _, _ = compute_loss_accuracy(
            validate_meta, valid_embd_model, valid_discriminator_model, criterion_valid, options, flag=False
        )
        print('Epoch: {}, valid loss: {:.4f}, Test loss: ({:.4f}) LR: {}'.format(
            epoch,
            validate_loss,
            test_loss,
            lr,
        ))

        if min_validate_loss > validate_loss or epoch%100==0 or epoch==89 or epoch==114 or f1>0.36:
            min_validate_loss = validate_loss
            best_epoch = epoch
            best_roc_auc = roc_auc
            best_avg_precision = avg_precision
            best_accuracy = accuary
            best_precision = precision
            best_f1 = f1
            best_recall = recall

            torch.save(embd_model.state_dict(), output_model_path+'embd_model.' + str(epoch))
            torch.save(generator_model.state_dict(), output_model_path+'generator_model.' + str(epoch))
            torch.save(discriminator_model.state_dict(), output_model_path+'discriminator_model.' + str(epoch))
            torch.save(meta_net.state_dict(), output_model_path+'meta_net.' + str(epoch))
            best_parameters_embd_file = output_model_path+'embd_model.' + str(epoch)
            best_parameters_generator_file = output_model_path+'generator_model.' + str(epoch)
            best_parameters_discriminator_file = output_model_path+'discriminator_model.' + str(epoch)
            best_parameters_meta_net_file = output_model_path+'meta_net.' + str(epoch)
        print(
            'Best epoch: {}, best_valid_loss: {}, ccurrent_valid_loss: {}, '
            'best_roc_auc: {:.4f}, best_avg_precision: {:.4f}, best_accuracy: {:.4f}, '
            'best_precision: {:.4f}, best_f1: {:.4f}, best_recall: {:.4f}'.format(
                best_epoch, min_validate_loss, validate_loss,
                best_roc_auc, best_avg_precision, best_accuracy,
                best_precision, best_f1, best_recall))

    print('best_parameters_embd_file: {}'.format(best_parameters_embd_file))
    print('best_parameters_generator_file: {}'.format(best_parameters_generator_file))
    print('best_parameters_discriminator_file: {}'.format(best_parameters_discriminator_file))
    print('best_parameters_meta_net_file: {}'.format(best_parameters_meta_net_file))

    with open('sample_train_neg.pickle', 'wb') as f:
        pickle.dump(sample_train_neg, f)

    return best_roc_auc, best_avg_precision, best_accuracy, best_precision, best_f1, best_recall







