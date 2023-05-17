import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import torch
import copy
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, \
    recall_score, f1_score, roc_auc_score

def load_data(training_file, validation_file, testing_file):
    train = np.array(pickle.load(open(training_file, 'rb')))
    validate = np.array(pickle.load(open(validation_file, 'rb')))
    test = np.array(pickle.load(open(testing_file, 'rb')))
    return train, validate, test

def cut_data(training_file, validation_file, testing_file):
    train = list(pickle.load(open(training_file, 'rb')))
    validate = list(pickle.load(open(validation_file, 'rb')))
    test = list(pickle.load(open(testing_file, 'rb')))
    for dataset in [train, validate, test]:
        dataset[0] = dataset[0][0: len(dataset[0]) // 18]
        dataset[1] = dataset[1][0: len(dataset[1]) // 18]
        dataset[2] = dataset[2][0: len(dataset[2]) // 18]
    return train, validate, test


def pad_time(seq_time_step):
    lengths = np.array([len(seq) for seq in seq_time_step])
    maxlen = np.max(lengths)
    for k in range(len(seq_time_step)):
        while len(seq_time_step[k]) < maxlen:
            # seq_time_step[k].append(100000)
            seq_time_step[k].append(0)

    return seq_time_step

def pad_matrix_new(seq_diagnosis_codes, seq_labels, options):
    lengths = np.array([len(seq) for seq in seq_diagnosis_codes])
    n_samples = len(seq_diagnosis_codes)
    maxcode_ori = options['maxcode']
    maxlen = np.max(lengths)
    lengths_code = []
    for seq in seq_diagnosis_codes:
        for code_set in seq:
            lengths_code.append(len(code_set))
    lengths_code = np.array(lengths_code)
    maxcode = np.max(lengths_code)
    maxcode = min(maxcode, maxcode_ori)

    batch_diagnosis_codes = np.zeros((n_samples, maxlen, maxcode), dtype=np.int64) + options['n_diagnosis_codes']
    batch_mask = np.zeros((n_samples, maxlen), dtype=np.float32)
    batch_mask_code = np.zeros((n_samples, maxlen, maxcode), dtype=np.float32)
    batch_mask_final = np.zeros((n_samples, maxlen), dtype=np.float32)

    for bid, seq in enumerate(seq_diagnosis_codes):
        for pid, subseq in enumerate(seq):
            for tid, code in enumerate(subseq):
                if tid > maxcode - 1:
                    continue
                batch_diagnosis_codes[bid, pid, tid] = code
                batch_mask_code[bid, pid, tid] = 1


    for i in range(n_samples):
        batch_mask[i, 0:lengths[i]] = 1
        max_visit = lengths[i] - 1
        batch_mask_final[i, max_visit] = 1

    batch_labels = np.array(seq_labels, dtype=np.int64)

    return batch_diagnosis_codes, batch_labels, batch_mask, batch_mask_final, batch_mask_code

def select_data(train, pos_flag):
    target_label = int(pos_flag)
    labels = train[-1]
    selected_train_diagnosis_codes = []
    selected_train_time_step = []
    selected_train_label = []
    for i in range(len(labels)):
        if int(labels[i])==target_label:
            selected_train_diagnosis_codes.append(train[0][i])
            selected_train_time_step.append(train[1][i])
            selected_train_label.append(train[-1][i])

    selected_train = np.array((selected_train_diagnosis_codes, selected_train_time_step, selected_train_label))
    return selected_train

def sample_data(train, upsample=1):
    # select positive
    train_pos = select_data(train, pos_flag=True)
    train_neg = select_data(train, pos_flag=False)

    # sample negative
    n_samples_ori = len(train_pos[0])
    n_samples = n_samples_ori * upsample
    index = random.sample(range(len(train_neg[0])), n_samples)
    diagnosis_codes = list(np.array(train_neg[0])[index])
    time_step = list(np.array(train_neg[1])[index])
    labels = list(np.array(train_neg[-1])[index])

    sample_train_neg = np.array((diagnosis_codes, time_step, labels))
    return train_pos, sample_train_neg

def get_batch_data_single_label(data, options, batch_size, index):
    max_len = options['maxlen']
    diagnosis_codes = data[0][batch_size * index: batch_size * (index + 1)]
    time_step = data[1][batch_size * index: batch_size * (index + 1)]
    batch_diagnosis_codes, batch_time_step = adjust_input(diagnosis_codes, time_step,
                                                          max_len)

    batch_labels = data[-1][batch_size * index: batch_size * (index + 1)]
    diagnosis_codes, labels, mask, mask_final, mask_code_ori = pad_matrix_new(
        batch_diagnosis_codes,
        batch_labels, options)

    lengths = torch.from_numpy(np.array([len(seq) for seq in batch_diagnosis_codes])).to(options['device'])

    diagnosis_codes = torch.LongTensor(diagnosis_codes).to(options['device'])
    mask_mult = torch.BoolTensor(1-mask).unsqueeze(2).to(options['device'])
    mask_final = torch.Tensor(mask_final).unsqueeze(2).to(options['device'])
    mask_code = torch.Tensor(mask_code_ori).unsqueeze(3).to(options['device'])
    labels = torch.LongTensor(labels).to(options['device'])

    batch_time_step = pad_time(batch_time_step)
    seq_time_step = (torch.FloatTensor(batch_time_step).unsqueeze(2) / 180).to(options['device'])

    return diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, labels

def pad_time(seq_time_step):
    lengths = np.array([len(seq) for seq in seq_time_step])
    maxlen = np.max(lengths)
    for k in range(len(seq_time_step)):
        while len(seq_time_step[k]) < maxlen:
            seq_time_step[k].append(100000)

    seq_time_step = np.array(list(seq_time_step))
    return seq_time_step


def get_batch_data(data1, options,
                   batch_size_1,
                   index_1, flag=True):
    max_len = 80
    diagnosis_codes = data1[0][batch_size_1 * index_1: batch_size_1 * (index_1 + 1)]
    time_step = data1[1][batch_size_1 * index_1: batch_size_1 * (index_1 + 1)]

    batch_diagnosis_codes, batch_time_step = adjust_input(diagnosis_codes, time_step,
                                                          max_len)

    batch_labels = data1[-1][batch_size_1 * index_1: batch_size_1 * (index_1 + 1)]
    lengths = torch.from_numpy(np.array([len(seq) for seq in batch_diagnosis_codes])).to(options['device'])

    if flag==False:
        batch_labels_new = np.array([1 for i in batch_labels])
    else:
        batch_labels_new = batch_labels

    diagnosis_codes, labels, mask, mask_final, mask_code_ori = pad_matrix_new(
        batch_diagnosis_codes,
        batch_labels_new, options)

    diagnosis_codes = torch.LongTensor(diagnosis_codes).to(options['device'])
    mask_mult = torch.BoolTensor(1-mask).unsqueeze(2).to(options['device'])
    mask_final = torch.Tensor(mask_final).unsqueeze(2).to(options['device'])
    mask_code = torch.Tensor(mask_code_ori).unsqueeze(3).to(options['device'])
    labels = torch.LongTensor(labels).to(options['device'])

    batch_time_step = pad_time(batch_time_step)
    seq_time_step = (torch.FloatTensor(batch_time_step).unsqueeze(2) / 180).to(options['device'])

    return diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, labels

def get_balanced_batch_data(data1, data2, options,
                   batch_size_1, batch_size_2,
                   index_1, index_2,
                   flag=False):
    max_len = options['maxlen']
    diagnosis_codes_pos = data1[0][batch_size_1 * index_1: batch_size_1 * (index_1 + 1)]
    time_step_pos = data1[1][batch_size_1 * index_1: batch_size_1 * (index_1 + 1)]

    diagnosis_codes_neg = data2[0][batch_size_2 * index_2: batch_size_2 * (index_2 + 1)]
    time_step_neg = data2[1][batch_size_2 * index_2: batch_size_2 * (index_2 + 1)]

    diagnosis_codes = np.concatenate((diagnosis_codes_pos, diagnosis_codes_neg), axis=0)
    time_step = np.concatenate((time_step_pos, time_step_neg), axis=0)

    batch_diagnosis_codes, batch_time_step = adjust_input(diagnosis_codes, time_step,
                                                          max_len)

    labels_pos = data1[-1][batch_size_1 * index_1: batch_size_1 * (index_1 + 1)]
    labels_neg_ori = data2[-1][batch_size_2 * index_2: batch_size_2 * (index_2 + 1)]
    real_size = labels_pos.size
    gan_size = labels_neg_ori.size
    if flag==False:
        labels_neg = labels_neg_ori
    else:
        labels_neg = np.array([1 for i in labels_neg_ori])
    batch_labels = np.concatenate((labels_pos, labels_neg), axis=0)

    lengths = torch.from_numpy(np.array([len(seq) for seq in batch_diagnosis_codes])).to(options['device'])

    diagnosis_codes, labels, mask, mask_final, mask_code_ori = pad_matrix_new(
        batch_diagnosis_codes,
        batch_labels, options)

    diagnosis_codes = torch.LongTensor(diagnosis_codes).to(options['device'])
    mask_mult = torch.BoolTensor(1-mask).unsqueeze(2).to(options['device'])
    mask_final = torch.Tensor(mask_final).unsqueeze(2).to(options['device'])
    mask_code = torch.Tensor(mask_code_ori).unsqueeze(3).to(options['device'])
    labels = torch.LongTensor(labels).to(options['device'])

    batch_time_step = pad_time(batch_time_step)
    seq_time_step = (torch.FloatTensor(batch_time_step).unsqueeze(2) / 180).to(options['device'])

    return diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, labels, real_size, gan_size


def calculate_cost_tran_pretrain(embd_model, generator_model, discriminator_model, data, options,
                                 loss_function):
    embd_model.eval()
    generator_model.eval()
    discriminator_model.eval()

    batch_size = int(options['batch_size'] / 2)
    cost_sum = 0.0

    # sample data
    data_pos, sample_data_neg = sample_data(data)
    n_batches = int(np.ceil(float(len(data_pos[0])) / float(batch_size)))
    for index in range(n_batches):
        diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, \
        labels, _, _ = get_balanced_batch_data(data_pos, sample_data_neg, options,
                                batch_size, batch_size,
                                index, index)
        h = embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths)
        # # get XN_noise part
        # XT = h[:batch_size, ]
        # XN_noise = h[batch_size:, ]
        # XN = generator_model(XN_noise)
        # X = torch.cat((XT, XN), 0)
        # logits = discriminator_model(X)
        logits = discriminator_model(h)

        loss = loss_function(logits, labels)
        cost_sum += loss.cpu().data.numpy()

    embd_model.train()
    generator_model.train()
    discriminator_model.train()


    return cost_sum / n_batches



def calculate_cost_tran_train(embd_model, generator_model, discriminator_model,
                              data, options, bceloss):
    embd_model.eval()
    generator_model.eval()
    discriminator_model.eval()

    batch_size = options['batch_size']
    cost_sum_D = 0.0
    cost_sum_G = 0.0

    # sample data
    data_pos, sample_data_neg = sample_data(data)
    n_batches = int(np.ceil(float(len(data_pos[0])) / float(batch_size)))
    for index in range(n_batches):
        # D network
        ## all-real batch
        diagnosis_codes_pos, seq_time_step_pos, mask_mult_pos, mask_final_pos, \
        mask_code_pos, lengths_pos, labels_pos = get_batch_data_single_label(data_pos, options, batch_size, index)
        h_pos = embd_model(diagnosis_codes_pos, seq_time_step_pos, mask_mult_pos, mask_final_pos,
                           mask_code_pos, lengths_pos)
        logits_pos = discriminator_model(h_pos)
        errD_pos = bceloss(logits_pos, labels_pos)

        ## all-fake batch
        diagnosis_codes_neg, seq_time_step_neg, mask_mult_neg, mask_final_neg, \
        mask_code_neg, lengths_neg, labels_neg = get_batch_data_single_label(sample_data_neg, options, batch_size, index)
        h_neg = embd_model(diagnosis_codes_neg, seq_time_step_neg, mask_mult_neg, mask_final_neg,
                           mask_code_neg, lengths_neg)
        h_prime_neg = generator_model(h_neg)
        logits_neg = discriminator_model(h_prime_neg)
        errD_neg = bceloss(logits_neg, labels_neg)
        errD = errD_pos + errD_neg

        cost_sum_D += errD.cpu().data.numpy()

        ############################
        #  G network
        ###########################
        logits_neg_new = discriminator_model(h_prime_neg)
        temp_batch = logits_neg_new.size(dim=0)
        labels_gen = torch.full((temp_batch,), 1, dtype=torch.int64, device=options['device'])
        errG = compute_G_loss(h_neg, h_prime_neg, logits_neg_new, labels_gen, bceloss)

        cost_sum_G += errG.cpu().data.numpy()

    embd_model.train()
    generator_model.train()
    discriminator_model.train()
    return cost_sum_D / n_batches, cost_sum_G / n_batches


def calculate_cost_tran_train_discriminator(embd_model, generator_model, discriminator_model,
                                            data, options, focal_loss, flag=False):
    embd_model.eval()
    generator_model.eval()
    discriminator_model.eval()

    batch_size = options['batch_size']
    n_batches = int(np.ceil(float(len(data[0])) / float(batch_size)))
    cost_sum = 0.0

    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.array([])

    for index in range(n_batches):
        diagnosis_codes, seq_time_step, mask_mult, mask_final, \
        mask_code, lengths, labels = get_batch_data_single_label(data, options, batch_size, index)
        h = embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final,
                       mask_code, lengths)
        logits = discriminator_model(h)

        temp = nn.functional.softmax(logits).data.cpu().numpy()
        temp = temp[:, 1]
        prediction = torch.max(logits, 1)[1].view((len(labels),)).data.cpu().numpy()
        labels = labels.data.cpu().numpy()

        y_score = np.concatenate((y_score, temp))
        y_true = np.concatenate((y_true, labels))
        y_pred = np.concatenate((y_pred, prediction))

        loss = focal_loss(logits, labels)
        cost_sum += loss.cpu().data.numpy()

    accuary = accuracy_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_score)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)

    print('roc_auc: {}, avg_precision: {}, accuary: {}, precision: {}, f1: {}, recall: {}'.format(
    roc_auc, avg_precision, accuary, precision, f1, recall))

    embd_model.train()
    generator_model.eval()
    discriminator_model.train()
    return cost_sum / n_batches



def adjust_input(batch_diagnosis_codes, batch_time_step, max_len):
    interval_list_new = []
    for interval in batch_time_step:
        new_interval = []
        for i in range(1, len(interval)):
            new_interval.append(sum(interval[i:]))
        new_interval.extend([0])
        interval_list_new.append(new_interval)

    batch_time_step = interval_list_new
    batch_diagnosis_codes = copy.deepcopy(batch_diagnosis_codes)
    for ind in range(len(batch_diagnosis_codes)):
        if len(batch_diagnosis_codes[ind]) > max_len:
            batch_diagnosis_codes[ind] = batch_diagnosis_codes[ind][-(max_len):]
            batch_time_step[ind] = batch_time_step[ind][-(max_len):]

    batch_time_step = np.array(interval_list_new)
    return batch_diagnosis_codes, batch_time_step

def compute_G_loss(h_neg, h_prime_neg, logits_neg_new, labels_gen, bceloss):
    h_temp = h_neg - h_prime_neg
    temp1 = torch.norm(h_temp, p=2, dim=1).mean()
    temp2 = bceloss(logits_neg_new, labels_gen)
    return 0.05*temp1 + temp2
