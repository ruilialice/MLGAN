import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import random
import time
import shutil
from models.units import sample_data, \
    get_balanced_batch_data, calculate_cost_tran_pretrain, get_batch_data_single_label, \
    compute_G_loss, calculate_cost_tran_train, \
    calculate_cost_tran_train_discriminator
import os
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, \
    recall_score, f1_score, roc_auc_score


class CrossEntropyLoss_define(nn.Module):
    def __init__(self, options=None):
        super(CrossEntropyLoss_define, self).__init__()
        self.device = options['device']
        self.size_average = True

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = nn.functional.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()

        batch_loss = -log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, options=None):
        super(FocalLoss, self).__init__()
        # if alpha is None:
        #     self.alpha = Variable(torch.ones(class_num, 1), requires_grad=True)
        # else:
        #     if isinstance(alpha, Variable):
        #         self.alpha = alpha
        #     else:
        #         self.alpha = Variable(alpha)
        # if alpha is None:
        #     self.alpha = torch.Tensor([1, 1])
        # self.alpha = torch.Tensor([alpha ,, 1-alpha])
        self.alpha = torch.Tensor([1, 1])
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.device = options['device']

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = nn.functional.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        self.alpha = self.alpha.to(self.device)
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def pretrain(embd_model, generator_model, discriminator_model,
             train, valid, test, options,
             embd_discri_para_optimizer):
    batch_size = int(options['batch_size']/2)
    device = options['device']
    pretrain_epoch = options['pretrain_epoch']

    output_file = options['output_file']
    model_name = options['model_name']

    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    best_test_cost = 0.0
    epoch_duaration = 0.0
    best_epoch = 0.0

    bceloss = CrossEntropyLoss_define(options)
    embd_model.to(device)
    generator_model.to(device)
    discriminator_model.to(device)
    embd_model.train()
    generator_model.eval()
    discriminator_model.train()

    # sample data
    train_pos, sample_train_neg = sample_data(train)

    n_batches = int(np.ceil(float(len(train_pos[0])) / float(batch_size)))
    for epoch in range(pretrain_epoch):
        iteration = 0
        cost_vector = []
        start_time = time.time()
        samples = random.sample(range(n_batches), n_batches)

        for index in samples:
            embd_discri_para_optimizer.zero_grad()
            diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, \
            labels, _, _ = get_balanced_batch_data(train_pos, sample_train_neg, options,
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

            loss = bceloss(logits, labels)
            loss.backward()
            embd_discri_para_optimizer.step()

            cost_vector.append(loss.cpu().data.numpy())

            if (iteration % 50 == 0):
                print('epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, loss.cpu().data.numpy()))
            # print('epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, loss.cpu().data.numpy()))
            iteration += 1

        duration = time.time() - start_time
        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, np.mean(cost_vector), duration))

        train_cost = np.mean(cost_vector)
        validate_cost = calculate_cost_tran_pretrain(embd_model, generator_model, discriminator_model,
                                                     valid, options, bceloss)
        test_cost = calculate_cost_tran_pretrain(embd_model, generator_model, discriminator_model,
                                                     valid, options, bceloss)
        print('epoch:%d, validate_cost:%f, duration:%f' % (epoch, validate_cost, duration))
        epoch_duaration += duration

        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_test_cost = test_cost
            best_epoch = epoch

            shutil.rmtree(output_file)
            os.mkdir(output_file)

            torch.save(embd_model.state_dict(), output_file + model_name + 'embd_model.' + str(epoch))
            torch.save(generator_model.state_dict(), output_file + model_name + 'generator_model.' + str(epoch))
            torch.save(discriminator_model.state_dict(), output_file + model_name + 'discriminator_model.' + str(epoch))
            best_parameters_embd_file = output_file + model_name + 'embd_model.' + str(epoch)
            best_parameters_generator_file = output_file + model_name + 'generator_model.' + str(epoch)
            best_parameters_discriminator_file = output_file + model_name + 'discriminator_model.' + str(epoch)

        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (
        best_epoch, best_train_cost, best_validate_cost, best_test_cost)
        print(buf)

    # testing
    embd_model.load_state_dict(torch.load(best_parameters_embd_file))
    generator_model.load_state_dict(torch.load(best_parameters_generator_file))
    discriminator_model.load_state_dict(torch.load(best_parameters_discriminator_file))
    embd_model.eval()
    generator_model.eval()
    discriminator_model.eval()

    batch_size_test = int(batch_size*2)
    n_batches = int(np.ceil(float(len(test[0])) / float(batch_size_test)))
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.array([])
    for index in range(n_batches):
        diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, \
        labels = get_batch_data_single_label(test, options, batch_size, index)
        h = embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths)
        logit = discriminator_model(h)

        temp = nn.functional.softmax(logit).data.cpu().numpy()
        temp = temp[:, 1]
        prediction = torch.max(logit, 1)[1].view((len(labels),)).data.cpu().numpy()
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

    print(roc_auc, avg_precision, accuary, precision, f1, recall)

    return embd_model, generator_model, discriminator_model

def c_train(embd_model, generator_model, discriminator_model, train, valid, test,
          options, discriminator_para_optimizer, generator_para_optimizer):

    device = options['device']
    embd_model.train()
    generator_model.train()
    discriminator_model.train()
    embd_model.to(device)
    generator_model.to(device)
    discriminator_model.to(device)

    n_epoch_g = options['n_epoch_g']
    batch_size = int(options['batch_size'])
    bceloss = CrossEntropyLoss_define(options)

    print('#############################')
    print('####  start train    ########')
    print('#############################')

    for epoch in range(n_epoch_g):
        # # sample data
        train_pos, sample_train_neg = sample_data(train)

        n_batches = int(np.ceil(float(len(train_pos[0])) / float(batch_size)))
        samples = random.sample(range(n_batches), n_batches)
        iteration = 0
        cost_vector_D = []
        cost_vector_G = []
        start_time = time.time()

        for index in samples:
            # (1) Update D network
            ## Train with all-real batch
            discriminator_para_optimizer.zero_grad()
            diagnosis_codes_pos, seq_time_step_pos, mask_mult_pos, mask_final_pos, mask_code_pos, lengths_pos, \
            labels_pos = get_batch_data_single_label(train_pos, options, batch_size, index)
            h_pos = embd_model(diagnosis_codes_pos, seq_time_step_pos, mask_mult_pos, mask_final_pos,
                               mask_code_pos, lengths_pos)
            logits_pos = discriminator_model(h_pos)
            errD_pos = bceloss(logits_pos, labels_pos)
            errD_pos.backward()
            # D_x = output.mean().item()
            ## Train with all-fake batch
            diagnosis_codes_neg, seq_time_step_neg, mask_mult_neg, mask_final_neg, mask_code_neg, lengths_neg, \
            labels_neg = get_batch_data_single_label(sample_train_neg, options, batch_size, index)
            h_neg = embd_model(diagnosis_codes_neg, seq_time_step_neg, mask_mult_neg, mask_final_neg,
                               mask_code_neg, lengths_neg)
            h_prime_neg = generator_model(h_neg)
            logits_neg = discriminator_model(h_prime_neg.detach())
            errD_neg = bceloss(logits_neg, labels_neg)
            errD_neg.backward()
            errD = errD_pos + errD_neg
            discriminator_para_optimizer.step()

            cost_vector_D.append(errD.cpu().data.numpy())

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator_para_optimizer.zero_grad()
            logits_neg_new = discriminator_model(h_prime_neg)
            temp_batch = logits_neg_new.size(dim=0)
            labels_gen = torch.full((temp_batch,), 1, dtype=torch.int64, device=options['device'])
            errG = compute_G_loss(h_neg, h_prime_neg, logits_neg_new, labels_gen, bceloss)
            errG.backward()
            generator_para_optimizer.step()

            cost_vector_G.append(errG.cpu().data.numpy())

            if (iteration % 5 == 0):
                print('epoch:%d, iteration:%d/%d, generator loss:%f, discriminator loss:%f'
                      % (epoch, iteration, n_batches, errG.cpu().data.numpy(), errD.cpu().data.numpy()))
            iteration += 1

        duration = time.time() - start_time
        print('epoch:%d, mean_cost D:%f, mean_cost G:%f, duration:%f'
              % (epoch, np.mean(cost_vector_D), np.mean(cost_vector_G), duration))

        validate_cost_D, validate_cost_G = calculate_cost_tran_train(embd_model, generator_model, discriminator_model,
                                                  valid, options, bceloss)

        print('epoch:%d, validate_cost_D:%f, validate_cost_G:%f, duration:%f'
              % (epoch, validate_cost_D, validate_cost_G, duration))

        test_cost_D, test_cost_G = calculate_cost_tran_train(embd_model, generator_model, discriminator_model,
                                                             test, options, bceloss)

        print('epoch:%d, test_cost_D:%f, test_cost_G:%f, duration:%f'
              % (epoch, test_cost_D, test_cost_G, duration))

    return embd_model, generator_model, discriminator_model

def c_train_together_discriminator(embd_model, generator_model, discriminator_model, train,
                                       validate, test, options, embd_discri_para_optimizer):
    device = options['device']
    embd_model.train()
    generator_model.train()
    discriminator_model.train()
    embd_model.to(device)
    generator_model.to(device)
    discriminator_model.to(device)

    n_epoch_together = options['n_epoch_together']
    batch_size = int(options['batch_size'])
    output_file = options['output_file']
    model_name = options['model_name']
    focal_loss = FocalLoss(2, gamma=2., alpha=.25, options=options)

    best_train_cost = 0.0
    best_validate_cost = 100000000.0
    best_test_cost = 0.0
    best_epoch = 0.0

    # sample data original
    train_pos, sample_train_neg = sample_data(train)

    print('##########################################')
    print('########  start together train    ########')
    print('##########################################')

    for epoch in range(n_epoch_together):
        # # sample data
        # train_pos, sample_train_neg = sample_data(train)

        # train real data
        n_batches = int(np.ceil(float(len(train[0])) / float(batch_size)))
        samples = random.sample(range(n_batches), n_batches)
        cost_vector = []
        iteration = 0
        start_time = time.time()

        for index in samples:
            embd_discri_para_optimizer.zero_grad()
            diagnosis_codes, seq_time_step, mask_mult, mask_final, \
            mask_code, lengths, labels = get_batch_data_single_label(train, options, batch_size, index)
            h = embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final,
                           mask_code, lengths)
            logits = discriminator_model(h)
            loss = focal_loss(logits, labels)
            loss.backward()
            embd_discri_para_optimizer.step()
            cost_vector.append(loss.cpu().data.numpy())

            if (iteration % 50 == 0):
                print('epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, loss.cpu().data.numpy()))
            # print('epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, loss.cpu().data.numpy()))
            iteration += 1

        # train generated data
        iteration = 0
        n_batches_neg = int(np.ceil(float(len(sample_train_neg[0])) / float(batch_size)))
        samples = random.sample(range(n_batches_neg), n_batches_neg)
        for index in samples:
            embd_discri_para_optimizer.zero_grad()
            diagnosis_codes, seq_time_step, mask_mult, mask_final, \
            mask_code, lengths, labels = get_batch_data_single_label(sample_train_neg, options, batch_size, index)
            h = embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final,
                           mask_code, lengths)
            h_gen = generator_model(h)
            logits = discriminator_model(h_gen)
            temp_batch = logits.size(dim=0)
            labels_gen = torch.full((temp_batch,), 1, dtype=torch.int64, device=options['device'])
            loss = focal_loss(logits, labels_gen)
            loss.backward()
            embd_discri_para_optimizer.step()
            cost_vector.append(loss.cpu().data.numpy())

            if (iteration % 5 == 0):
                print('epoch:%d, iteration:%d/%d, cost:%f' % (epoch, iteration, n_batches, loss.cpu().data.numpy()))
            iteration += 1

        duration = time.time() - start_time
        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch, np.mean(cost_vector), duration))
        train_cost = np.mean(cost_vector)

        validate_cost = calculate_cost_tran_train_discriminator(embd_model, generator_model, discriminator_model,
                                                           validate, options, focal_loss)
        test_cost = calculate_cost_tran_train_discriminator(embd_model, generator_model, discriminator_model,
                                                           test, options, focal_loss)
        print('epoch:%d, validate_cost:%f, duration:%f' % (epoch, validate_cost, duration))

        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_test_cost = test_cost
            best_epoch = epoch

            shutil.rmtree(output_file)
            os.mkdir(output_file)

            torch.save(embd_model.state_dict(), output_file + model_name + 'embd_model_final.' + str(epoch))
            torch.save(discriminator_model.state_dict(), output_file + model_name + 'discriminator_model_final.' + str(epoch))
            best_parameters_embd_file = output_file + model_name + 'embd_model_final.' + str(epoch)
            best_parameters_discriminator_file = output_file + model_name + 'discriminator_model_final.' + str(epoch)

        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (
        best_epoch, best_train_cost, best_validate_cost, best_test_cost)
        print(buf)

    # testing
    embd_model.load_state_dict(torch.load(best_parameters_embd_file))
    discriminator_model.load_state_dict(torch.load(best_parameters_discriminator_file))
    embd_model.eval()
    discriminator_model.eval()

    n_batches = int(np.ceil(float(len(test[0])) / float(batch_size)))
    y_true = np.array([])
    y_pred = np.array([])
    y_score = np.array([])
    for index in range(n_batches):
        diagnosis_codes, seq_time_step, mask_mult, mask_final, \
        mask_code, lengths, labels = get_batch_data_single_label(test, options, batch_size, index)
        h = embd_model(diagnosis_codes, seq_time_step, mask_mult, mask_final,
                       mask_code, lengths)
        logit = discriminator_model(h)

        temp = nn.functional.softmax(logit).data.cpu().numpy()
        temp = temp[:, 1]
        prediction = torch.max(logit, 1)[1].view((len(labels),)).data.cpu().numpy()
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

    print(roc_auc, avg_precision, accuary, precision, f1, recall)

    return roc_auc, avg_precision, accuary, precision, f1, recall, \
           embd_model, discriminator_model, \
           best_parameters_embd_file, best_parameters_discriminator_file