import os
import pickle
import torch
import time
import random
import shutil
from baselines.conan import *
import models.units as units
from models.transformer import Get_embd, Generator, Discriminator
# from models.Dipole import Get_embd, Generator, Discriminator
# from models.RNN import Get_embd, Generator, Discriminator
from baselines.meta_model import MLP, convert_meta_data, train_embd_dis_meta, upsample


def train_model(training_file='training_file',
                validation_file='validation_file',
                testing_file='testing_file',
                n_diagnosis_codes=10000,
                n_labels=2,
                output_file='output_file',
                batch_size=100,
                dropout_rate=0.5,
                L2_reg=0.001,
                pretrain_epoch=1000,
                n_epoch_g=100,
                n_epoch_together=20,
                log_eps=1e-8,
                visit_size=512,
                hidden_size=256,
                device='cpu',
                model_name='',
                disease = 'hf',
                code2id = None,
                running_data='',
                gamma=0.5,
                model_file = None,
                layer=1,
                maxcode=None,
                maxlen=None,
                meta_lr=1e-5,
                normal_lr=1e-4,
                lr=1e-1,
                batch_size_ori=None, batch_size_meta=None, batch_size_gan=None,
                gan_meta_epoch=None, meta_interval=None, output_model_path=None):
    options = locals().copy()

    print('***********************')
    print('pretrain_epoch: {}, n_epoch_g: {}'.format(pretrain_epoch, n_epoch_g))
    print('normal_lr: {}, meta_lr: {}, lr: {}'.format(normal_lr, meta_lr, lr))
    print('batch_size_ori: {}, batch_size_meta: {}, batch_size_gan: {}'.format(batch_size_ori, batch_size_meta, batch_size_gan))

    print('***********************')

    print('building the model ...')
    with torch.backends.cudnn.flags(enabled=False):

        embd_model = Get_embd(options).to(device)
        generator_model = Generator(options).to(device)
        discriminator_model = Discriminator(options).to(device)
        print('constructing the optimizer ...')
        embd_discri_para = list(embd_model.parameters()) + list(discriminator_model.parameters())
        embd_discri_para_optimizer = torch.optim.SGD(embd_discri_para,
                                                     lr=options['normal_lr'],
                                                     weight_decay=options['L2_reg'])

        generator_para = list(generator_model.parameters())
        generator_para_optimizer = torch.optim.SGD(generator_para,
                                                   lr=options['normal_lr'],
                                                   weight_decay=options['L2_reg'])

        discriminator_para = list(discriminator_model.parameters())
        discriminator_para_optimizer = torch.optim.SGD(discriminator_para,
                                                       lr=options['normal_lr'],
                                                       weight_decay=options['L2_reg'])

        print('loading data ...')
        train, validate, test = units.load_data(training_file, validation_file, testing_file)
        train = upsample(train, upsample_num=1)
        validate_meta = convert_meta_data(validate, rate=1)
        n_batches = int(np.ceil(float(len(train[0])) / float(batch_size)))

        print('pretraining start')
        embd_model, generator_model, discriminator_model = pretrain(embd_model, generator_model, discriminator_model, train,
                                                                    validate, test, options, embd_discri_para_optimizer)

        print('training start')
        embd_model, generator_model, discriminator_model = c_train(embd_model, generator_model, discriminator_model, train,
                                                                   validate, test, options, discriminator_para_optimizer,
                                                                   generator_para_optimizer)

        torch.save(embd_model.state_dict(), output_model_path+'embd_model.temp')
        torch.save(generator_model.state_dict(), output_model_path+'generator_model.temp')
        torch.save(discriminator_model.state_dict(), output_model_path+'discriminator_model.temp')

        embd_model.load_state_dict(torch.load(output_model_path+'embd_model.temp'))
        generator_model.load_state_dict(torch.load(output_model_path+'generator_model.temp'))
        discriminator_model.load_state_dict(
            torch.load(output_model_path+'discriminator_model.temp'))

        print('Train together')
        # convert meta_data
        embd_discri_para_meta_optimizer = torch.optim.SGD(embd_discri_para,
                                                          lr=options['lr'], weight_decay=5e-4,
                                                          momentum=0.9, dampening=0,
                                                          nesterov=False)
        meta_net = MLP()
        meta_optimizer = torch.optim.SGD(meta_net.parameters(),
                        lr=options['meta_lr'], weight_decay=5e-4,
                        momentum=0.9, dampening=0,
                        nesterov=False)
        roc_auc, avg_precision, accuary, precision, f1, recall = \
            train_embd_dis_meta(embd_model, generator_model, discriminator_model, meta_net,
                            train, validate_meta, validate, test, options,
                            embd_discri_para_meta_optimizer, meta_optimizer)

        return roc_auc, avg_precision, accuary, precision, f1, recall


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("available device: {}".format(device))
    # parameters
    batch_size = 128
    batch_size_ori = 512
    batch_size_meta = 100
    batch_size_gan = 32
    maxcode = 50
    maxlen = 100
    dropout_rate = 0.5
    L2_reg = 1e-3
    log_eps = 1e-8
    pretrain_epoch = 20
    n_epoch_g_list = [1000]
    n_epoch_together = 1000
    n_labels = 2     # binary classification
    visit_size = 64    # size of input embedding
    hidden_size = 128    # size of hidden layer
    gamma = 0.0     # setting for Focal Loss, when it's zero, it's equal to standard cross loss
    layer = 1    # layer of Transformer
    meta_lr = 1e-5
    normal_lr = 1e-4
    lr = 1e-2
    gan_meta_epoch = 1000
    meta_interval = 1

    model_file = eval('Get_embd')
    disease_list = ['ipf', 'rks', 'hes', 'mas']   # name of the sample data set, you can place you own data set by following the same setting
    print('dataset: {}'.format(disease_list[0]))
    model_choice = 'transformer'  # name of the proposed HiTANet in our paper
    model_save_file = model_choice + '_' + disease_list[0]

    roc_auc_list = []
    avg_precision_list = []
    accuary_list = []
    precision_list = []
    f1_list = []
    recall_list = []

    for n_epoch_g in n_epoch_g_list:
        for disease in disease_list:
            model_name = 'tran_%s_%s_L%d_wt_1e-4_focal%.2f' % (model_choice, disease, layer, gamma)
            print(model_name)
            log_file = 'results/' + model_name + '.txt'
            path = '../dataset/' + disease + '_dataset/'
            trianing_file = path + disease + '.train_sample'
            validation_file = path + disease + '.valid_sample'
            testing_file = path + disease + '.test_sample'

            dict_file = path + disease + '.record_code_dict_sample'
            code2id = pickle.load(open(dict_file, 'rb'))
            n_diagnosis_codes = len(pickle.load(open(dict_file, 'rb'))) + 1

            output_file_path = 'cache/' + model_choice + '_outputs/'
            if os.path.isdir(output_file_path):
                pass
            else:
                os.mkdir(output_file_path)
            results = []

            output_model_path = 'saved_models/' + disease + '_saved_models/'
            if os.path.isdir(output_model_path):
                pass
            else:
                os.mkdir(output_model_path)

            for seed in range(2022, 2021, -1):
                torch.manual_seed(seed)  # cpu
                torch.cuda.manual_seed(seed)  # gpu
                np.random.seed(seed)  # numpy
                random.seed(seed)  # random and transforms

                roc_auc, avg_precision, accuary, precision, f1, recall\
                    = train_model(trianing_file, validation_file, testing_file,
                                n_diagnosis_codes, n_labels, output_file_path, batch_size, dropout_rate,
                                L2_reg, pretrain_epoch, n_epoch_g, n_epoch_together,
                                log_eps, visit_size, hidden_size,
                                device, model_name, disease=disease, code2id=code2id,
                                gamma=gamma, layer=layer, model_file=model_file, maxcode=maxcode, maxlen=maxlen,
                                meta_lr=meta_lr, normal_lr=normal_lr, lr=lr,
                                batch_size_ori=batch_size_ori, batch_size_meta=batch_size_meta, batch_size_gan=batch_size_gan,
                                gan_meta_epoch=gan_meta_epoch, meta_interval=meta_interval, output_model_path=output_model_path)

            roc_auc_list.append(roc_auc)
            avg_precision_list.append(avg_precision)
            accuary_list.append(accuary)
            precision_list.append(precision)
            f1_list.append(f1)
            recall_list.append(recall)
            print("***************************************************")
            print("***************************************************")

    for n_epoch_g, a, b, c, d, e, f in zip(n_epoch_g_list,
                                           roc_auc_list, avg_precision_list, accuary_list,
                                           precision_list, f1_list, recall_list):
        print('n_epoch_g: {}, roc_auc: {}, avg_precision: {}, accuary: {}, '
              'precision: {}, f1: {}, recall: {}'.format(n_epoch_g, a, b, c, d, e, f))



