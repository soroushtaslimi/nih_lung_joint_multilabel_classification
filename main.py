# -*- coding:utf-8 -*-
import os
import torch
import torchvision
import torchvision.transforms as transforms
from data.cifar import CIFAR10, CIFAR100
from data.mnist import MNIST
import argparse, sys
import datetime
from algorithm.jocor import JoCoR
# from data.XRay import XRaysTestDataset

from data.custom_image_folder import MyImageFolder

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='results')
parser.add_argument('--noise_rate', type=float, help='corruption rate, should be less than 1', default=0.0)
parser.add_argument('--forget_rate', type=float, help='forget rate', default=None)
parser.add_argument('--noise_type', type=str, help='[pairflip, symmetric, asymmetric] or clean', default='clean')
parser.add_argument('--num_gradual', type=int, default=10,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
parser.add_argument('--dataset', type=str, help='mnist, cifar10, cifar100, nih', default='nih')
parser.add_argument('--n_epoch', type=int, default=1000)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--num_iter_per_epoch', type=int, default=400)
parser.add_argument('--epoch_decay_start', type=int, default=80)
# parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--multi_gpu', type=str, help='when using multiple gpus', default="True")
parser.add_argument('--co_lambda', type=float, default=0.9)
parser.add_argument('--adjust_lr', type=int, default=1)
parser.add_argument('--model_type', type=str, help='[mlp,cnn]', default='cnn')
parser.add_argument('--save_model', type=str, help='save model?', default="False")
parser.add_argument('--save_result', type=str, help='save result?', default="True")
parser.add_argument('--save_epoch', type=int, help='number of epochs between saving models', default=10)
# parser.add_argument('--class_name', type=str, help='class to be used for classification', default='Mass')

parser.add_argument('--nih_img_size', type=int, help='image resized size in nih_lung dataset', default=128)

args = parser.parse_args()

CHECKPOINT_PATH = '../Checkpoints/'

# Seed
torch.manual_seed(args.seed)
# if args.gpu is not None:
if torch.cuda.is_available():
    # device = torch.device('cuda:{}'.format(args.gpu))
    device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)

# Hyper Parameters
batch_size = 128
learning_rate = args.lr

# load dataset
if args.dataset == 'mnist':
    input_channel = 1
    num_classes = 10
    init_epoch = 0
    filter_outlier = True
    args.epoch_decay_start = 80
    args.model_type = "mlp"
    # args.n_epoch = 200
    train_dataset = MNIST(root='./data/',
                          download=True,
                          train=True,
                          transform=transforms.ToTensor(),
                          noise_type=args.noise_type,
                          noise_rate=args.noise_rate
                          )

    test_dataset = MNIST(root='./data/',
                         download=True,
                         train=False,
                         transform=transforms.ToTensor(),
                         noise_type=args.noise_type,
                         noise_rate=args.noise_rate
                         )

if args.dataset == 'cifar10':
    input_channel = 3
    num_classes = 10
    init_epoch = 20
    args.epoch_decay_start = 80
    filter_outlier = True
    args.model_type = "cnn"
    # args.n_epoch = 200
    train_dataset = CIFAR10(root='./data/',
                            download=True,
                            train=True,
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                            )

    test_dataset = CIFAR10(root='./data/',
                           download=True,
                           train=False,
                           transform=transforms.ToTensor(),
                           noise_type=args.noise_type,
                           noise_rate=args.noise_rate
                           )

if args.dataset == 'cifar100':
    input_channel = 3
    num_classes = 100
    init_epoch = 5
    args.epoch_decay_start = 100
    # args.n_epoch = 200
    filter_outlier = False
    args.model_type = "cnn"


    train_dataset = CIFAR100(root='./data/',
                             download=True,
                             train=True,
                             transform=transforms.ToTensor(),
                             noise_type=args.noise_type,
                             noise_rate=args.noise_rate
                             )

    test_dataset = CIFAR100(root='./data/',
                            download=True,
                            train=False,
                            transform=transforms.ToTensor(),
                            noise_type=args.noise_type,
                            noise_rate=args.noise_rate
                            )

if args.dataset == 'nih':
    input_channel = 1
    num_classes = 14
    init_epoch = 0
    filter_outlier = True
    args.epoch_decay_start = 80
    args.model_type = "resnet"

    data_path = '/mnt/sda1/project/nih-preprocess/Dataset/'  # TODO isnert data path
    train_data_path = data_path + 'train2_cropped/'
    test_data_path = data_path + 'test2_cropped/'

    train_csv_path = '../processed_data_csv/train_without_nofinding.csv'
    test_csv_path = '../processed_data_csv/test_without_nofinding.csv'

    img_transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((args.nih_img_size, args.nih_img_size)),  
                                    # transforms.RandomAffine(degrees=20, translate = (0.05, 0.05)),
                                    transforms.ToTensor()])

    img_transform1 = transforms.Compose([transforms.Grayscale(),
                                        transforms.Resize((args.nih_img_size, args.nih_img_size)),  
                                        transforms.ToTensor()])


    train_dataset = MyImageFolder(
        root=train_data_path,
        csv_path=train_csv_path,
        transform=img_transform
    )

    test_dataset = MyImageFolder(
        root=test_data_path,
        csv_path=test_csv_path,
        transform=img_transform1
    )

if args.forget_rate is None:
    forget_rate = args.noise_rate
else:
    forget_rate = args.forget_rate



def save_models(model, epoch):
    checkpoint = {
        'epoch': epoch,
        'model1': model.model1.state_dict(),
        'model2': model.model2.state_dict(),
        'optimizer': model.optimizer.state_dict(),
        'scheduler': model.scheduler
    }
    torch.save(checkpoint, CHECKPOINT_PATH + 'resnet34' + 
                                            '_img' + str(args.nih_img_size) + 
                                            '_forget_rate' + str(args.forget_rate) +
                                            '_co_lambda' + str(args.co_lambda) +
                                            '_epoch' + str(epoch) + '.pth')


def main():
    # Data Loader (Input Pipeline)
    print('loading dataset...')
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               num_workers=args.num_workers,
                                               drop_last=True,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              num_workers=args.num_workers,
                                              drop_last=True,
                                              shuffle=False)
    
    print('len(train_loader.dataset):', len(train_loader.dataset))

    train_dataset_array = next(iter(train_loader))[0]
    print('train_dataset_array.shape:', train_dataset_array.shape)
    
    # Define models
    print('building model...')

    model = JoCoR(args, train_dataset, device, input_channel, num_classes, len(test_loader) * batch_size)

    epoch = 0
    train_acc1 = 0
    train_acc2 = 0

    # evaluate models with random weights
    test_acc1, test_acc2, test_auc1, test_auc2 = model.evaluate(test_loader)

    """print(
        'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %%, Model2 %.4f %%' % (
            epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))"""
    print(
        'Epoch [%d/%d] Test Accuracy on the %s test images:' % (
            epoch + 1, args.n_epoch, len(test_dataset)))
    print('Model1 AUCs:', test_auc1)
    print('Model1 mean AUC:', sum(test_auc1)/len(test_auc1))
    print('Model2 AUCs:', test_auc2)
    print('Model2 mean AUC:', sum(test_auc2)/len(test_auc2))
    """
    print(
        'Epoch [%d/%d] Test AUC on the %s test images: Model1 %.4f, Model2 %.4f ' % (
            epoch + 1, args.n_epoch, len(test_dataset), test_auc1, test_auc2))
    """
    """
    aucs = model.roc_auc(test_dataset, args)
    print('AUCs:', aucs)
    print('mean AUC:', sum(aucs)/len(aucs))
    """

    acc_list = []
    # training
    for epoch in range(1, args.n_epoch):
        # train models
        train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list = model.train(train_loader, epoch)

        # evaluate models
        test_acc1, test_acc2, test_auc1, test_auc2 = model.evaluate(test_loader)

        """
        # save results
        if pure_ratio_1_list is None or len(pure_ratio_1_list) == 0:
            print(
                'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
        else:
            # save results
            mean_pure_ratio1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
            mean_pure_ratio2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)
            print(
                'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1,
                    mean_pure_ratio2))
        """
        print(
            'Epoch [%d/%d] Test Accuracy on the %s test images:' % (
                epoch + 1, args.n_epoch, len(test_dataset)))
        print('Model1 AUCs:', test_auc1)
        print('Model1 mean AUC:', sum(test_auc1)/len(test_auc1))
        print('Model2 AUCs:', test_auc2)
        print('Model2 mean AUC:', sum(test_auc2)/len(test_auc2))
        """
        print(
                'Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %%, Model2 %.4f %%' % (
                    epoch + 1, args.n_epoch, len(test_dataset), test_acc1, test_acc2))
        print(
        'Epoch [%d/%d] Test AUC on the %s test images: Model1 %.4f, Model2 %.4f ' % (
            epoch + 1, args.n_epoch, len(test_dataset), test_auc1, test_auc2))
        """

        if epoch % args.save_epoch == 0:
            save_models(model, epoch)

        # if epoch >= 190:
        #     acc_list.extend([test_acc1, test_acc2])

    # avg_acc = sum(acc_list)/len(acc_list)
    # print(len(acc_list))
    # print("the average acc in last 10 epochs: {}".format(str(avg_acc)))


if __name__ == '__main__':
    main()
