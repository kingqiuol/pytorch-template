import numpy
import torchvision
import torchvision.transforms as transforms

from datasets.cifar100_dataset import CIFAR100Train


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0]/255 for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1]/255 for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2]/255 for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

def compute_mean_std_v2(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


if __name__=="__main__":
    from  conf.global_settings import *
    cifar100_training = CIFAR100Train(path=PATH,
                                      transform=None)
    mean, std=compute_mean_std_v2(cifar100_training)
    print(f"mean:{mean},std:{std}")
