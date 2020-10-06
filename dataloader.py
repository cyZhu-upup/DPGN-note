from __future__ import print_function
from PIL import Image as pil_image
import random
import os
import numpy as np
import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchnet as tnt


class MiniImagenet(data.Dataset):
    """
    preprocess the MiniImageNet dataset
    """
    def __init__(self, root, partition='train', category='mini'):
        super(MiniImagenet, self).__init__()
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]
        # set normalizer
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        # set transformer
        """ColorJitter()随机改变图像的亮度，对比度，饱和度"""
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ColorJitter(brightness=.1,
                                                                        contrast=.1,
                                                                        saturation=.1,  
                                                                        hue=.1),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])            
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([
                lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        print('Loading {} ImageNet dataset -phase {}'.format(category, partition))
        # load data
        dataset_path = os.path.join(self.root, 'mini_imagenet', 'mini_imagenet_%s.pickle' % self.partition)
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)#type=dict,每一类的大小是(600,84,84,3)

        self.full_class_list = list(data.keys())#类的index，训练集共64个
        self.data, self.labels = data2datalabel(data)
        """data是（12000，84，84，3），label是(12000,)"""
        self.label2ind = buildLabelIndex(self.labels)

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        image_data=pil_image.fromarray(np.uint8(img))#np.uint8(img)还是(84,84,3)，到image_data就(84,84)了——>pil.image的格式:mode=RGB，显示的size是长宽，不显示通道
        """
        并且原来的H*W在pil_image里是W*H，所以下面一行要变换，不过mini_imagenet的长宽一样
        如果pil_image直接转torch.Tensor，图片的格式会从list时的H*W*C转为C*H*W
        """
        image_data=image_data.resize((self.data_size[2], self.data_size[1]))
        return image_data, label

    def __len__(self):
        return len(self.data)


class DataLoader:
    """
    The dataloader of DPGN model for MiniImagenet dataset
    """
    def __init__(self, dataset, num_tasks, num_ways, num_shots, num_queries, epoch_size, num_workers=4, batch_size=1):

        self.dataset = dataset
        self.num_tasks = num_tasks  #train_batch_size=25;test_batch_size=10
        self.num_ways = num_ways    #5
        self.num_shots = num_shots  #1
        self.num_queries = num_queries  #1
        self.num_workers = num_workers
        self.batch_size = batch_size    
        self.epoch_size = epoch_size    #train:100000
        self.data_size = dataset.data_size  #[3,84,84]
        self.full_class_list = dataset.full_class_list#test的话是[76, 83, 69, 70, 78, 80, 81, 79, 67, 71, 72, 82, 77, 75, 65, 73, 64, 66, 68, 74]
        self.label2ind = dataset.label2ind#label2inds[class_index]=(600,)
        self.transform = dataset.transform
        self.phase = dataset.partition
        self.is_eval_mode = (self.phase == 'test') or (self.phase == 'val')

    def get_task_batch(self):
        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(self.num_ways * self.num_shots):#5
            data = np.zeros(shape=[self.num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[self.num_tasks],
                             dtype='float32')
            support_data.append(data)#[5,25,3,84,84]
            support_label.append(label)
        for _ in range(self.num_ways * self.num_queries):#5
            data = np.zeros(shape=[self.num_tasks] + self.data_size,
                            dtype='float32')#[num_task,3,84,84]
            label = np.zeros(shape=[self.num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)
        # for each task
        for t_idx in range(self.num_tasks):
            task_class_list = random.sample(self.full_class_list, self.num_ways) #train的长度为64，64个类里面选5个,test为5,输出->（举例）[0,1,2,3,4]
            # for each sampled class in task
            for c_idx in range(self.num_ways):#5
                """一次循环里，data_idx是在某一类，共600个样本中，抽取6个"""
                data_idx = random.sample(self.label2ind[task_class_list[c_idx]], self.num_shots + self.num_queries)#在随机抽中的这5类中分别随机抽取2个，一个shot,一个query数据,随便试了次[5562, 5966]
                """把这2个样本对应的图像放入class_data_list中，格式为list(pil_image)"""
                """对于dataset来说,dataset[a][b]，a为第a+1个样本，b=0时为img，b=1为label(__getitem__设定的img和label）"""
                class_data_list = [self.dataset[img_idx][0] for img_idx in data_idx]#这里的是对应的pil_image
                for i_idx in range(self.num_shots):#1
                    # set data
                    """"c_idx=0时，support_data[0+0*1][0] = class_data_list[0]"""
                    """第一个变量是控制的类，共5个，第二个是task"""
                    support_data[i_idx + c_idx * self.num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * self.num_shots][t_idx] = c_idx
                # load sample for query set
                for i_idx in range(self.num_queries):
                    """这个和上面的一样，存入的数据是class_data_list的另一个"""
                    query_data[i_idx + c_idx * self.num_queries][t_idx] = \
                        self.transform(class_data_list[self.num_shots + i_idx])#class_data_list[1]
                    query_label[i_idx + c_idx * self.num_queries][t_idx] = c_idx
        """以support_data为例，shape由[5,25,3,84,84]--->[25,5,3,84,84],,query_data一样"""
        support_data = torch.stack([torch.from_numpy(data).float() for data in support_data], 1)#按照第2维拼接，num_task=25,data是[25,3,84,84]
        support_label = torch.stack([torch.from_numpy(label).float() for label in support_label], 1)#[25,5]
        query_data = torch.stack([torch.from_numpy(data).float() for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float() for label in query_label], 1)
        return support_data, support_label, query_data, query_label

    def get_iterator(self, epoch=0):
        rand_seed = epoch
        random.seed(rand_seed)
        np.random.seed(rand_seed)

        def load_function(iter_idx):
            support_data, support_label, query_data, query_label = self.get_task_batch()
            return support_data, support_label, query_data, query_label
        """这下面不是很懂"""
        tnt_dataset = tnt.dataset.ListDataset(
            elem_list=range(self.epoch_size), load=load_function)
        data_loader = tnt_dataset.parallel(
            batch_size=self.batch_size,
            num_workers=(1 if self.is_eval_mode else self.num_workers),
            shuffle=(False if self.is_eval_mode else True))
        return data_loader

    def __call__(self, epoch=0):
        return self.get_iterator(epoch)

    def __len__(self):
        return self.epoch_size // self.batch_size

"""
pickle文件转换为两个list,分别存储图像数据和标签

input:
    ori_data-dict
output:
    data:list->[class*num(20*600=12000),84,84,3]    
    label:list->[class*num(12000),]
"""
def data2datalabel(ori_data):
    data = []
    label = []
    for c_idx in ori_data:
        for i_idx in range(len(ori_data[c_idx])):
            data.append(ori_data[c_idx][i_idx])
            label.append(c_idx)
    return data, label

"""
给12000维的label添加引索
结果是label2inds[class_index]=(600,)
"""
def buildLabelIndex(labels):
    label2inds = {}
    for idx, label in enumerate(labels):
        if label not in label2inds:
            label2inds[label] = []
        label2inds[label].append(idx)
    return label2inds


"""--------------------------------------"""
"""--------------------------------------"""
"""--------------------------------------"""

class TieredImagenet(data.Dataset):
    def __init__(self, root, partition='train', category='tiered'):
        super(TieredImagenet, self).__init__()

        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]

        # set normalizer
        mean_pix = [x/255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x/255.0 for x in [70.68188272, 68.27635443,  72.54505529]]

        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        print('Loading {} ImageNet dataset -phase {}'.format(category, partition))
        if category == 'tiered':
            dataset_path = os.path.join(self.root, 'tiered-imagenet', '%s_images.npz' % self.partition)
            label_path = os.path.join(self.root, 'tiered-imagenet', '%s_labels.pkl' % self.partition)
            with open(dataset_path, 'rb') as handle:
                self.data = np.load(handle)['images']
            with open(label_path, 'rb') as handle:
                label_ = pickle.load(handle)
                self.labels = label_['labels']
                self.label2ind = buildLabelIndex(self.labels)
            self.full_class_list = sorted(self.label2ind.keys())
        else:
            print('No such category dataset')

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        image_data = pil_image.fromarray(img)
        return image_data, label

    def __len__(self):
        return len(self.data)


class Cifar(data.Dataset):
    """
    preprocess the MiniImageNet dataset
    """
    def __init__(self, root, partition='train', category='cifar'):
        super(Cifar, self).__init__()
        self.root = root
        self.partition = partition
        self.data_size = [3, 32, 32]
        # set normalizer
        mean_pix = [x/255.0  for x in [129.37731888, 124.10583864, 112.47758569]]
        std_pix = [x/255.0  for x in [68.20947949, 65.43124043, 70.45866994]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(32, padding=2),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        print('Loading {} dataset -phase {}'.format(category, partition))
        # load data
        if category == 'cifar':
            dataset_path = os.path.join(self.root, 'cifar-fs', 'cifar_fs_%s.pickle' % self.partition)
            with open(dataset_path, 'rb') as handle:
                u = pickle._Unpickler(handle)
                u.encoding = 'latin1'
                data = u.load()
            self.data = data['data']
            self.labels = data['labels']
            self.label2ind = buildLabelIndex(self.labels)
            """标签引索"""
            self.full_class_list = sorted(self.label2ind.keys())
        else:
            print('No such category dataset')

    def __getitem__(self, index):
        img, label = self.data[index], self.labels[index]
        image_data = pil_image.fromarray(img)
        return image_data, label

    def __len__(self):
        return len(self.data)


class CUB200(data.Dataset):
    def __init__(self, root, partition='train', category='cub'):
        super(CUB200, self).__init__()
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]
        # set normalizer
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)
        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.Resize(84, interpolation = pil_image.BICUBIC),
                                                 transforms.RandomCrop(84, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([transforms.Resize(84, interpolation = pil_image.BICUBIC),
                                                 transforms.CenterCrop(84),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        print('Loading {} dataset -phase {}'.format(category, partition))
        if category == 'cub':
            IMAGE_PATH = os.path.join(self.root, 'cub-200-2011', 'images')
            txt_path = os.path.join(self.root, 'cub-200-2011/split', '%s.csv' % self.partition)
            lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
            data = []
            label = []
            lb = -1
            self.wnids = []
            for l in lines:
                context = l.split(',')
                name = context[0]
                wnid = context[1]
                path = os.path.join(IMAGE_PATH, wnid, name)
                if wnid not in self.wnids:
                    self.wnids.append(wnid)
                    lb += 1
                data.append(path)
                label.append(lb)
            self.data = data
            self.labels = label
            self.full_class_list = list(np.unique(np.array(label)))
            self.label2ind = buildLabelIndex(self.labels)
        else:
            print('No such category dataset')

    def __getitem__(self, index):
        path, label = self.data[index], self.labels[index]
        image_data = pil_image.open(path).convert('RGB')
        return image_data, label

    def __len__(self):
        return len(self.data)



if __name__ == '__main__':

    dataset_train = MiniImagenet(root='../dataset', partition='train')
    epoch_size = len(dataset_train)
    dloader_train = DataLoader(dataset_train)
    bnumber = len(dloader_train)
    for epoch in range(0, 3):
        for idx, batch in enumerate(dloader_train(epoch)):
            print("epoch: ", epoch, "iter: ", idx)








