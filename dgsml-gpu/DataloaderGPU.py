from utils import *
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

transform_train = transforms.Compose([
    #transforms.RandomHorizontalFlip(p=0.75),
    #transforms.RandomCrop(224, padding=4),
    #transforms.CenterCrop(224),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


transform_test = transforms.Compose([
    #transforms.CenterCrop(224),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_domain_name(args):
    if args.dataset == 'PACS':
        return {'0': 'photo', '1': 'art_painting', '2': 'cartoon', '3': 'sketch'}
    else: 
        return {'0': 'Caltech', '1': 'Labelme', '2': 'Pascal', '3': 'Sun'}

def get_data_folder(args):
    if args.dataset == 'PACS':
        data_folder = args.filelist
        train_data = ['photo_train_kfold.txt',
                      'art_painting_train_kfold.txt',
                      'cartoon_train_kfold.txt',
                      'sketch_train_kfold.txt']

        val_data = ['photo_crossval_kfold.txt',
                    'art_painting_crossval_kfold.txt',
                    'cartoon_crossval_kfold.txt',
                    'sketch_crossval_kfold.txt']

        test_data = ['photo_test_kfold.txt',
                     'art_painting_test_kfold.txt',
                     'cartoon_test_kfold.txt',
                     'sketch_test_kfold.txt']
    else: 
         data_folder = args.filelist
         train_data = ['Caltech_train.txt',
                      'Labelme_train.txt',
                      'Pascal_train.txt',
                      'Sun_train.txt']

         val_data = ['Caltech_crossval.txt',
                    'Labelme_crossval.txt',
                    'Pascal_crossval.txt',
                    'Sun_crossval.txt']

         test_data = ['Caltech_test.txt',
                     'Labelme_test.txt',
                     'Pascal_test.txt',
                     'Sun_test.txt']
    return data_folder, train_data, val_data, test_data

class BatchImageGenerator:
    def __init__(self, args, stage, file_path, metatest, b_unfold_label):

        if stage not in ['train', 'val', 'test']:
            assert ValueError('invalid stage!')

        self.configuration(args, stage, file_path, metatest)
        self.load_data(args, b_unfold_label)

    def configuration(self, args, stage, file_path, metatest):
        if metatest == False:
            self.batch_size = args.batch_size
        if metatest == True:
            self.batch_size = args.batch_size
        self.current_index = -1
        self.current_index2 = -1
        self.file_path = file_path
        self.stage = stage
        self.shuffled = False
        self.rate = args.unlabeled_rate
        self.batch_size_un = self.batch_size
        if self.rate < 0.5:
            self.batch_size_un = 32
        if self.rate > 0.5: 
            self.batch_size_un = 64
        if self.rate == 0.5:
            self.batch_size_un = 32 

    def load_data(self, args, b_unfold_label):
        file_path = self.file_path
        images = []
        labels = []
        for ls in file_path:
            with open(ls,'r') as file_to_read:
                while True:
                    lines = file_to_read.readline()
                    if not lines:
                        break
                        pass
                    image, label = [i for i in lines.split()]
                    images.append(image)
                    if args.dataset == 'VLCS':
                        labels.append(int(label))
                    else: 
                        labels.append(int(label)-1)
                    pass
        if b_unfold_label:
            labels = unfold_label(labels=labels, classes=len(np.unique(labels)))
        self.images = np.array(images)
        self.labels = np.array(labels)
        self.file_num_train = len(self.labels)

        if self.stage is 'train':
            self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)
            self.images, self.unlab_img, self.labels, self.unlab_lab = train_test_split(
                self.images, self.labels, test_size=self.rate)
            self.file_num_train = len(self.labels)
            self.file_num_train_unlab = len(self.unlab_lab)

    def get_images_labels_batch(self, args):

        images = []
        labels = []
        unlab_img = []
        unlab_lab = []        
        for index in range(self.batch_size):
            self.current_index += 1
            # void over flow
            if self.current_index > self.file_num_train - 1:
                self.current_index %= self.file_num_train
                self.images, self.labels = shuffle_data(samples=self.images, labels=self.labels)
            img = Image.open(args.data_root+self.images[self.current_index])
            img = img.convert('RGB')
            img = transform_train(img)
            img = np.array(img)
            images.append(img)
            labels.append(self.labels[self.current_index])
            
        for index in range(self.batch_size_un):
            self.current_index2 += 1
            # void over flow
            if self.current_index2 > self.file_num_train_unlab - 1:
                self.current_index2 %= self.file_num_train_unlab
                self.unlab_img, self.unlab_lab = shuffle_data(samples=self.unlab_img, labels=self.unlab_lab)
            
            img2 = Image.open(args.data_root+self.unlab_img[self.current_index2])
            img2 = img2.convert('RGB')
            img2 = transform_train(img2)
            img2 = np.array(img2)
            unlab_img.append(img2)
            unlab_lab.append(self.unlab_lab[self.current_index2])            
            

        return np.array(images), np.array(labels), np.array(unlab_img), np.array(unlab_lab)

def get_image(args, images):
    images_data = []
    for img in images:
        img = Image.open(args.data_root+img)
        if img.mode == 'L':
            img = img.convert("RGB")
        img = transform_train(img)
        img = np.array(img)
        images_data.append(img)
    return np.array(images_data)

def get_dataloader(args, train_paths, val_paths, unseen_data_path):

    batImageGenTrains = []
    for train_path in train_paths:
        batImageGenTrain = BatchImageGenerator(args, file_path=train_path, stage='train', metatest=False, b_unfold_label=False)
        batImageGenTrains.append(batImageGenTrain)

    batImageGenTrains_metatest = []
    for train_path in train_paths:
        batImageGenTrain_metatest = BatchImageGenerator(args, file_path=train_path, stage='train', metatest=True, b_unfold_label=False)
        batImageGenTrains_metatest.append(batImageGenTrain_metatest)

    batImageGenVals = []
    for val_path in val_paths:
        batImageGenVal = BatchImageGenerator(args, file_path=val_path, stage='val', metatest=True, b_unfold_label=True)
        batImageGenVals.append(batImageGenVal)

    batImageGenTests = []
    for test_path in unseen_data_path:
        batImageGenTest = BatchImageGenerator(args, file_path=test_path, stage='test', metatest=True, b_unfold_label=True)
        batImageGenTests.append(batImageGenTest)            

    return batImageGenTrains, batImageGenTrains_metatest, batImageGenVals, batImageGenTests      
