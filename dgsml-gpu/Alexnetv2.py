import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from utils import *

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'http://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self,):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
        )      

    def forward(self, x, param = None):
        if param == None:
            x1 = self.features(x)
            x2 = x1.view(x1.size(0), 256 * 6 * 6)
            out = self.classifier(x2)
        else: 
            x1 = F.conv2d(x, param['features.0.weight'], param['features.0.bias'], stride=4, padding=2)
            x2 = F.relu(x1)
            x3 = F.max_pool2d(x2, kernel_size=3, stride=2)
            x4 = F.conv2d(x3, param['features.3.weight'], param['features.3.bias'], padding=2)
            x5 = F.relu(x4)
            x6 = F.max_pool2d(x5, kernel_size=3, stride=2)            
            x7 = F.conv2d(x6, param['features.6.weight'], param['features.6.bias'], padding=1)
            x8 = F.relu(x7)
            x9 = F.conv2d(x8, param['features.8.weight'], param['features.8.bias'], padding=1)
            x10 = F.relu(x9)
            x11 = F.conv2d(x10, param['features.10.weight'], param['features.10.bias'], padding=1)
            x12 = F.relu(x11)            
            x13 = F.max_pool2d(x12, kernel_size=3, stride=2)
            x14 = x13.view(x13.size(0), 256 * 6 * 6)
            x15 = F.dropout(x14)
            x16 = F.linear(x15, param['classifier.1.weight'], param['classifier.1.bias'])
            x17 = F.relu(x16)
            x18 = F.dropout(x17)
            out = F.linear(x18, param['classifier.4.weight'], param['classifier.4.bias'])
        return out


def alexnet(pretrained=False, **kwargs):
    
    model = AlexNet(**kwargs)

    if pretrained:
        #import pdb 
        #pdb.set_trace()
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'], progress=True)
        #predicted_dict = torch.load('./alexnet-owt-4df8aa71.pth')


        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v.data for k, v in pretrained_dict.items() if
                           k in model_dict and v.shape == model_dict[k].size()}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model

class NewModel(nn.Module):
    def __init__(self, args):
        super(NewModel, self).__init__()

        self.model = alexnet(pretrained=True)
        self.fc1 = nn.Linear(4096, args.num_classes)


    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.model.classifier(x)
        x2 = nn.functional.relu(self.fc1(x))
        x3 = nn.functional.softmax(x2, dim=1)

        return x, x2, x3
    
def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()    
            
def update(model, param):
    def param_fn(model, name=None):
        if len(model._modules)!=0:
            for(key,val) in model._modules.items():
                if name is None:
                    param_fn(val, name=str(key))
                else:
                    param_fn(val, name=str(name+'.'+key))
        else:
            for (key,val) in model._parameters.items():
                if not isinstance(val, torch.Tensor):
                    continue
                model._parameters[key] = param[str(name + '.' + key)]

    param_fn(model)
    return model

def classifier(class_num):
    model = nn.Sequential(
        #nn.ReLU(),
        nn.Linear(4096, class_num),
    )

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)
    return model


class Classifier_homo(nn.Module):
    def __init__(self, num_classes):
        super(Classifier_homo, self).__init__()
        self.CLS = torch.nn.Sequential(
            nn.ReLU(),
            nn.Linear(4096, num_classes))
    def forward(self, x, param = None):
        if param == None: 
            out = self.CLS(x)
        else: 
            x1 = F.relu(x)
            out = F.linear(x1, param['CLS.1.weight'], param['CLS.1.bias'])
        return out 
    
    
def cloned_state_dict(Model):
    cloned_state_dict = {
        key: val.clone()
        for key, val in Model.state_dict().items()
    }
    return cloned_state_dict     
