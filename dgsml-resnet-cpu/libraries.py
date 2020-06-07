import argparse
import os
import sys
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import torchvision.models as models
from collections import OrderedDict
import math
from scipy.stats import entropy
from itertools import combinations
import torch.optim as optim
import itertools
from collections import OrderedDict
sys.setrecursionlimit(1000000)
import warnings
from pycocotools.coco import COCO
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable