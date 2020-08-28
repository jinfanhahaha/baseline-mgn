from fastreid.modeling.meta_arch.mgn import MGN
from config import get_cfg
from fastreid.engine import default_argument_parser, default_setup
# from fastreid.solver import optim
import torch.optim as optim
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import random


seed = 2020
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# data loader
class ImageDataset(data.Dataset):
    def __init__(self, path, transform=None):
        with open(path, 'r') as f:
            self.annotations = f.read().split("\n")
            self.annotations = self.annotations[:-1]
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.annotations[index].split(':')
        image = Image.open('./data/train/images/' + img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        return image, int(label)

    def __len__(self):
        return len(self.annotations)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def build_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "heads" in key:
            lr *= cfg.SOLVER.HEADS_LR_FACTOR
        if "bias" in key:
            lr *= cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"name": key, "params": [value], "lr": lr, "weight_decay": weight_decay, "freeze": False}]

    solver_opt = cfg.SOLVER.OPT
    if hasattr(optim, solver_opt):
        if solver_opt == "SGD":
            opt_fns = getattr(optim, solver_opt)(params, momentum=cfg.SOLVER.MOMENTUM)
        else:
            opt_fns = getattr(optim, solver_opt)(params)
    else:
        raise NameError("optimizer {} not support".format(cfg.SOLVER.OPT))
    return opt_fns


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
trainset = ImageDataset('./data/train/label.txt', transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

# get cfg
args = default_argument_parser().parse_args()
cfg = get_cfg()
# create MGN model
model = MGN(cfg)
print(model)
# default Adam lr=1.5e-4 momentum=0.9 weight_decay=0.0005
# optimizer = build_optimizer(cfg=cfg, model=model)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model.to(device)
EPOCHS = 41
save_epoch = [10, 20, 30, 40]

print(model.state_dict())

for epoch in range(EPOCHS):
    running_loss = 0
    train_correct = 0
    train_total = 0
    for i, data in enumerate(trainloader):
        images, labels = data[0].to(device), data[1].to(device, dtype=torch.int64)
        optimizer.zero_grad()
        input_data = {'images': images, 'targets': labels}
        outputs = model.forward(input_data)
        # print(len(outputs))
        # print(outputs[0].shape)
        # print(outputs[0])
        losses = model.losses(outputs, labels)
        losses.backward()
        optimizer.step()

        running_loss += losses.item()
        if i % 200 == 199:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/200))
            running_loss = 0.0
    # train_accuracy = 100 * train_correct / train_total
    # print('epoch %d train dataset accuracy %.4f' % (epoch, train_accuracy))
    if epoch in save_epoch:
        PATH = './checkpoints/model2_epoch%d.pth' % epoch
        torch.save(model.state_dict(), PATH)
