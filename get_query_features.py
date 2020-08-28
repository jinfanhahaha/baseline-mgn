# import os
#
#
# PATH1 = '/raid/ours_data/Competition/REID_DATA/image_A/gallery'
# filenames = None
# for _, _, files in os.walk(PATH1):
#     filenames = files


from fastreid.modeling.meta_arch.mgn import MGN
from config import get_cfg
from fastreid.engine import test_default_argument_parser, default_setup
# from fastreid.solver import optim
import torch.optim as optim
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import json
import random


seed = 2020
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# data loader
class TestDataset(data.Dataset):
    def __init__(self, DIR, ID, transform=None):
        self.annotations = [DIR + "/" + f for f in ID]
        self.ID = ID
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.annotations[index]
        image = Image.open(img_path).convert('RGB')
        id = self.ID[index]

        if self.transform is not None:
            image = self.transform(image)
        return image, id

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


PATH1 = './data/image_A/query'
ID = None
for _, _, files in os.walk(PATH1):
    ID = files

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
testset = TestDataset(PATH1, ID, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)

args = test_default_argument_parser().parse_args()
cfg = get_cfg()
# create MGN model
model = MGN(cfg)

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
PATH = './checkpoints/model2_epoch40.pth'
model.load_state_dict(torch.load(PATH))
model.to(device)
model.eval()
print(model)

res_dict = {}
with torch.no_grad():
    for i, data in enumerate(testloader):
        images, id = data[0].to(device), data[1][0]
        input_data = {'images': images}
        features = model.get_features(input_data)
        # print(features.shape)
        features = features.squeeze()
        # print(features.shape)
        # print(np.count_nonzero(features.cpu().data.numpy()))
        res_dict[id] = features.cpu().data.numpy().tolist()
        # print(res_dict)
        if i % 100 == 99:
            print('doing {}'.format(i))

    with open("./features/query_features.json", 'w', encoding='utf-8') as f:
        f.write(json.dumps(res_dict))


