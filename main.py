import argparse

import torch
import torchvision.models as models
from torch.utils.data import DataLoader

from PlainNet import PlainNet
from data.dataset import MyTrainDataSet, MyTestDataSet
from utils import (
    set_random_seed
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_epochs", type=int, default=30)  # 30
    parser.add_argument("--gan_epochs", type=int, default=10)  # 10
    parser.add_argument("--aed_meta_epochs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)  # 10
    parser.add_argument("--shot_num", type=int, default=2)
    parser.add_argument("--ex_ratio", type=int, default=50)

    parser.add_argument("--mix_noise", type=int, default=1)
    parser.add_argument("--noise_std", type=float, default=0.010)
    parser.add_argument("--sdas_transparency_range", type=float, default=[0.4, 1.0])
    parser.add_argument("--dtd_transparency_range", type=float, default=[0.2, 1.0])
    parser.add_argument("--patchsize", type=int, default=3)
    parser.add_argument("--patchstride", type=int, default=1)
    parser.add_argument("--train_backbone", type=bool, default=False)
    parser.add_argument("--intput_shape", type=int, default=[3, 224, 224])
    parser.add_argument("--adapter_lr", type=float, default=1e-3)
    parser.add_argument("--discriminator_lr", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--channel_num_reduced", type=int, default=1024)
    parser.add_argument("--embed_dimension", type=int, default=1536)
    parser.add_argument('--adapter_layer_number', type=int, default=1)
    parser.add_argument('--adapter_layer_type', type=int, default=0)
    parser.add_argument('--discriminator_layer', type=int, default=2)
    parser.add_argument("--discriminator_hidden", type=int, default=1024)
    parser.add_argument("--resize", default=256, type=int)
    parser.add_argument("--imagesize", default=224, type=int)
    parser.add_argument("--perlin_noise_threshold", default=0.5, type=float)
    parser.add_argument("--anomaly_type", type=str, default='sdas')
    parser.add_argument("--weight", type=float,
                        default=[[0.35, 0.64, 0.01], [0.2, 0.4, 0.4], [0.33, 0.33, 0.33], [0.3, 0.3, 0.4]])
    parser.add_argument("--seed", type=int, default=666666)
    parser.add_argument("--work_nums", type=int, default=8)
    parser.add_argument("--layers_extract", type=str, default=['layer2', 'layer3'])
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    extractor = {
        'layers_extract': args.layers_extract,
        'backbone': models.wide_resnet50_2(weights='DEFAULT', progress=True)
    }

    mvtec_class = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                   'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    mpdd_class = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    visa_class = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
                  'pcb3', 'pcb4', 'pipe_fryum']
    btad_class = ['01', '02', '03']
    dataset_map = {'MVTec-AD': mvtec_class,
                    #'BTAD': btad_class,
                   #'MPDD': mpdd_class,
                   #'VisA': visa_class,
                   }

    for dataset_key, sub_dataset_key in dataset_map.items():
        print(f'----------------------------------------------------------------------{dataset_key}----------------------------------------------------------------------')
        for type_idx, one_class in enumerate(sub_dataset_key):  # i, 'bottle'
            print(f'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<{one_class}>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            train_dataset = MyTrainDataSet(one_class, dataset_key, args.shot_num, args.ex_ratio)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.work_nums, shuffle=True)
            test_dataset = MyTestDataSet(one_class, dataset_key)
            test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=args.work_nums, shuffle=True)
            plainnet = PlainNet(extractor, device, train_dataloader, test_dataloader, one_class, dataset_key, args)
            plainnet.train_self()


if __name__ == '__main__':
    main()
