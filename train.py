import argparse
from pathlib import Path
import torch

from models import get_model
from datasets import build_dataset
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split
from utils.utils import train_model
import warnings
warnings.filterwarnings("ignore") # 忽略警告


def get_args_parser():
    parser = argparse.ArgumentParser("OCT layer segmentation")
    parser.add_argument('--batch_size', default=8,type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--lr', type=float, default=0.0008, metavar='LR', help='init learning rate (absolute lr)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--save_path', type=str, default='', help='save pretrained path')
    parser.add_argument('--num_layer', type=int, default=8, help='number of layer')
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', type=str, default='cuda', help='cuda,cpu')
    # args = parser.parse_args()
    # return args
    return parser
def main(args):
    # dataset_train = build_dataset(is_train=True, args=args)
    # dataset_val = build_dataset(is_train=False, args=args)

    ds_train = build_dataset(is_train=True)
    ds_valid = build_dataset(is_train=False)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size)
    val_loader = DataLoader(ds_valid, batch_size=args.batch_size)


    model = get_model(n_classes=args.num_layer)
    # print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.param_groups, lr=args.lr, weight_decay=args.weight_decay)
    model.optimizer=optimizer

    train_model(args,model, args.epochs, train_loader,val_loader, PATH=args.save_path, n_class=args.num_layer, optimizer=optimizer)

# print args.integer

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
