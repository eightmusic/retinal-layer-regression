import argparse
from pathlib import Path
import torch

from models import get_model
from datasets import build_dataset
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split
from utils.utils import train_model
from utils.loss import loss_compu


def get_args_parser():
    parser = argparse.ArgumentParser("OCT layer segmentation")
    parser.add_argument('--batch_size', default=8,type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='init learning rate (absolute lr)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--save_path', type=str, default='', help='save pretrained path')
    parser.add_argument('--num_layer', type=int, default=8, help='number of layer')
    args = parser.parse_args()
    return args
def main(args):
    # dataset_train = build_dataset(is_train=True, args=args)
    # dataset_val = build_dataset(is_train=False, args=args)

    ds_train = build_dataset(is_train=True, args=args)
    ds_valid = build_dataset(is_train=False, args=args)

    train_loader = DataLoader(ds_train, batch_size=args.batch_size)
    val_loader = DataLoader(ds_valid, batch_size=args.batch_size)


    model=get_model()
    optimizer = torch.optim.AdamW(model.param_groups, lr=args.lr, weight_decay=args.weight_decay)

    train_model(model, args.epochs, train_loader,val_loader, PATH=args.save_path, n_class=args.num_layer, optimizer=optimizer)
    # train_model(relaynet_model, 300, train_loader, val_loader, PATH=T_path, n_class=8, l=0.0008)

# print args.integer

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)