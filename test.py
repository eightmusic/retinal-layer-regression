from utils.utils import evaluation
from models import get_model
from datasets import build_dataset
import argparse
from pathlib import Path
import torch
from torch.utils.data import TensorDataset,Dataset,DataLoader,random_split


def get_args_parser():
    parser = argparse.ArgumentParser("OCT layer segmentation")
    parser.add_argument('--batch_size', default=8,type=int, help='batch size')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--pretrained_model', default='B:/eee/demo/pycharm/oct/weights/transformer/end/DED9.pkl', type=str, help='pretrained model path')
    args = parser.parse_args()
    return args

def main(args):

    ds_valid = build_dataset(is_train=False, args=args)
    val_loader = DataLoader(ds_valid, batch_size=args.batch_size)

    model = get_model().cuda()
    model.load_state_dict(torch.load(args.pretrained_model))

    evaluation(model, val_loader)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)