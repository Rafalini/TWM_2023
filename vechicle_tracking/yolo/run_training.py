import torch
from utils import train


def main():
    # parser = argparse.ArgumentParser(description='PyTorch YOLO')
    #
    # parser.add_argument('--use_cuda', type=bool, default=True,
    #                     help='use cuda or not')
    # parser.add_argument('--epochs', type=int, default=10,
    #                     help='Epochs')
    # parser.add_argument('--batch_size', type=int, default=1,
    #                     help='Batch size')
    # parser.add_argument('--lr', type=float, default=1e-3,
    #                     help='Learning rate')
    # parser.add_argument('--seed', type=int, default=1234,
    #                     help='Random seed')

    args = {
        'use_cuda': False,
        'epochs': 100,
        'batch_size': 6,
        'lr': 1e-3,
        'seed': 1234
    }

    # args = parser.parse_args()
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.benchmark = args['use_cuda']
    train.train(args)




if __name__ == '__main__':
    main()
