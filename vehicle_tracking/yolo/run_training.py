import torch
from utils import train


def main():
    args = {
        'use_cuda': False,
        'epochs': 100,
        'batch_size': 1,
        'lr': 1e-3,
        'seed': 1234
    }

    # args = parser.parse_args()
    torch.manual_seed(args['seed'])
    torch.backends.cudnn.benchmark = args['use_cuda']
    train.train(args)




if __name__ == '__main__':
    main()
