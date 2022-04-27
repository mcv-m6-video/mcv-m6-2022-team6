import sys, argparse
import pandas as pd
from siamese_net import train, get_data_loader


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_csv', type=str, default='gt/gt_car_patches_annotations.csv',
                        )
    parser.add_argument('--gt_patches', type=str, default='gt/gt_car_patches/',
                        )
    parser.add_argument('--save_model', type=str, default='model/',
                        )
    parser.add_argument('--epochs', type=int, default=10,
                        )
    parser.add_argument('--lr', type=float, default=1e-3,
                        )
    
    parser.add_argument('--embeddings', type=int, default=512,
                        )
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args()
    labels = pd.read_csv(args.gt_csv)

    train_data, train_loader = get_data_loader('train', labels, args.gt_patches)
    test_data, test_loader = get_data_loader('test', labels, args.gt_patches)

    train(train_data, test_data, args.save_model, args.epochs, args.lr, args.embeddings, args.batch_size)