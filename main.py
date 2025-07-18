from train import train
from config import Config
from argparse import ArgumentParser
from pathlib import Path
import torch
def main():
    args = get_args()
    if args.action == 'fresh':
        cfg = Config()
        train(cfg, train_type='fresh')
    elif args.action == 'resume':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if args.load:
            checkpoint_path = Path(args.load)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            train(cfg=checkpoint['config'], train_type='resume', **checkpoint)
            


def get_args():
    parser = ArgumentParser(description="get user command to train or predict")
    parser.add_argument('--action', '-a', type=str, help='Specify what action: ["pred", "fresh", "load", "resume"]')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load a pth file')
    parser.add_argument('--pred_in', type=str, default=False, help='Directory to load images')
    parser.add_argument('--pred_out', type=str, default=False, help='Directory to save predictions')
    
    return parser.parse_args()
    
if __name__ == "__main__":
    main()