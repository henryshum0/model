import torch
from pathlib import Path
def print_pth_info(pth_path: str):
    """
    Prints information about the model saved in a .pth file.
    
    Args:
        pth_path (str): Path to the .pth file.
    """
    pth_path = Path(pth_path)
    if not pth_path.exists():
        raise FileNotFoundError(f"Path {pth_path} does not exist.")
    
    checkpoint = torch.load(pth_path, map_location='cpu', weights_only=False)
    cfg = checkpoint.get('config', None)
    if cfg:
        print(f"Model ID: {cfg.model_id}")
        print(f"Model type: {cfg.model}")
        print(f"Epoch trained: {cfg.epoch_trained}")
        print(f"Global step: {cfg.global_step}")
        print(f"Learning rate: {cfg.lr}")
        print(f"Loss functions: {cfg.loss}")
        print(f"Optimizer: {cfg.optimizer}")
        print(f"Scheduler: {cfg.scheduler}")
        print(f"Checkpoint save directory: {cfg.checkpoint_save_dir}")
        print(f"Logger directory: {cfg.log_dir_train}")
    else:
        print("No configuration found in the checkpoint.")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python print_pth_info.py <path_to_pth_file>")
    else:
        pth_file = sys.argv[1]
        print_pth_info(pth_file)