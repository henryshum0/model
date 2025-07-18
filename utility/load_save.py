from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch

from pathlib import Path

from config import Config
from data.crack_dataset import CrackDataset
import loss
from models.unet import UNet
import trainers

def load_from_cfg(cfg: Config):
    '''
    Load training settings using the configuration file
    '''
    dataset_train, dataset_val = get_datasets(cfg)
    loader_train, loader_val = get_loaders(dataset_train, dataset_val, cfg.train_batch_size, cfg.val_batch_size)
    model = get_model(cfg.model, dataset_train.get_n_classes())
    model.to(cfg.device)
    logger = get_logger(cfg.log_dir_train)
    optimizer = get_optimizer(model=model, cfg=cfg)
    scheduler = get_scheduler(optimizer=optimizer, cfg=cfg)
    loss_criterions = get_loss_criterion(cfg=cfg)
    trainer = get_trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterions=loss_criterions,
    )
    
    checkpoint_save_dir = cfg.checkpoint_save_dir
    checkpoint_save_dir = Path(checkpoint_save_dir)  # ensure directory exists
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
    
    
    return {
        'dataset_train': dataset_train,
        'dataset_val': dataset_val,
        'loader_train': loader_train,
        'loader_val': loader_val,
        'model': model,
        'logger': logger,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'trainer': trainer,
    }

def load_model(cfg:Config):
    '''
    Load from a pre trained model and return the model as a new model with settings in the config file.
    '''
    checkpoint = torch.load(cfg.checkpoint_load_path, map_location=cfg.device)
    model, model_settings = get_model(
        cfg.model,
        cfg.n_classes
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(cfg.device)
    return model

def resume(cfg:Config, model_state_dict:dict, optimizer_state_dict:dict, scheduler_state_dict:dict):
    '''
    Resume training from a saved checkpoint, using the configuration
    saved in the checkpoint i.e. model, optimizer, scheduler, etc.
    '''
    if cfg is None:
        raise ValueError("No configuration found in the checkpoint.")
    
    dataset_train, dataset_val = get_datasets(cfg)
    loader_train, loader_val = get_loaders(dataset_train, dataset_val, cfg.train_batch_size, cfg.val_batch_size)
    model = get_model(
        cfg.model,
        cfg.n_classes
    )
    model.load_state_dict(model_state_dict)
    model.to(cfg.device)
    logger = get_logger(cfg.log_dir_train)
    
    optimizer = get_optimizer(model=model, cfg=cfg)
    optimizer.load_state_dict(state_dict=optimizer_state_dict)
    scheduler = get_scheduler(optimizer=optimizer, cfg=cfg)
    scheduler.load_state_dict(state_dict=scheduler_state_dict)
    loss_criterions = get_loss_criterion(cfg=cfg)
    trainer = get_trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterions=loss_criterions,
    )
    return {
        'dataset_train': dataset_train,
        'dataset_val': dataset_val,
        'loader_train': loader_train,
        'loader_val': loader_val,
        'model': model,
        'logger': logger,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'trainer': trainer,
    }
    
    
    
def save(cfg:Config, model_state_dict:dict, optimizer_state_dict:dict, scheduler_state_dict:dict,
         epoch, global_step):
    '''
    Save the training configuration and
    '''
    cfg.epoch_trained = epoch
    cfg.global_step = global_step
    checkpoint = {
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': optimizer_state_dict,
            'scheduler_state_dict': scheduler_state_dict,
            'config': cfg,
        }

    checkpoint_path = Path(cfg.checkpoint_save_dir) / f'epoch_{cfg.epoch_trained}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved at {cfg.checkpoint_save_dir}/epoch_{cfg.epoch_trained}.pth")
    
def get_datasets(cfg: Config):
    dataset_train = CrackDataset(
        img_path=cfg.img_tr_dir,
        mask_path=cfg.mask_tr_dir,
        transforms=cfg.transforms_tr
    )
    
    dataset_val = CrackDataset(
        img_path=cfg.img_val_dir,
        mask_path=cfg.mask_val_dir,
        transforms=cfg.transforms_val
    )

    assert dataset_train.get_n_classes() == dataset_val.get_n_classes(), \
        "Number of classes in training and validation datasets must match."
    
    return dataset_train, dataset_val

def get_loaders(dataset_tr, dataset_val, batchsize_tr, batchsize_val):
    
    dataloader_tr = DataLoader(
        dataset_tr,
        batch_size=batchsize_tr,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batchsize_val,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader_tr, dataloader_val
    
def get_model(choice_model, n_classes=1):
    model = None
    if choice_model == 'unet':
        print(f"using model UNet")
        model = UNet(n_channels=3, n_classes=n_classes, bilinear=False)
    elif choice_model == 'deepcrack':
        raise NotImplementedError("not supported yet")
    elif choice_model == 'attention_unet':
        raise NotImplementedError("not supported yet")
    elif choice_model == 'segformer':
        raise NotImplementedError("not supported yet")
    elif choice_model == 'hnet':
        raise NotImplementedError("not supported yet")
    else:
        raise TypeError(f"model {choice_model} does not exist")
    
    return model

def get_optimizer(model, cfg:Config):
    if cfg.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.adam_weight_decay)
    elif cfg.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.sgd_momentum, weight_decay=cfg.sgd_weight_decay)
    elif cfg.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=cfg.lr, momentum=cfg.rmsprop_momentum, weight_decay=cfg.rmsprop_weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} is not supported")
    
def get_scheduler(optimizer, cfg:Config):
    if cfg.scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.step_gamma)
    elif cfg.scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.consine_T_max)
    elif cfg.scheduler == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=cfg.plateau_mode, factor=cfg.plateau_factor, patience=cfg.plateau_patience)
    else:
        raise NotImplementedError(f"Scheduler {cfg.scheduler} is not supported")

def get_loss_criterion(cfg:Config):
    criterions = []
    if 'dice' in cfg.loss:
        criterions.append(loss.DiceLoss(reduction=cfg.reduction))
    if 'focal' in cfg.loss:
        criterions.append(loss.FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, beta=cfg.focal_beta, reduction=cfg.reduction))
    if 'bce' in cfg.loss:
        criterions.append(nn.BCEWithLogitsLoss(reduction=cfg.reduction, pos_weight=cfg.bcd_pos_weight))
    if cfg.loss == []:
        raise ValueError("At least one loss criterion must be specified")
    if len(criterions) == 0:
        raise ValueError("No valid loss criteria specified. Please check your configuration.")
    return criterions
    
def get_trainer(cfg:Config, model, optimizer, scheduler, criterions):
    trainer = None
    if cfg.model == 'unet':
        trainer = trainers.UNetTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterions=criterions
        )
    elif cfg.model == 'deepcrack':
        raise ModuleNotFoundError("not supported yet")
    elif cfg.model == 'attention_unet':
        raise ModuleNotFoundError("not supported yet")
    elif cfg.model == 'segformer':
        raise ModuleNotFoundError("not supported yet")
    elif cfg.model == 'hnet':
        raise ModuleNotFoundError("not supported yet")
    else:
        raise TypeError(f"model {cfg.model} does not exist")
    
    return trainer

def get_logger(log_dir):
    return SummaryWriter(log_dir=log_dir)

if __name__ == "__main__":
    settings = resume('test.pth')