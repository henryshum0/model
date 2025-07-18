from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch

from pathlib import Path

from config import Config as cfg
from data.crack_dataset import CrackDataset
from data.data_generation import Resize
import loss
from models.unet import UNet
import trainers

def load_from_cfg():
    '''
    Load training settings using the configuration file
    '''
    
    dataset_train, dataset_val = get_datasets()
    loader_train, loader_val = get_loaders(dataset_train, dataset_val, cfg.train_batch_size, cfg.val_batch_size)
    model, model_settings = get_model(cfg.model, dataset_train.get_n_classes())
    model.to(cfg.device)
    logger = get_logger(cfg.log_dir_train)
    optimizer, optimizer_settings = get_optimizer_settings(  
        cfg.optimizer, model,
        lr=cfg.lr, 
        adam_weight_decay=cfg.adam_weight_decay,
        sgd_momentum=cfg.sgd_momentum,
        sgd_weight_decay=cfg.sgd_weight_decay,
        rmsprop_weight_decay=cfg.rmsprop_weight_decay,
        rmsprop_momentum=cfg.rmsprop_momentum
    )
    scheduler, scheduler_settings = get_scheduler_settings(
        cfg.scheduler, optimizer,
        step_size=cfg.step_size,
        step_gamma=cfg.step_gamma,
        consine_T_max=cfg.consine_T_max,
        plateau_mode=cfg.plateau_mode,
        plateau_factor=cfg.plateau_factor,
        plateau_patience=cfg.plateau_patience
    )
    loss_criterions, loss_settings = get_loss_criterion(
        cfg.loss,
        reduction=cfg.reduction,
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        focal_beta=cfg.focal_beta,
        bcd_pos_weight=cfg.bcd_pos_weight
    )
    
    trainer = get_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterions=loss_criterions,
    )
    
    checkpoint_save_dir = cfg.checkpoint_save_dir
    checkpoint_save_dir = Path(checkpoint_save_dir)  # ensure directory exists
    checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
    
    training_settings = {
        'lr': cfg.lr,
        'epoch': cfg.epoch,
        'batchsize_tr': cfg.train_batch_size,
        'batchsize_val': cfg.val_batch_size,
    }
    
    return {
        'model_id':cfg.model_id,
        'dataset_train': dataset_train,
        'dataset_val': dataset_val,
        'loader_train': loader_train,
        'loader_val': loader_val,
        'model': model,
        'model_settings': model_settings,
        'logger': logger,
        'logger_dir': cfg.log_dir_train,
        'checkpoint_save_dir': cfg.checkpoint_save_dir,
        'optimizer': optimizer,
        'optimizer_settings': optimizer_settings,
        'scheduler': scheduler,
        'scheduler_settings': scheduler_settings,
        'trainer': trainer,
        'epoch_trained': 0,
        'global_step':0,
        'training_settings': training_settings,
        'loss_settings': loss_settings,
    }

def load(checkpoint_path):
    '''
    Load the basic model and settings for prediction or training
    '''
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model, model_settings = get_model(
        checkpoint['model_settings']['model_type'], 
        checkpoint['model_settings']['n_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(cfg.device)
    

def resume(checkpoint_path):
    '''
    Resume training from a saved checkpoint, using the configuration
    saved in the checkpoint i.e. model, optimizer, scheduler, etc.
    '''
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    dataset_train, dataset_val = get_datasets()
    loader_train, loader_val = get_loaders(dataset_train, dataset_val, cfg.train_batch_size, cfg.val_batch_size)
    model, model_settings = get_model(
        checkpoint['model_settings']['model_type'], 
        checkpoint['model_settings']['n_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(cfg.device)
    logger = get_logger(checkpoint['logger_dir'])
    
    optimizer, optimizer_settings = get_optimizer_settings(
        checkpoint['optimizer_settings']['type'],
        model,
        **checkpoint['optimizer_settings']['params']
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler, scheduler_settings = get_scheduler_settings(
        checkpoint['scheduler_settings']['type'],
        optimizer,
        **checkpoint['scheduler_settings']['params']
    )
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    loss_criterions, loss_settings = get_loss_criterion(
        checkpoint['loss_settings']['loss_used'],
        **checkpoint['loss_settings']['params']
    )
    trainer = get_trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterions=loss_criterions,
    )
    return {
        'model_id': checkpoint['model_id'],
        'dataset_train': dataset_train,
        'dataset_val': dataset_val,
        'loader_train': loader_train,
        'loader_val': loader_val,
        'model': model,
        'model_settings': model_settings,
        'logger': logger,
        'logger_dir': checkpoint['logger_dir'],
        'checkpoint_save_dir': checkpoint['checkpoint_save_dir'],
        'optimizer': optimizer,
        'optimizer_settings': optimizer_settings,
        'scheduler': scheduler,
        'scheduler_settings': scheduler_settings,
        'trainer': trainer,
        'epoch_trained': checkpoint['epoch_trained'],
        'global_step': checkpoint['global_step'],
        'training_settings': checkpoint['training_settings'],
        'loss_settings': loss_settings
    }
    
    
    
def save(model_id, model_dict, optimizer_dict, scheduler_dict,
         model_settings, optimizer_settings, scheduler_settings, training_settings, loss_settings,
         logger_dir, checkpoint_save_dir, epoch_trained, global_step ):
    checkpoint = {
            'model_id': model_id,
            'model_state_dict': model_dict,
            'optimizer_state_dict': optimizer_dict,
            'scheduler_state_dict': scheduler_dict,
            'model_settings': model_settings,
            'optimizer_settings': optimizer_settings,
            'scheduler_settings': scheduler_settings,
            'training_settings': training_settings,
            'loss_settings': loss_settings,
            'logger_dir': logger_dir,
            'checkpoint_save_dir': checkpoint_save_dir,
            'epoch_trained': epoch_trained,
            'global_step': global_step
        }

    checkpoint_path = Path(checkpoint_save_dir) / f'epoch_{epoch_trained}.pth'
    torch.save(checkpoint, checkpoint_path)
    print(f"Model saved at {checkpoint_save_dir}/epoch_{epoch_trained}.pth")
    
def get_datasets():
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
    model_settings = {}
    if choice_model == 'unet':
        print(f"using model UNet")
        model = UNet(n_channels=3, n_classes=n_classes, bilinear=False)
        model_settings['model_type'] = 'unet'
        model_settings['n_channels'] = 3
        model_settings['n_classes'] = n_classes
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
    
    return model, model_settings

def get_optimizer_settings(optimizer_used:str, model:nn.Module, **parameters):
    optimizer_settings = {
        'type': optimizer_used,
        'params': parameters
    }
    if optimizer_used == 'adam':
        return (
            torch.optim.Adam(model.parameters(), lr=parameters['lr'], weight_decay=parameters['adam_weight_decay']),
            optimizer_settings
        )
    elif optimizer_used == 'sgd':
        return (
            torch.optim.SGD(model.parameters(), lr=parameters['lr'], momentum=parameters['sgd_momentum'], weight_decay=parameters['sgd_weight_decay']),
            optimizer_settings
        )
    elif optimizer_used == 'rmsprop':
        return(
            torch.optim.RMSprop(model.parameters(), lr=parameters['lr'], momentum=parameters['rmsprop_momentum'], weight_decay=parameters['rmsprop_weight_decay']),
            optimizer_settings
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer_used} is not supported")
    
def get_scheduler_settings(scheduler_used:str, optimizer:torch.optim.Optimizer, **parameters):
    scheduler_settings = {
        'type': scheduler_used,
        'params': parameters
    }
    if scheduler_used == 'step':
        return(
            torch.optim.lr_scheduler.StepLR(optimizer, step_size=parameters['step_size'], gamma=parameters['gamma']),
            scheduler_settings
        )
    elif scheduler_used == 'cosine':
        return(
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=parameters['consine_T_max']),
            scheduler_settings
        )
    elif scheduler_used == 'plateau':
        return (
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=parameters['plateau_mode'], factor=parameters['plateau_factor'], patience=parameters['plateau_patience']),
            scheduler_settings
        )
    else:
        raise NotImplementedError(f"Scheduler {scheduler_used} is not supported")

def get_loss_criterion(loss_used: list[str], **parameters):
    criterions = []
    loss_settings = {
        'loss_used': loss_used,
        'params':parameters
    }
    if 'dice' in loss_used:
        criterions.append(loss.DiceLoss(reduction=parameters['reduction']))
    if 'focal' in loss_used:
        criterions.append(loss.FocalLoss(alpha=parameters['focal_alpha'], gamma=parameters['focal_gamma'], beta=parameters['focal_beta'], reduction=parameters['reduction']))
    if 'bce' in loss_used:
        criterions.append(nn.BCEWithLogitsLoss(reduction=parameters['reduction'], pos_weight=parameters['bcd_pos_weight']))
    if loss_used == []:
        raise ValueError("At least one loss criterion must be specified")
    if len(criterions) == 0:
        raise ValueError("No valid loss criteria specified. Please check your configuration.")
    return criterions, loss_settings
    
def get_trainer(model, optimizer, scheduler, criterions):
    choice_model = cfg.model.lower()
    trainer = None
    if choice_model == 'unet':
        trainer = trainers.UNetTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterions=criterions
        )
    elif choice_model == 'deepcrack':
        raise ModuleNotFoundError("not supported yet")
    elif choice_model == 'attention_unet':
        raise ModuleNotFoundError("not supported yet")
    elif choice_model == 'segformer':
        raise ModuleNotFoundError("not supported yet")
    elif choice_model == 'hnet':
        raise ModuleNotFoundError("not supported yet")
    else:
        raise TypeError(f"model {choice_model} does not exist")
    
    return trainer

def get_logger(log_dir):
    return SummaryWriter(log_dir=log_dir)

if __name__ == "__main__":
    settings = resume('test.pth')