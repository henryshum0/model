import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from data.crack_dataset import CrackDataset
from config import Config as cfg
from models.unet import UNet
import trainers, loss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def main():
    device = torch.device(f'cuda:{cfg.gpu_id}' if torch.cuda.is_available() else 'cpu')
    # 1.Initialize the datasets and loaders
    dataset_train, dataset_val = get_datasets()
    loader_train, loader_val = get_loaders(dataset_train, dataset_val)
    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_val)}")
    print(f"Training batch size: {loader_train.batch_size}")
    print(f"Validation batch size: {loader_val.batch_size}")
    
    assert dataset_train.get_n_classes() == dataset_val.get_n_classes(), \
        "Number of classes in training and validation datasets must match."
        
    # 2.Get the model, optimizer, scheduler, loss criterion, and trainer
    if cfg.load:
        pass
    else:
        model = get_model(dataset_train.get_n_classes()).to(device)
        logger = get_logger()
        optimizer = get_optimizer(model)
        scheduler = get_scheduler(optimizer)
        loss_criterions = get_loss_criterion()
        trainer = get_trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterions=loss_criterions,
        )
    
    # 3. start training
    global_step = 0
    
    # create checkpoint directory if it doesn't exist
    checkpoint_save  = Path(cfg.checkpoint_path)
    checkpoint_save.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, cfg.epoch+1):
        epoch_loss = 0.0
        bce_loss = 0.0
        dice_loss = 0.0
        focal_loss = 0.0
        print(f"Epoch {epoch}/{cfg.epoch}")
        
        #training round
        with tqdm(total=len(loader_train), desc=f"Training") as pbar:
            for batch in loader_train:
                images, masks = batch['image'], batch['mask']
                assert images.shape[1] == model.n_channels, \
                    f"Expected {model.n_channels} channels, got {images.shape[1]}"
                assert masks.shape[1] == model.n_classes, \
                    f"Expected {model.n_classes} classes, got {masks.shape[1]}"
            
                images = images.to(device)
                masks = masks.to(device)
                
                # Train the model
                log_loss = trainer.train_op(images, masks)
                
                # get the total losses for each batch
                epoch_loss += log_loss['total_loss']
                bce_loss += log_loss.get('bce_loss', 0.0)
                dice_loss += log_loss.get('dice_loss', 0.0)
                focal_loss += log_loss.get('focal_loss', 0.0)
                #set progress bar after each batch
                pbar.set_postfix({'loss(batch)': log_loss['total_loss']})
                pbar.update(1)
                
                global_step += 1
            
            avg_epoch_loss = epoch_loss / len(loader_train)
            # Set the average loss for the epoch
            pbar.set_postfix({'loss_avg(epoch)': avg_epoch_loss})
            # log the average loss for the epoch
            logger.add_scalar('train/epoch_loss', avg_epoch_loss, epoch)
            logger.add_scalar('train/epoch_bce_loss', bce_loss / len(loader_train), epoch)
            logger.add_scalar('train/epoch_dice_loss', dice_loss / len(loader_train), epoch)
            logger.add_scalar('train/epoch_focal_loss', focal_loss / len(loader_train), epoch)
        pbar.close()
        
        # Validation round
        val_loss = 0.0
        val_bce_loss = 0.0
        val_dice_loss = 0.0
        val_focal_loss = 0.0
        if epoch % cfg.val_after_epoch == 0:
            with torch.no_grad():
                with tqdm(total=len(loader_val), desc=f"Validation") as pbar:
                    for batch in loader_val:
                        images, masks = batch['image'], batch['mask']
                        images = images.to(device)
                        masks = masks.to(device)
                        
                        log_loss = trainer.val_op(images, masks)
                        val_loss += log_loss['total_loss']
                        val_bce_loss += log_loss.get('bce_loss', 0.0)
                        val_dice_loss += log_loss.get('dice_loss', 0.0)
                        val_focal_loss += log_loss.get('focal_loss', 0.0)
                        # progress bar for every batch
                        pbar.set_postfix({'val_loss(batch)': log_loss['total_loss']})
                        pbar.update(1)
                    
                    #progress bar for the whole validation round    
                    pbar.set_postfix({'val_loss_avg(epoch)': val_loss / len(loader_val)})
                    pbar.close()
                    
        # log the average validation loss for the epoch
        avg_val_loss = val_loss / len(loader_val)
        logger.add_scalar('val/epoch_bce_loss', val_bce_loss / len(loader_val), epoch)
        logger.add_scalar('val/epoch_dice_loss', val_dice_loss / len(loader_val), epoch)
        logger.add_scalar('val/epoch_focal_loss', val_focal_loss / len(loader_val), epoch)
        logger.add_scalar('val/epoch_loss', avg_val_loss, epoch)

        #save the model checkpoint
        if epoch % cfg.save_after_epoch == 0:
            torch.save(model.state_dict(), f"{cfg.checkpoint_path}/epoch_{epoch}.pth")
            print(f"Model saved at {cfg.checkpoint_path}/epoch_{epoch}.pth")
        
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
    
    return dataset_train, dataset_val

def get_loaders(dataset_tr, dataset_val):
    
    dataloader_tr = DataLoader(
        dataset_tr,
        batch_size=cfg.train_batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )
    
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return dataloader_tr, dataloader_val
    
def get_model(n_classes=1):
    choice_model = cfg.model.lower()
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

def get_optimizer(model:nn.Module):
    scheduler = cfg.optimizer.lower()
    
    if scheduler == 'adam':
        print("Using Adam optimizer")
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif scheduler == 'sgd':
        print("Using SGD optimizer")
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum, )
    else:
        print("Using RMSprop optimizer")
        return torch.optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, momentum=cfg.momentum)
    
def get_scheduler(optimizer):
    scheduler = cfg.scheduler.lower()
    
    if scheduler == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=cfg.lr_decay)
    elif scheduler == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch, eta_min=0)
    else:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

def get_loss_criterion():
    criterions = []
    if cfg.use_dice_loss:
        criterions.append(loss.DiceLoss(reduction=cfg.reduction))
        print("Using Dice Loss")
    if cfg.use_focal_loss:
        criterions.append(loss.FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma, beta=cfg.focal_beta, reduction=cfg.reduction))
        print("Using Focal Loss")
    if cfg.use_bce_loss or criterions == []:
        criterions.append(nn.BCEWithLogitsLoss(reduction=cfg.reduction, pos_weight=torch.tensor(10.0)))
        print("Using Binary Cross Entropy Loss")
    return criterions
    
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
        print("Using UNetTrainer")
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

def get_logger():
    return SummaryWriter(log_dir=cfg.log_dir)




main()