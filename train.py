import torch
from pathlib import Path
from config import Config as cfg
from tqdm import tqdm
from load_save import load_from_cfg, save, resume

def main():
    if cfg.load:
        pass
    elif cfg.resume:
        settings = resume(cfg.checkpoint_load_path)
    else:
        settings = load_from_cfg()
    model_id = settings['model_id']
    dataset_train = settings['dataset_train']
    dataset_val = settings['dataset_val']
    loader_train = settings['loader_train']
    loader_val = settings['loader_val']
    model = settings['model']
    model_settings = settings['model_settings']
    logger = settings['logger']
    logger_dir = settings['logger_dir']
    optimizer = settings['optimizer']
    optimizer_settings = settings['optimizer_settings']
    scheduler = settings['scheduler']
    scheduler_settings = settings['scheduler_settings']
    trainer = settings['trainer']
    device = settings['device']
    epoch_trained = settings['epoch_trained']
    global_step = settings['global_step']
    checkpoint_save_dir = settings['checkpoint_save_dir']
    training_settings = settings['training_settings']
    loss_settings = settings['loss_settings']
    
    print(f"Training on device: {device}")
    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_val)}")
    for key, value in training_settings.items():
        print(f"{key}: {value}")
    print(f"Loss functions used: {loss_settings['loss_used']}")
    print(f"Model type: {model_settings['model_type']}")
    print(f"Optimizer: {optimizer_settings['type']} parameters: {optimizer_settings['params']}")
    print(f"Scheduler: {scheduler_settings['type']} parameters: {scheduler_settings['params']}")
    print(f"Checkpoint will be saved to: {checkpoint_save_dir}")
    print(f"Logger directory: {logger_dir}")
    
    try:
        for epoch in range(epoch_trained, training_settings['epoch']+1):
            epoch_loss = 0.0
            bce_loss = 0.0
            dice_loss = 0.0
            focal_loss = 0.0
            print(f"Epoch {epoch}/{training_settings['epoch']}")
            
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
                save(
                    model_id,
                    model.state_dict(), 
                    optimizer.state_dict(),
                    scheduler.state_dict(),
                    model_settings, 
                    optimizer_settings, 
                    scheduler_settings, 
                    training_settings,
                    loss_settings,
                    logger_dir, 
                    checkpoint_save_dir,
                    epoch, 
                    global_step
                )
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model...")
        
        save(
            model_id,
            model.state_dict(), 
            optimizer.state_dict(),
            scheduler.state_dict(),
            model_settings, 
            optimizer_settings, 
            scheduler_settings, 
            training_settings,
            loss_settings,
            logger_dir, 
            checkpoint_save_dir,
            epoch, 
            global_step, 
        )
        print("Model saved. Exiting...")
        

main()