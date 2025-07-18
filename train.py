import torch
from pathlib import Path
from config import Config
from tqdm import tqdm
from utility.load_save import load_from_cfg, save, resume

def train(cfg:Config, train_type:str='fresh', **kwargs):
    # getting the required modules for training
    if train_type.lower() == 'load':
        pass
    elif train_type.lower() == 'resume':
        settings = resume(cfg, model_state_dict=kwargs['model_state_dict'],
                          optimizer_state_dict=kwargs['optimizer_state_dict'],
                          scheduler_state_dict=kwargs['scheduler_state_dict'])
    elif train_type.lower() == 'fresh':
        settings = load_from_cfg(cfg)
    else:
        raise ValueError(f"Invalid train_type: {train_type}. Use 'load', 'resume', or 'fresh'.")
    dataset_train = settings['dataset_train']
    dataset_val = settings['dataset_val']
    loader_train = settings['loader_train']
    loader_val = settings['loader_val']
    model = settings['model']
    logger = settings['logger']
    optimizer = settings['optimizer']
    scheduler = settings['scheduler']
    trainer = settings['trainer']
    
    #print training configuration
    print(f"Model ID: {cfg.model_id}")
    print(f"Model type: {cfg.model}")
    print(f"Training type: {train_type}")
    print(f"Epoch: {cfg.epoch}")
    print(f"lr: {cfg.lr}")
    print(f"Loss functions: {cfg.loss}")
    print(f"Optimizer: {cfg.optimizer}")
    print(f"Scheduler: {cfg.scheduler}")
    print(f"Checkpoint will be saved to: {cfg.checkpoint_save_dir}")
    print(f"Logger directory: {cfg.log_dir_train}")
    print(f"Training dataset size: {len(dataset_train)}")
    print(f"Validation dataset size: {len(dataset_val)}")
    print(f"Training on device: {cfg.device}")

    # training loop
    try:
        global_step = cfg.global_step
        for epoch in range(cfg.epoch_trained, cfg.epoch+1):
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
                
                    images = images.to(cfg.device)
                    masks = masks.to(cfg.device)
                    
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
                            images = images.to(cfg.device)
                            masks = masks.to(cfg.device)
                            
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
                    cfg,
                    model.state_dict(), 
                    optimizer.state_dict(),
                    scheduler.state_dict(),
                    epoch, 
                    global_step
                )

    # stop the training                
    except KeyboardInterrupt:
        print("Training interrupted. Saving the model...")
        save(
            cfg,
            model.state_dict(), 
            optimizer.state_dict(),
            scheduler.state_dict(),
            epoch, 
            global_step
        )
        print("Model saved. Exiting...")
      
      
if __name__ == "__main__":  
    cfg = Config()
    train(cfg)