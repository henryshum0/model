from torch import nn
import torch

#import utils
from config import Config as cfg
from loss import FocalLoss, DiceLoss


class UNetTrainer(nn.Module):
    """Trainer for single-output models like UNet and HNet"""
    from typing import Union

    def __init__(self, model, optimizer, scheduler, 
                 criterions: list[Union[FocalLoss, DiceLoss, nn.BCEWithLogitsLoss]]):
        super().__init__()
        self.model = model
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.criterions = criterions

    def train_op(self, input:torch.Tensor, target:torch.Tensor):
        #input is a batch
        self.model.train()
        self.optimizer.zero_grad()

        pred = self.model(input)
        total_loss, log_loss = self.calculate_loss(pred, target, requires_grad=True) 
        total_loss.backward()
        self.optimizer.step()

        return log_loss

    def val_op(self, input:torch.Tensor, target:torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input)
            total_loss, log_loss = self.calculate_loss(pred, target, requires_grad=False)
        self.scheduler.step(total_loss)
        return log_loss

    def calculate_loss(self, pred:torch.Tensor, target:torch.Tensor, requires_grad=True):
        log_loss = {}
        total_loss = torch.tensor(0.0, requires_grad=requires_grad, device=pred.device)
        #for handling binary class
        for criterion in self.criterions:
            if isinstance(criterion, FocalLoss):
                focal_loss = criterion(pred.view(-1, 1), target.view(-1, 1))
                total_loss = total_loss + focal_loss
                log_loss['focal_loss'] = focal_loss.item()
            elif isinstance(criterion, DiceLoss):
                dice_loss = criterion(pred, target)
                total_loss = total_loss + dice_loss
                log_loss['dice_loss'] = dice_loss.item()
            elif isinstance(criterion, nn.BCEWithLogitsLoss):
                bce_loss = criterion(pred.view(-1,1), target.view(-1,1))
                total_loss = total_loss + bce_loss
                log_loss['bce_loss'] = bce_loss.item()
        
        #TODO: handling multi class

        log_loss['total_loss'] = total_loss.item()
        return total_loss, log_loss