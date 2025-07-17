from pprint import pprint
import os
from datetime import datetime

from data import data_generation as D

class Config:
    load = False
    model = 'unet' #Options: 'deepcrack', 'unet', 'attention_unet', 'segformer', 'hnet' case insensitive
    now = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    name = f'{model}_{now}'

    gpu_id = '0'
    
    #<-- path settigs -->
    
    # image path
    img_tr_dir = 'data/crop_img_tr'
    img_val_dir = 'data/crop_img_val'
    img_test_dir = 'data/crop_img_test'
    mask_tr_dir = 'data/crop_msk_tr'
    mask_val_dir = 'data/crop_msk_val'
    mask_test_dir = 'data/crop_msk_test'
    
    #checkpoint paths
    checkpoint_path = f'checkpoints/{model}/{now}'
    
    #logger paths
    log_dir = f'log/{model}/{now}'
    #//// path settigs ////
    
    

    # <-- preprocessing settings -->
    target_width = 448
    target_height = 448
    target_size = (target_width, target_height)
    min_size = 448   # Minimum size for images
    transforms_tr = [
        D.Resize(target_size),
        #D.RandomCrop((target_height, target_width), has_mask=0.95) #to specify the probability of cropping an image with a crack
                                                                    #if you are planning on using a big image as input
    ]
    transforms_val = [D.Resize(target_size)]
    transforms_test = [D.Resize(target_size)]
    #/// preprocessing settings ///
   
    #<-- training settings -->
    optimizer = 'rmsprop'  # Options: 'adam', 'sgd', 'rmsprop', 'none'
    momentum = 0.999
    weight_decay = 1e-4
    scheduler = 'plateau'  # Options: 'step', 'cosine', 'plateau', 'nones
    plateau_mode = 'max' #Options: 'min', 'max'
    
    epoch = 500
    pretrained_model = ''  # Path to the pretrained model
    lr = 1e-4
    train_batch_size = 4
    val_batch_size = 4
    test_batch_size = 4
    
    val_after_epoch = 1 # Validate after every n epochs
    save_after_epoch = 1 # save after every n epochs
    #/// training settings ///
    
    #<-- loss settings -->
    use_dice_loss = False #Whether to use Dice Loss
    use_focal_loss = False  # Whether to use Focal Loss
    use_bce_loss = True  # Whether to use Binary Cross-Entropy Loss
    reduction = 'mean'  # Options: 'mean', 'sum'
    
    #focal loss parameters
    focal_alpha = 0.75  # Weight for positive examples in Focal Loss
    focal_gamma = 3.0  # Focusing parameter (higher values focus more on hard examples)
    focal_beta = 1.0   # Global scaling factor for the focal term
    pos_pixel_weight = 1  # Legacy parameter, used when use_focal_loss = False
    acc_sigmoid_th = 0.5
    #/// loss settings ///
    
    
    # checkpointer
    save_format = ''
    save_acc = -1
    save_pos_acc = -1    # Model configuration
    # model_type = 'deepcrack'  # Options: 'deepcrack', 'unet', 'attention_unet', 'segformer', 'hnet'
    
    
    # SegFormer-specific configuration
    segformer_variant = 'b5'  # Options: 'b0', 'b1', 'b2', 'b3', 'b4', 'b5'
    segformer_pretrained = False  # Whether to use ImageNet pretrained weights
    
    
    # HNet-specific configuration
    group_norm_groups = 8  # Number of groups for GroupNorm layers
    
    # Multi-scale convolution parameters
    ms_dilations = [1, 2, 4]  # Dilation rates for multi-scale convolution
    
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

    def show(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')