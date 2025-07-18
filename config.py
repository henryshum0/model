from pprint import pprint
import torch
from datetime import datetime

from data import data_generation as D

class Config:
    now = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    
    load = False
    resume = False
    
    model = 'unet' #Options: 'deepcrack', 'unet', 'attention_unet', 'segformer', 'hnet' case insensitive
    model_id = f"{model}-{now}"

    #<--------------- path settigs ----------------->
    # image path
    img_tr_dir = 'data/img_tr'
    img_val_dir = 'data/img_val'
    img_test_dir = 'data/img_test'
    mask_tr_dir = 'data/mask_tr'
    mask_val_dir = 'data/mask_val'
    mask_test_dir = 'data/mask_test'
    
    #checkpoint paths
    checkpoint_save_dir = f'checkpoints/{model}/{model_id}'
    checkpoint_load_path = 'test.pth'
    checkpoint_resume_path = ''
    #logger paths
    log_dir_train = f'log/train/{model}/{model_id}'
    log_dir_pred = f'log/pred/{model}/{model_id}'
    #<---------------------------------------------->
    
    # <----------- preprocessing settings ---------->
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
    #<-------------------------------------------->
   
    #<----------- training settings -------------->
    epoch = 500
    lr = 1e-4
    loss = ['bce', 'dice'] # Options: 'bce', 'dice', 'focal'
    train_batch_size = 4
    val_batch_size = 4
    test_batch_size = 4
    
    val_after_epoch = 1 # Validate after every n epochs
    save_after_epoch = 1 # save after every n epochs
    #<-------------------------------------------->
    
    #<----------- optimizer settings ------------->
    optimizer = 'rmsprop'  # Options: 'adam', 'sgd', 'rmsprop', 'none'
    # Adam optimizer settingss
    adam_weight_decay = 1e-4
    
    # SGD settings
    sgd_momentum = 0.9
    sgd_weight_decay = 1e-4
    
    # RMSprop settings
    rmsprop_weight_decay = 1e-4
    rmsprop_momentum = 0.999
    #<-------------------------------------------->
    
    #<----------- scheduler settings ------------->
    scheduler = 'plateau'  # Options: 'step', 'cosine', 'plateau', 'nones
    
    # StepLR scheduler settings
    step_size = 20
    step_gamma = 0.1  
    
    # CosineAnnealingLR scheduler settings
    consine_T_max = 50  # Maximum number of iterations for cosine annealing
    
    # ReduceLROnPlateau scheduler settings
    plateau_mode = 'max' #Options: 'min', 'max'
    plateau_factor = 0.5
    plateau_patience = 10
    #<-------------------------------------------->
    
    #<-------------- loss settings --------------->
    reduction = 'mean'  # Options: 'mean', 'sum'
    
    #bcd parameters
    bcd_pos_weight=torch.tensor([10]).to(device) # Weight for positive examples in BCE loss
    
    #focal loss parameters
    focal_alpha = 0.75  # Weight for positive examples in Focal Loss
    focal_gamma = 3.0  # Focusing parameter (higher values focus more on hard examples)
    focal_beta = 1.0   # Global scaling factor for the focal term
    pos_pixel_weight = 1  # Legacy parameter, used when use_focal_loss = False
    acc_sigmoid_th = 0.5
    #<-------------------------------------------->
    
    
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