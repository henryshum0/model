from pprint import pprint
import torch
from datetime import datetime

from data import data_generation as D

class Config:
    def __init__(self):
        self.now = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

        #<-------------- model settings ---------------->
        self.model = 'unet' #Options: 'deepcrack', 'unet', 'attention_unet', 'segformer', 'hnet' case insensitive
        self.model_id = f"{self.model}-{self.now}"  #model id must not be constant expression, i.e. not changing
        self.n_channels = 3  # Number of input channels (e.g., RGB images)
        self.n_classes = 1  # Number of output classes (e.g., binary segmentation)
        #<---------------------------------------------->
        
        #<--------------- path settigs ----------------->
        # image path
        self.img_tr_dir = 'data/img_tr'
        self.img_val_dir = 'data/img_val'
        self.img_test_dir = 'data/img_test'
        self.mask_tr_dir = 'data/mask_tr'
        self.mask_val_dir = 'data/mask_val'
        self.mask_test_dir = 'data/mask_test'
        
        #checkpoint paths
        self.checkpoint_save_dir = f'checkpoints/{self.model}/{self.model_id}'
        self.checkpoint_load_path = 'test.pth'
        self.checkpoint_resume_path = ''
        #logger paths
        self.log_dir_train = f'log/train/{self.model}/{self.model_id}'
        self.log_dir_pred = f'log/pred/{self.model}/{self.model_id}'
        #prediction result paths
        self.pred_save_dir = f'predictions/{self.model}/{self.model_id}'
        #<---------------------------------------------->
        
        # <----------- preprocessing settings ---------->
        self.target_width = 448
        self.target_height = 448
        self.target_size = (self.target_width, self.target_height)
        self.min_size = 448   # Minimum size for images
        self.transforms_tr = [
            D.Resize(self.target_size),
            #D.RandomCrop((self.target_height, self.target_width), has_mask=0.95) #to specify the probability of cropping an image with a crack
                                                                    #if you are planning on using a big image as input
        ]
        self.transforms_val = [D.Resize(self.target_size)]
        self.transforms_test = [D.Resize(self.target_size)]
        #<-------------------------------------------->
   
        #<----------- training settings -------------->
        self.epoch = 20
        self.epoch_trained = 0
        self.global_step = 0
        self.lr = 1e-4
        self.loss = ['bce', 'dice'] # Options: 'bce', 'dice', 'focal'
        self.train_batch_size = 4
        self.val_batch_size = 4
        self.test_batch_size = 4
        
        self.val_after_epoch = 1 # Validate after every n epochs
        self.save_after_epoch = 1 # save after every n epochs
        #<-------------------------------------------->
    
        #<----------- optimizer settings ------------->
        self.optimizer = 'rmsprop'  # Options: 'adam', 'sgd', 'rmsprop'
    
        # Adam optimizer settingss
        self.adam_weight_decay = 1e-4
    
        # SGD settings
        self.sgd_momentum = 0.9
        self.sgd_weight_decay = 1e-4
    
        # RMSprop settings
        self.rmsprop_weight_decay = 1e-4
        self.rmsprop_momentum = 0.999
        #<-------------------------------------------->
    
        #<----------- scheduler settings ------------->
        self.scheduler = 'plateau'  # Options: 'step', 'cosine', 'plateau', 'nones
    
        # StepLR scheduler settings
        self.step_size = 20
        self.step_gamma = 0.1  
    
        # CosineAnnealingLR scheduler settings
        self.consine_T_max = 50  # Maximum number of iterations for cosine annealing
    
        # ReduceLROnPlateau scheduler settings
        self.plateau_mode = 'max' #Options: 'min', 'max'
        self.plateau_factor = 0.5
        self.plateau_patience = 10
        #<-------------------------------------------->
    
        #<-------------- loss settings --------------->
        self.reduction = 'mean'  # Options: 'mean', 'sum'
    
        #bcd parameters
        self.bcd_pos_weight=torch.tensor([10]).to(self.device) # Weight for positive examples in BCE loss
    
        #focal loss parameters
        self.focal_alpha = 0.75  # Weight for positive examples in Focal Loss
        self.focal_gamma = 3.0  # Focusing parameter (higher values focus more on hard examples)
        self.focal_beta = 1.0   # Global scaling factor for the focal term
        self.pos_pixel_weight = 1  # Legacy parameter, used when use_focal_loss = False
        self.acc_sigmoid_th = 0.5
        #<-------------------------------------------->
    
    
        # SegFormer-specific configuration
        self.segformer_variant = 'b5'  # Options: 'b0', 'b1', 'b2', 'b3', 'b4', 'b5'
        self.segformer_pretrained = False  # Whether to use ImageNet pretrained weights
    
    
        # HNet-specific configuration
        self.group_norm_groups = 8  # Number of groups for GroupNorm layers
    
        # Multi-scale convolution parameters
        self.ms_dilations = [1, 2, 4]  # Dilation rates for multi-scale convolution

