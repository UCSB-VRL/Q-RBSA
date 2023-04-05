import argparse
import torch

class Argparser:
    """
    The actual argparser

    """ 
    def __init__(self):
        self.args = self.prepare_arg_parser().parse_args()

    def prepare_arg_parser(self):
        """
        Add all args to the argparser     

        """
        
        arg_parser = argparse.ArgumentParser()
    
        # Hardware specifications
    
        arg_parser.add_argument('--GPU_ID', type=str, default='4',
                                help='Which gpu to run on your program')
        arg_parser.add_argument('--cpu', action='store_true',
                                help='use cpu only')
        arg_parser.add_argument('--chop', action='store_true',
                                help='enable memory-efficient forward')
        arg_parser.add_argument('--n_GPUs', type=int, default=1,
                                help='number of GPUs')
        arg_parser.add_argument('--device', type=str,
                                default= torch.device("cuda"),
                                help='use the given device (cuda/cpu) for training')
        arg_parser.add_argument('--precision', type=str, default='single',
                                choices=('single', 'half'),
                                help='FP precision for test (single | half)') 
 
        
        # Train, Val, Test DataSet specifications
         
        arg_parser.add_argument('--input_dir', type=str, 
                                default='/data/dkjangid/superresolution/Material_Dataset/Ti7_Deformed_Afterdef_HomoCubo',
                                help=' directory path of input datasets')
        arg_parser.add_argument('--lr_data_dir', type=str,  
                                default='LR_Images/X4/train',
                                help=' path of low resolution training dataset')
        arg_parser.add_argument('--hr_data_dir', type=str,  
                                default='HR_Images/train',
                                help=' path of high resolution training datasets')
        arg_parser.add_argument('--val_lr_data_dir', type=str,
                                default='LR_Images/X4/val',
                                help='path of low resolution validation dataset')
        arg_parser.add_argument('--val_hr_data_dir', type=str,  
                                default='HR_Images/val',
                                help='path of high resolution validation datasets')
        arg_parser.add_argument('--batch_size', type=int, default=16,
                                help='Batch size of training datasets')
        arg_parser.add_argument('--val_batch_size', type=int, default=1,
                                help='batch size of validation datasets')
        arg_parser.add_argument('--patch_size', type=int, default=32,
                                help='output patch size from the network')
        arg_parser.add_argument('--rgb_range', type=int, default=2,
                                help='maximum value of RGB')
        arg_parser.add_argument('--n_colors', type=int, default=3,
                                help='number of channels to use')
        arg_parser.add_argument('--scalar_first', action='store_true', 
                                help = 'Format for Quaternion data') 
        arg_parser.add_argument('--scale', type=int, default=4,
                                help='super resolution scale')
        arg_parser.add_argument('--noise', type=str, default='.',
                                help='Gaussian noise std.')
        arg_parser.add_argument('--test_dataset_type', type=str, default='val',
                                help='which dataset you want to test')

        arg_parser.add_argument('--upsample_2d', action='store_true',
                                 help='1D or 2D upsampling') 



        # Models specificaitons
 
        arg_parser.add_argument('--model', default='san',
                                help='name of super-resolution model')        
        arg_parser.add_argument('--act', type=str, default='relu',
                                help='activation function')
        arg_parser.add_argument('--n_resblocks', type=int, default=2,
                                help='number of residual blocks')
        arg_parser.add_argument('--n_resgroups', type=int, default=5,
                                help='number of residual groups')
        arg_parser.add_argument('--reduction', type=int, default=16,
                                help='number of feature maps reduction')
        arg_parser.add_argument('--n_feats', type=int, default=64,
                                help='number of feature maps')
        arg_parser.add_argument('--res_scale', type=float, default=1,
                                help='residual scaling')
        arg_parser.add_argument('--shift_mean', default=True,
                                help='subtract pixel mean from the input')
        arg_parser.add_argument('--save_results', action='store_false', 
                                help='Save output results')
        arg_parser.add_argument('--test_only', action='store_true',
                                help='set this option to test the model')
        arg_parser.add_argument('--print_model', action='store_true',
                                help='print model')

        # Parameters saving specificaitons

        arg_parser.add_argument('--print_every', type=int, default=1,
                                help='how many batches to wait before logging training status')
        arg_parser.add_argument('--save_model_freq', type=int, default=500,
                                help='how many epoch after we save the model')
        arg_parser.add_argument('--pre_train', type=str, default='.',
                                help='pre-trained model directory')
        arg_parser.add_argument('--save', type=str, default='SAN_Misorientation_Cubo_Symm_debug',
                                help='file name to save trained model')

        arg_parser.add_argument('--load', type=str, default='.',
                                help='file name to load')

        arg_parser.add_argument('--save_models', action='store_false', 
                                help = 'save all intermediate models')
        arg_parser.add_argument('--self_ensemble', action='store_true',
                                help='use self-ensemble method for test')
        arg_parser.add_argument('--model_to_load', type=str, default='model_best',
                                help='file name to load')


                        
        #Training Parameters

        arg_parser.add_argument('--leak_value', type=float, default=0.2,
                                help='leak value in leaky relu')
        arg_parser.add_argument('--lr', type=float, default=0.0002,
                                help='learning rate')
        arg_parser.add_argument('--lr_decay', type=int, default=100,
                                help='learning rate decay per N epochs')
        arg_parser.add_argument('--optim_lr_min', type=float, default=1e-6, 
                                help=' minimum learning rate for optimizer')
        arg_parser.add_argument('--decay_type', type=str, default='step',
                                help='learning rate decay type')
        arg_parser.add_argument('--gamma', type=float, default=0.6,
                                help='learning rate decay factor for step decay')
        arg_parser.add_argument('--optimizer', default='ADAM',
                                choices=('SGD', 'ADAM', 'RMSprop'),
                                help='optimizer to use (SGD | ADAM | RMSprop)')
        arg_parser.add_argument('--momentum', type=float, default=0.9,
                                help='SGD momentum')
        arg_parser.add_argument('--beta1', type=float, default=0.9,
                                help='ADAM beta1')
        arg_parser.add_argument('--beta2', type=float, default=0.99,
                                help='ADAM beta2')
        arg_parser.add_argument('--epsilon', type=float, default=1e-8,
                                help='ADAM epsilon for numerical stability')
        arg_parser.add_argument('--weight_decay', type=float, default=0,
                                help='weight decay')

        arg_parser.add_argument('--epochs', type=int, default=2002,
                                help='number of epochs to train')
        arg_parser.add_argument('--current_epoch', type=int, default=0,
                                help='current epoch training/epoch to start with')     
        arg_parser.add_argument('--reset', action='store_true',
                                help='reset the training')
        arg_parser.add_argument('--gan_k', type=int, default=1,
                                help='k value for adversarial loss')
        arg_parser.add_argument('--loss', type=str, default='1*MisOrientation',
                                 help='loss function configuration')
        arg_parser.add_argument('--dist_type', type=str, default='rot_dist_approx',
                                 help='select from [l1, l2, rot_dist,rot_dist_approx ]')
        arg_parser.add_argument('--act_loss', type=str, default=None,
                                 help='select from [None, tanhc ]')
        arg_parser.add_argument('--syms_type', type=str, default='HCP',
                                 help='select from [HCP, FCC ]')
        arg_parser.add_argument('--syms_req', action='store_true',
                                 help='want hcp syms or not')
        arg_parser.add_argument('--prog_patch', action='store_true',
                                 help='progressive patch size during training') 
 
        arg_parser.add_argument('--skip_threshold', type=float, default='1e6',
                                help='skipping batch that has large error')
        arg_parser.add_argument('--extend', type=str, default='.',
                                help='pre-trained model directory')
        arg_parser.add_argument('--resume', type=int, default=0,
                                help='resume from specific checkpoint')
        arg_parser.add_argument('--val_freq', type=int, default=200,
                                help='number of epochs to train')

 
       
        return arg_parser 
    
