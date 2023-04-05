import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.autograd import Variable
from model.quat_utils.quaternion_layers import QuaternionConv, QuaternionLinear#, QuaternionTransposeConv #, QuaternionLinear
# from torch.nn import Parameter
from model.quat_utils.QBN_Vecchi2 import QuaternionBatchNorm2d as QBatchNorm
from model.quat_utils.QBN_Vecchi2 import QuatLayerNorm
from model.quat_utils.QSN2 import Qspectral_norm

from einops import rearrange

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample,self).__init__()    
        
    def forward(self,input):
        value = input.permute(0,2,3,1)
        sh = [int(x) for x in value.shape]
        dim = len(sh[1:-1])
        
        out = (torch.reshape(value, [-1] + sh[-dim:]))
        
        for i in range(dim, 0, -1):
        
          out = torch.cat([out, torch.zeros(out.shape, device=(out.device))], i)
        
        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = torch.reshape(out, out_size)

        return out.permute(0,3,1,2)

def spectral_norm(module, name='weight'):
    module = torch.nn.utils.spectral_norm(module, name=name)
    return module



class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding=1,
                spectral_normed = False):
        super(conv2d, self).__init__()

        self.conv = QuaternionConv(in_channels, out_channels, kernel_size, stride, padding = padding)
        torch.nn.init.normal_(self.conv.r_weight.data, std=0.02)
        torch.nn.init.normal_(self.conv.i_weight.data, std=0.02)
        torch.nn.init.normal_(self.conv.j_weight.data, std=0.02)
        torch.nn.init.normal_(self.conv.k_weight.data, std=0.02)
        
        if spectral_normed:
            self.conv = Qspectral_norm(self.conv)

    def forward(self, input):
        out = self.conv(input)
        return out


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class QFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor):
        super(QFeedForward, self).__init__()

        #import pdb; pdb.set_trace()
        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = conv2d(dim, hidden_features*2, kernel_size=1, stride=1, padding=0)

        self.dwconv = conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1)

        self.project_out = conv2d(hidden_features, dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################


## Multi-DConv Head Transposed Self-Attention (MDTA)
class QAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(QAttention, self).__init__()
        #import pdb; pdb.set_trace()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = conv2d(dim, dim*3, kernel_size=1, stride=1, padding=0)
        self.qkv_dwconv = conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1)
        self.project_out = conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class QTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=1):
        super(QTransformerBlock, self).__init__()       
        self.norm1 = QuatLayerNorm(dim)
        self.attn = QAttention(dim, num_heads)
        self.norm2 = QuatLayerNorm(dim)
        self.ffn = QFeedForward(dim, ffn_expansion_factor)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        #x = x + self.attn(x)
        #x = x + self.ffn(x)

        return x  

class Residual_SA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3, stride = 1,
                spectral_normed = False, down_sampling = False):
        super(Residual_SA, self).__init__()  
 
        #import pdb; pdb.set_trace() 
        self.conv1 = conv2d(in_channels, in_channels, kernel_size = kernel, padding = 1, stride = stride,  spectral_normed = spectral_normed)

        self.relu = nn.ReLU()

        self.conv2 = conv2d(in_channels, out_channels,  kernel_size = kernel, stride = stride, padding = 1, spectral_normed = spectral_normed)
    
        self.sa = QTransformerBlock(out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu(y)
        y = self.conv2(y)

        y = self.sa(y)

        y += x

        return y


class Residual_G(nn.Module):
    def __init__(self, in_channels, out_channels = 256, kernel_size = 3, stride = 1, 
                batch_normed=False, up_sampling = False, n_classes=0):
        super(Residual_G, self).__init__()
        
        self.batch_normed = batch_normed
        self.up_sampling = up_sampling
        self.diff_dims = in_channels != out_channels or self.up_sampling

        
        self.relu = nn.ReLU()
        
        self.batch_norm1 = QBatchNorm(in_channels)
        self.batch_norm2 = QBatchNorm(out_channels)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'nearest')
        # self.upsample = Upsample()
        
        self.conv1 = conv2d(in_channels, out_channels,
                            kernel_size = kernel_size, stride = stride, padding = 1)
        self.conv2 = conv2d(out_channels, out_channels, 
                            kernel_size = kernel_size, stride = stride, padding = 1)

        # if classes > 0 : # implementation needed for CategoricalConditionalBatchNormalization
        #     self.b1 = CategoricalConditionalBatchNormalization(in_channels, n_cat=n_classes)
        #     self.b2 = CategoricalConditionalBatchNormalization(hidden_channels, n_cat=n_classes)
        # else:
        #     self.b1 = L.BatchNormalization(in_channels)
        #     self.b2 = L.BatchNormalization(hidden_channels)
        
        # if residual block has output features different from input features
        if self.diff_dims:    
            self.short_conv = conv2d(in_channels, out_channels,
                            kernel_size = 1, stride = stride, padding = 0)



            
    def residual(self, x):
        # input = x
        if self.batch_normed:
            x = self.relu(self.batch_norm1(x))
        else:
            x = self.relu(x)

        if self.up_sampling:  
            x = self.upsample(x)
        x = self.conv1(x)
        
        if self.batch_normed:
            x = self.batch_norm2(x)
        y = self.conv2(self.relu(x))
        return y
    
    
    def shortcut(self, x):
        if self.diff_dims:
            if self.up_sampling:
                x = self.upsample(x)
            x = self.short_conv(x)
            return x
        else:
            return x
        
    def forward(self, input):
        
        return self.residual(input) + self.shortcut(input)
        

class Residual_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3, stride = 1,
                spectral_normed = False, down_sampling = False):
        super(Residual_D, self).__init__()
        
        self.down_sampling = down_sampling
        self.diff_dims = (in_channels!=out_channels) or self.down_sampling

        self.conv1 = conv2d(in_channels, in_channels, 
                            kernel_size = kernel, stride = stride, padding = 1, spectral_normed = spectral_normed)

        self.relu = nn.ReLU()

        self.conv2 = conv2d(in_channels, out_channels, 
                            kernel_size = kernel, stride = stride, padding = 1, spectral_normed = spectral_normed)
    
        
        if self.diff_dims:
            self.conv_short = conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0,
                                spectral_normed = spectral_normed)
            
            if self.down_sampling:      
                self.avgpool_short = nn.AvgPool2d(2, 2, padding = 1)

        if self.down_sampling:
            self.avgpool2 = nn.AvgPool2d(2, 2, padding = 1) 

    def shortcut(self, x):
        short = x
        if self.diff_dims:
            short = self.conv_short(short)
    
            if self.down_sampling:
                return self.avgpool_short(short)
            else:
                return short
                
        else:
            return short
        
        
    def residual(self, x):
        #conv1 = self.conv1(self.relu(x))
        #import pdb; pdb.set_trace()
        conv1 = self.conv1(x)
        conv1 = self.relu(conv1) 
        conv2 = self.conv2(conv1)
        if self.down_sampling:
            conv2 = self.avgpool2(conv2)
        return conv2
        
    
    def forward(self, x):
        #import pdb; pdb.set_trace()
        return self.residual(x) + self.shortcut(x)



class First_Residual_D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel = 3, stride = 1,
                spectral_normed = False):
        super(First_Residual_D, self).__init__()
        
        self.avgpool_short = nn.AvgPool2d(2, 2, padding = 1)
        
        self.conv1 = conv2d(in_channels, out_channels, spectral_normed = spectral_normed,
                            kernel_size = kernel, stride = stride, padding = 1)
    
        self.conv2 = conv2d(out_channels, out_channels, spectral_normed = spectral_normed,
                            kernel_size = kernel, stride = stride, padding = 1)
       
        self.avgpool2 = nn.AvgPool2d(2, 2, padding = 1)
        self.relu = nn.ReLU()
        
          
        self.res_conv = conv2d(in_channels, out_channels, spectral_normed = spectral_normed,
                               kernel_size = 1, stride = 1, padding = 0)
        self.conv_short = conv2d(in_channels, out_channels, kernel_size = 1, stride = 1, padding = 0,
                                spectral_normed = spectral_normed)  



    def shortcut(self, x):
        short = x
        short = self.conv_short(short)
        return self.avgpool_short(short)

        
        
    def residual(self, x):
        
        conv1 = self.relu(self.conv1(x))
        conv2 = self.conv2(conv1)
        resi = self.avgpool2(conv2)
        return resi
        
    def forward(self, x):

        return self.residual(x) + self.shortcut(x)
       
          
       
class SNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, spectral_normed = False, bias=True):
        super(SNLinear, self).__init__()

        self.linear = nn.Linear(in_channels, out_channels, bias=bias)
        torch.nn.init.normal_(self.linear.weight.data, std=0.02)
            
        if spectral_normed:
            self.linear = spectral_norm(self.linear)

    def forward(self, input):
        out = self.linear(input)
        return out
    
    
    
class QSNLinear(nn.Module):
    def __init__(self, in_channels, out_channels, spectral_normed = False, bias=True):
        super(QSNLinear, self).__init__()
        
        self.linear = QuaternionLinear(in_channels, out_channels, bias=bias)
        torch.nn.init.normal_(self.linear.r_weight.data, std=0.02)
        torch.nn.init.normal_(self.linear.i_weight.data, std=0.02)
        torch.nn.init.normal_(self.linear.j_weight.data, std=0.02)
        torch.nn.init.normal_(self.linear.k_weight.data, std=0.02)
            
        if spectral_normed:
            self.linear = Qspectral_norm(self.linear)

    def forward(self, input):
        out = self.linear(input)
        return out    
    
