import torch
import torch.nn as nn
import math
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.fc1 = nn.Linear(in_features, hidden_features,bias=False)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        # self.fc2 = nn.Linear(hidden_features, out_features,bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CycleFC(nn.Module):
    """
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)


class CycleMLP(nn.Module):
    def __init__(self, stride,dim, qkv_bias=False):#, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)

        self.sfc_h = CycleFC(dim, dim, (1,stride), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (stride, 1), 1, 0)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        # self.proj = nn.Linear(dim, dim,bias=False)
        # self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        w = self.sfc_w(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        # x = self.proj_drop(x)
        return x

class CycleBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=2., qkv_bias=False, 
                  act_layer=nn.GELU, norm_layer=nn.LayerNorm,  mlp_fn=CycleMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        #Spatial Project
        self.attn = mlp_fn(11,dim,qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        #Spectral Project
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x=x+self.attn(self.norm1(x))
        x=x+self.mlp(self.norm2(x))
        return x

class SSMLP(nn.Module):
    def __init__(
            self,
            dim,
            num_blocks
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.input=nn.Linear(dim,dim*2)
        self.out=nn.Linear(dim*2,dim)
        for _ in range(num_blocks):
            self.blocks.append(CycleBlock(dim*2))
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        
        x = x.permute(0, 2, 3, 1)
        x=self.input(x)
        for (mlp)in self.blocks:
            x = mlp(x)
        x=self.out(x)
        x=x.permute(0,3,1,2)
        
        return x
 
class MDDP(nn.Module):
    def __init__(self, dim=28, depth=2, num_blocks=[1,1,1]):
        super(MDDP, self).__init__()
        self.dim = dim
        self. depth =  depth

        # Input projection
        self.embedding = nn.Conv2d(28, self.dim,1,1,0,bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_depth = dim
        for i in range( depth):
            self.encoder_layers.append(nn.ModuleList([
                SSMLP(
                    dim=dim_depth, num_blocks=num_blocks[i]),
                nn.Conv2d(dim_depth, dim_depth * 2, 3, 2, 1, bias=False),
                nn.Conv2d(dim_depth, dim_depth , 1, 1, 0, bias=False),
                nn.Conv2d(dim_depth, dim_depth , 1, 1, 0, bias=False),
            ]))
            dim_depth *= 2

        # Bottleneck
        self.bottleneck = SSMLP(
            dim=dim_depth, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(depth):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_depth, dim_depth // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_depth, dim_depth // 2,1,1,0,  bias=False),
                SSMLP(
                    dim=dim_depth // 2, num_blocks=num_blocks[depth - 1 - i]
                  ),
            ]))
            dim_depth //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, 28, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(self.dim, 28, 1, 1, 0, bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        ##SMAM
        self.fusion=nn.Conv2d(self.dim*2,dim,1,1,0,bias=False)
        self.att=CycleMLP(7,self.dim)

       
        

    def forward(self, x,encs,decs,att):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        # if mask == None:
        #     mask = torch.zeros((1,28,256,310)).cuda()

        # Embedding
        fea = self.lrelu(self.embedding(x))
        if att is not None:
            fea=self.fusion(torch.cat([fea,att],dim=1))
        # Encoder
        fea_encoder = []
        # masks = []
        for i,(SSMLP, FeaDownSample,mf_enc,mf_dec) in enumerate(self.encoder_layers):
            if encs is not None:
                fea=SSMLP(fea)+mf_enc(encs[i])+mf_dec(decs[self.depth-1-i]) #Information Delivery Pipline(IDP) mf:multi-stage fusion
            else:
                fea = SSMLP(fea)
            fea_encoder.append(fea)#b,h,w,c
            fea = FeaDownSample(fea)
        

        # Bottleneck
        fea = self.bottleneck(fea)
        fea_decoder=[]
        # Decoder
        for i, (FeaUpSample, Fusion, SSMLP) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(torch.cat([fea, fea_encoder[self.depth-1-i]], dim=1))
          
            fea = SSMLP(fea)
            fea_decoder.append(fea)

        # Mapping
        out = self.mapping(fea) + x
        # Staged MLP Attention Module(SMAM)
        att=self.att(out.permute(0,2,3,1)).permute(0,3,1,2)
        att=att*self.conv2(fea)+fea

        return out,fea_encoder,fea_decoder,att 




def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x
def shift_back_3d(inputs,step=2): #b,c,256,310 -> b,c,256,256
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    inputs=inputs[:,:,:,0:256]
    return inputs
def shift(inputs, step=2): #b,c,256,256->b,c,256,310
    [bs, nC, row, col] = inputs.shape
    output = torch.zeros(bs, nC, row, col + (nC - 1) * step).cuda().float()
    # output = torch.zeros(bs, nC, row, col + (nC - 1) * step)
    for i in range(nC):
        output[:, i, :, step * i:step * i + col] = inputs[:, i, :, :]
    return output
# def generate_masks(mask_path, batch_size):
#     mask = sio.loadmat(mask_path + '/mask.mat')
#     mask = mask['mask']
#     mask3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
#     mask3d = np.transpose(mask3d, [2, 0, 1])
#     mask3d = torch.from_numpy(mask3d)
#     [nC, H, W] = mask3d.shape
#     mask3d_batch = mask3d.expand([batch_size, nC, H, W]).cuda().float()
#     return mask3d_batch


class MFMLP(nn.Module):
    def __init__(self,stage=2,dim=28,depth=2,num_blocks=[1,1,1]):
        super().__init__()
        self.denoiser=MDDP(dim=dim,depth=depth,num_blocks=num_blocks)
        self.produce_z0=nn.Conv2d(29,28,1,1,0,bias=False)
        self.stage=stage
        self.miu=nn.Parameter(torch.Tensor([0.8 for _ in range(self.stage)]),requires_grad=True)
        self.act=nn.Softplus()
    def forward(self,y,input_mask=None):
        if input_mask==None:
            Phi=torch.randn(1,28,256,310).cuda()
            Phi_s=torch.randn(1,256,310).cuda()
        else:
            Phi,Phi_s=input_mask
        z=self.produce_z0(torch.cat([y.unsqueeze(1),Phi],1)) #b,28,256,310
        encs=None
        decs=None
        att=None
        for i in range(self.stage):
            x=z+At(torch.div((y-A(z,Phi)),(self.act(self.miu[i])+Phi_s)),Phi)#b,28,256,310
            x=shift_back_3d(x) #b,28,256,256
            out,encs,decs,att=self.denoiser(x,encs,decs,att)#x:[b,28,256,256]
            z=shift(out)
        return out

        



