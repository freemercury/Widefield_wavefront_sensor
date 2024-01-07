import torch
import torch.nn as nn
import torch.nn.functional as F
from .raclstm import ResAttenConvLSTM as racLSTM
from .rclstm import ResConvLSTM as rcLSTM
import torchvision.transforms as transforms
from .encoding import *
from .vit import ViT1, ViT2
import segmentation_models_pytorch as smp


class TurbsPred(nn.Module):
    def __init__(self, device=None, **kwargs):
        """
        backbone: None, rc, rac, vit1, vit2
        head: hw, chw, tchw, h+w, resize, centercrop, pool
        pos_enc: None, hash_sum, hash_cat, hash_mtp, freq_sum, freq_cat, freq_mtp   
        pad_type: None, zero, replicate, reflect, resize
        ext_size: (32,64)
        extractor: None, "unet", "unet++", "manet", "linknet", "fpn"
        encoder_name: "resnet18", ...
        encoder_weights: None, "imagenet"
        decoder_attention_type: None, "scse"
        """
        super().__init__()

        self.device = device
        self.kwargs = kwargs

        self.backbone_type = self.kwargs["model"]["backbone"]
        self.head_type = self.kwargs["model"]["head"]
        self.pos_enc = self.kwargs["model"]["pos_enc"]
        self.pad_type = self.kwargs["model"]["pad_type"]
        self.ext_size = self.kwargs["model"]["ext_size"]
        self.extractor_type = self.kwargs["model"]["extractor"]
        self.encoder_name = self.kwargs["model"]["encoder_name"]
        self.encoder_weights = self.kwargs["model"]["encoder_weights"]
        self.decoder_attention_type = self.kwargs["model"]["decoder_attention_type"]
        self.data_type = self.kwargs["dataset"]["data_type"]

        self.input_patch = self.kwargs["dataset"]["input_patch"]
        self.input_size = (self.input_patch[1] - self.input_patch[0], self.input_patch[3] - self.input_patch[2])
        self.output_patch = kwargs["dataset"]["target_patch"]
        self.output_size = (self.output_patch[1] - self.output_patch[0], self.output_patch[3] - self.output_patch[2])
        self.t_series = kwargs["dataset"]["t_series"]

        # pad
        if self.pad_type == "zero":
            self.pad = nn.ZeroPad2d(((self.ext_size[1]-self.input_size[1])//2 if self.input_size[1]%2==0 else (self.ext_size[1]-self.input_size[1])//2+1, (self.ext_size[1]-self.input_size[1])//2, 
                                     (self.ext_size[0]-self.input_size[0])//2 if self.input_size[0]%2==0 else (self.ext_size[0]-self.input_size[0])//2+1, (self.ext_size[0]-self.input_size[0])//2))
        elif self.pad_type == "replicate":
            self.pad = nn.ReplicationPad2d(((self.ext_size[1]-self.input_size[1])//2 if self.input_size[1]%2==0 else (self.ext_size[1]-self.input_size[1])//2+1, (self.ext_size[1]-self.input_size[1])//2, 
                                     (self.ext_size[0]-self.input_size[0])//2 if self.input_size[0]%2==0 else (self.ext_size[0]-self.input_size[0])//2+1, (self.ext_size[0]-self.input_size[0])//2))
        elif self.pad_type == "reflect":
            self.pad = nn.ReflectionPad2d(((self.ext_size[1]-self.input_size[1])//2 if self.input_size[1]%2==0 else (self.ext_size[1]-self.input_size[1])//2+1, (self.ext_size[1]-self.input_size[1])//2, 
                                     (self.ext_size[0]-self.input_size[0])//2 if self.input_size[0]%2==0 else (self.ext_size[0]-self.input_size[0])//2+1, (self.ext_size[0]-self.input_size[0])//2))
        elif self.pad_type == "resize":
            self.pad = transforms.Resize((self.ext_size[0], self.ext_size[1]))
        elif self.pad_type == None:
            self.pad = nn.Identity()
        self.in_dim = (self.ext_size[0], self.ext_size[1]) if self.pad_type is not None else self.input_size

        # encoding
        if self.pos_enc in ["hash_sum", "hash_cat", "hash_mtp"]:
            self.enc = MultiResHashGrid()
            self.pos = torch.stack(torch.meshgrid(torch.linspace(0,1,2*self.in_dim[0]+1)[1::2], torch.linspace(0,1,2*self.in_dim[1]+1)[1::2]), dim=-1).to(device)
        if self.pos_enc in ["freq_sum", "freq_cat", "freq_mtp"]:
            self.enc = Frequency()
            self.pos = torch.stack(torch.meshgrid(torch.linspace(0,1,self.in_dim[0]), torch.linspace(0,1,self.in_dim[1])), dim=-1).to(device)

        # backbone
        if self.backbone_type == "rc":
            self.backbone = rcLSTM(n_dim=self.kwargs["model"]["n_dim"], 
                                   kernel_size=self.kwargs["model"]["kernel_size"], 
                                   dropout=self.kwargs["model"]["dropout"])
            self.in_c = self.kwargs["model"]["n_dim"][0][0]
        elif self.backbone_type == "rac":
            self.backbone = racLSTM(n_dim=self.kwargs["model"]["n_dim"],
                                    kernel_size=self.kwargs["model"]["kernel_size"],
                                    emb_dim=self.kwargs["model"]["emb_dim"],
                                    emb_kernel=self.kwargs["model"]["emb_kernel"],
                                    dropout=self.kwargs["model"]["dropout"])
            self.in_c = self.kwargs["model"]["n_dim"][0][0]
        elif self.backbone_type == "vit1":
            self.backbone = ViT1(num_layers=self.kwargs["model"]["num_layers"],
                                 num_heads=self.kwargs["model"]["num_heads"],
                                 input_size=self.in_dim,
                                 t_series=self.kwargs["dataset"]["t_series"],
                                 base_dim=self.kwargs["model"]["base_dim"],
                                 hidden_dim=self.kwargs["model"]["hidden_dim"],
                                 mlp_dim=self.kwargs["model"]["mlp_dim"],
                                 dropout=self.kwargs["model"]["dropout"])
            self.in_c = self.kwargs["model"]["base_dim"]
        elif self.backbone_type == "vit2":
            self.backbone = ViT2(base_dim=self.kwargs["model"]["base_dim"],
                                 t_series=self.kwargs["dataset"]["t_series"],
                                 spa_hidden_dim=self.kwargs["model"]["spa_hidden_dim"],
                                 spa_mlp_dim=self.kwargs["model"]["spa_mlp_dim"],
                                 spa_dropout=self.kwargs["model"]["spa_dropout"],
                                 spa_num_heads=self.kwargs["model"]["spa_num_heads"],
                                 spa_num_layers=self.kwargs["model"]["spa_num_layers"],
                                 temp_hidden_dim=self.kwargs["model"]["temp_hidden_dim"],
                                 temp_mlp_dim=self.kwargs["model"]["temp_mlp_dim"],
                                 temp_dropout=self.kwargs["model"]["temp_dropout"],
                                 temp_num_heads=self.kwargs["model"]["temp_num_heads"],
                                 temp_num_layers=self.kwargs["model"]["temp_num_layers"],
                                 input_size=self.in_dim)
            self.in_c = self.kwargs["model"]["base_dim"]
        elif self.backbone_type is None:
            self.backbone = nn.Identity()
            model_dict = self.kwargs["model"]
            self.in_c = model_dict["n_dim"][0][0] if "n_dim" in model_dict.keys() else model_dict["base_dim"]

        # extractor
        if self.extractor_type == "unet":
            self.ext = smp.Unet(encoder_name=self.encoder_name, encoder_weights=self.encoder_weights, decoder_attention_type=self.decoder_attention_type, 
                                in_channels=self.in_c, classes=self.in_c)
        elif self.extractor_type == "unet++":
            self.ext = smp.UnetPlusPlus(encoder_name=self.encoder_name, encoder_weights=self.encoder_weights, decoder_attention_type=self.decoder_attention_type, 
                                        in_channels=self.in_c, classes=self.in_c)
        elif self.extractor_type == "manet":
            self.ext = smp.MAnet(encoder_name=self.encoder_name, encoder_weights=self.encoder_weights,
                                 in_channels=self.in_c, classes=self.in_c)
        elif self.extractor_type == "linknet":
            self.ext = smp.Linknet(encoder_name=self.encoder_name, encoder_weights=self.encoder_weights,
                                 in_channels=self.in_c, classes=self.in_c)
        elif self.extractor_type == "fpn":
            self.ext = smp.FPN(encoder_name=self.encoder_name, encoder_weights=self.encoder_weights,
                                 in_channels=self.in_c, classes=self.in_c)
        elif self.extractor_type is None:
            self.ext = nn.Identity()

        # head
        if self.head_type == "hw":
            self.head = nn.Linear(self.in_dim[0] * self.in_dim[1], self.output_size[0] * self.output_size[1])
        elif self.head_type == "h+w":
            self.head1 = nn.Linear(self.in_dim[0], self.output_size[0])
            self.head2 = nn.Linear(self.in_dim[1], self.output_size[1])
        elif self.head_type == "chw":
            self.head = nn.Linear(self.in_c * self.in_dim[0] * self.in_dim[1], self.in_c * self.output_size[0] * self.output_size[1])
        elif self.head_type == "tchw":
            self.head = nn.Linear(self.t_series * self.in_c * self.in_dim[0] * self.in_dim[1], self.t_series * self.in_c * self.output_size[0] * self.output_size[1])
        elif self.head_type == "centercrop":
            self.head = transforms.CenterCrop((self.output_size[0], self.output_size[1]))
        elif self.head_type == "resize":
            self.head = transforms.Resize((self.output_size[0], self.output_size[1]))
        elif self.head_type == "pool":
            self.head = nn.AdaptiveAvgPool2d((self.output_size[0], self.output_size[1]))

    def forward(self, input_data):
        """
        input_data shape (b, t, c, h, w)
        output shape (b, t, c, h', w')
        """
        
        b, t, c, h, w = input_data.shape
        
        # padding
        input_data = torch.stack([self.pad(input_data[:,i,:,:,:]) for i in range(t)], dim=1)

        # encoding
        if self.data_type == "zernike":
            if self.pos_enc in ["hash_sum", "freq_sum"]:
                input_data = self.enc(self.pos).permute(2,0,1).unsqueeze(0).unsqueeze(0).broadcast_to(input_data.shape) + input_data
            elif self.pos_enc in ["hash_cat", "freq_cat"]:
                input_data = torch.cat([self.enc(self.pos).permute(2,0,1).unsqueeze(0).unsqueeze(0).broadcast_to(input_data.shape), input_data], dim=2)
            elif self.pos_enc in ["hash_mtp", "freq_mtp"]:
                input_data = self.enc(self.pos).permute(2,0,1).unsqueeze(0).unsqueeze(0).broadcast_to(input_data.shape) * input_data

        # extractor
        input_data = torch.stack([self.ext(input_data[:,i,:,:,:]) for i in range(t)], dim=1)

        # backbone
        output = self.backbone(input_data)
        b, t, c, h, w = output.shape

        # head
        if self.head_type == "hw":
            output = self.head(output.reshape(b, t, c, h*w)).reshape(b, t, c, self.output_size[0], self.output_size[1])
        elif self.head_type == "h+w":
            output = self.head1(self.head2(output).permute(0,1,2,4,3)).permute(0,1,2,4,3)
        elif self.head_type == "chw":
            output = self.head(output.reshape(b, t, c*h*w)).reshape(b, t, c, self.output_size[0], self.output_size[1])
        elif self.head_type == "tchw":
            output = self.head(output.reshape(b, t*c*h*w)).reshape(b, t, c, self.output_size[0], self.output_size[1])
        elif self.head_type in ["centercrop", "resize", "pool"]:
            output = torch.stack([self.head(output[:,i,:,:,:]) for i in range(t)], dim=1)

        return output



