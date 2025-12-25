from collections import OrderedDict
from typing import Tuple, Union

import os
import hashlib
import warnings
import numpy as np
import urllib
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        # self.prompt_encoder = nn.Linear(d_model, d_model)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, M_type: str = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

        self.num_prompt = 8
        self.num_frame = 12

        self.modality_type = M_type
        # text prompt
        if self.modality_type == "text":
            self.learnable_prompt = nn.ParameterList(
                [nn.Parameter(torch.randn(1, self.num_prompt, width)) for _ in range(layers)])
        elif self.modality_type == "video":
            self.learnable_prompt_video = nn.ParameterList(
                [nn.Parameter(torch.randn(1, self.num_prompt, width)) for _ in range(layers)])
            self.learnable_prompt = nn.ParameterList(
                [nn.Parameter(torch.randn(self.num_frame, self.num_prompt, width)) for _ in range(layers)])
        else:
            TypeError("modality type must be text or video")

        if self.modality_type is not None:
            # Transformer
            # self.encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=8)
            # self.prompt_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
            # self.encoder_layer = nn.LSTM(width,width, bidirectional=True)
            # self.prompt_encoder = nn.Linear(width, width)
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=8)
            self.prompt_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def _prompt(self, x: torch.Tensor,
                generative_prompting: bool = False,
                modality_type: str = None,
                layer_index: int = None,
                prompt_encoder_type: str = None,
                frame_wise_for_video: bool = False,
                generation_without_video_or_text: bool = False,
                resblock: nn.Module = None
                ):
        '''
        x: video/text feature
            Torch.Tensor [bs*num_frame grids**2 width] for video
            Torch.Tensor [bs, ctx, width] for text
        generative_prompting: bool  是否使用生成式prompt
        modality_type: str  video or text when 使用生成式prompt
        layer_index: int    当前层数
        prompt_encoder_type: 确定prompt encoder用的是Transformer/Linear/LSTM/BiLSTM
        frame_wise_for_video: bool  仅在video时使用, 是否给每一帧都生成n个prompt
        generation_without_video_or_text: bool 是否不使用video/text feature生成prompt
          x = self._prompt(
                    x,
                    generative_prompting=True,
                    modality_type=self.modality_type,
                    layer_index=index,
                    prompt_encoder_type="linear",
                    frame_wise_for_video=False,
                    generation_without_video_or_text=False,
                    resblock=resblock
                )
        '''

        '''
        Masking for shallow and deep layer adaptively
        shallow for frame modeling
        deep for context modeling
        '''

        assert resblock is not None, "resblock must be not None"
        x = x.permute(1, 0, 2) #bs grid width

        if modality_type == 'video':
            expand_size = x.shape[0] // self.num_frame if frame_wise_for_video else x.shape[0]
            if generative_prompting:
                if generation_without_video_or_text:  # 生成器不输入video/text
                    if prompt_encoder_type == "transformer":
                        video_prompt = self.prompt_encoder(
                            self.learnable_prompt[layer_index].permute(1, 0, 2) if frame_wise_for_video else
                            self.learnable_prompt_video[layer_index].permute(1, 0, 2)).permute(1, 0, 2)
                    elif prompt_encoder_type == "linear":
                        video_prompt = self.prompt_encoder(
                            self.learnable_prompt[layer_index] if frame_wise_for_video else self.learnable_prompt_video[
                                layer_index])
                    elif prompt_encoder_type == "LSTM" or prompt_encoder_type == "BiLSTM":
                        pass  # TODO
                    else:
                        TypeError("prompt encoder type must be transformer, linear, LSTM or BiLSTM")
                    video_prompt = video_prompt.repeat(expand_size, 1, 1)
                else:
                    temp = torch.cat([x, self.learnable_prompt[layer_index].repeat(expand_size, 1,
                                                                                   1) if frame_wise_for_video else
                    self.learnable_prompt_video[layer_index].expand(expand_size, -1, -1)], dim=1)
                    if prompt_encoder_type == "transformer":
                        video_prompt = self.prompt_encoder(temp.permute(1, 0, 2)).permute(1, 0, 2)[:, -self.num_prompt:,
                                       :]
                    elif prompt_encoder_type == "linear":
                        video_prompt = self.prompt_encoder(temp)[:, -self.num_prompt:, :]
                    elif prompt_encoder_type == "LSTM" or prompt_encoder_type == "BiLSTM":
                        pass  # TODO
                    else:
                        TypeError("prompt encoder type must be transformer, linear, LSTM or BiLSTM")
                x = torch.cat(
                    [
                        x,
                        video_prompt

                    ], dim=1)
            else:
                x = torch.cat([
                    x,
                    self.learnable_prompt[layer_index].repeat(expand_size, 1, 1) if frame_wise_for_video else
                    self.learnable_prompt_video[layer_index].expand(expand_size, -1, -1)
                ], dim=1)

        elif modality_type == 'text':
            x = x[:, :-self.num_prompt, :]
            if generative_prompting:
                if generation_without_video_or_text:
                    if prompt_encoder_type == "transformer":
                        text_prompt = self.prompt_encoder(self.learnable_prompt[layer_index].permute(1, 0, 2)).permute(
                            1, 0, 2)
                    elif prompt_encoder_type == "linear":
                        text_prompt = self.prompt_encoder(self.learnable_prompt[layer_index])
                    else:
                        TypeError("prompt encoder type must be transformer or linear")
                    text_prompt = text_prompt.expand(x.shape[0], -1, -1)

                else:
                    temp = torch.cat([x, self.learnable_prompt[layer_index].expand(x.shape[0], -1, -1)], dim=1)
                    if prompt_encoder_type == "transformer":
                        text_prompt = self.prompt_encoder(temp.permute(1, 0, 2)).permute(1, 0, 2)[:, -self.num_prompt:,
                                      :]
                    elif prompt_encoder_type == "linear":
                        text_prompt = self.prompt_encoder(temp)[:, -self.num_prompt:, :]
                    else:
                        TypeError("prompt encoder type must be transformer or linear")
                x = torch.cat(
                    [
                        x,
                        text_prompt,

                    ], dim=1)
            else:
                x = torch.cat(
                    [
                        x,
                        self.learnable_prompt[layer_index].expand(x.shape[0], -1, -1)

                    ], dim=1)

        x = x.permute(1, 0, 2)
        x = resblock(x)
        if self.modality_type == 'video':
            x = x[:-self.num_prompt, :, :]

        return x

    def forward(self, x: torch.Tensor):

        layer_wise = True
        frame_wise = False
        if layer_wise:
            for index, resblock in enumerate(self.resblocks):
                x = self._prompt(
                    x,
                    generative_prompting=True,
                    modality_type=self.modality_type,
                    layer_index=index,
                    prompt_encoder_type="linear",
                    frame_wise_for_video=frame_wise,
                    generation_without_video_or_text=False,
                    resblock=resblock
                )
        else:
            for index, resblock in enumerate(self.resblocks):
                if index == 0:
                    x = x.permute(1, 0, 2)
                    expand_size = x.shape[0] // self.num_frame if frame_wise else x.shape[0]
                    # only input
                    if self.modality_type == 'video':

                        x = torch.cat([
                            x,
                            self.learnable_prompt[index].repeat(expand_size, 1, 1) if frame_wise else
                            self.learnable_prompt_video[index].expand(expand_size, -1, -1)
                        ], dim=1)

                    elif self.modality_type == 'text':
                        x = x[:, :-self.num_prompt, :]
                        x = torch.cat(
                            [
                                x,
                                self.learnable_prompt[index].expand(x.shape[0], -1, -1)

                            ], dim=1)
                    else:
                        TypeError("modality type must be video or text")

                    x = x.permute(1, 0, 2)
                    x = resblock(x)
                else:
                    x = resblock(x)

        return x
        # return self.resblocks(x) # initial version


class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, M_type="video")

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # 序列前端加入一个CLS token
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]

        # 嵌入位置信息
        x = x + self.positional_embedding.to(x.dtype)
        #
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND bs grid*2+1 width -》grid*2+1 bs width


        x = self.transformer(x) #


        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisualTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            M_type="text"
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # self.shared_encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_width, nhead=8)
        # self.shared_prompt_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        # self.index = 0

        # self.visual_embeddings = np.zeros(1000, 512)
        # self.text_embeddings = np.zeros(1000, 512)

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text, return_all_tokens=False):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        if return_all_tokens:
            return x @ self.text_projection

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image, self.shared_encoder_layer)
        text_features = self.encode_text(text, self.shared_encoder_layer)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # self.visual_embeddings[self.index, :] = image_features
        # self.text_embeddings[self.index, :] = text_features
        # self.index += 1

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()


_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
}


def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def available_models():
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load_clip(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit=True):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    """

    jit = False
    if name in _MODELS:
        model_path = _download(_MODELS[name])
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        graphs = [module.graph] if hasattr(module, "graph") else []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            graphs = [module.graph] if hasattr(module, "graph") else []
            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model







# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import pdb

import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),  # c_fc
            nn.GELU(),  # gelu
            nn.Linear(d_model, d_model)  # c_proj
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask
        # self.prompt_encoder = nn.Linear(d_model, d_model)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, M_type: str = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

        self.num_prompt = 8*2
        self.num_frame = 1

        self.modality_type = M_type
        # text prompt
        if self.modality_type == "text":
            self.learnable_prompt = nn.ParameterList(
                [nn.Parameter(torch.randn(1, self.num_prompt, width)) for _ in range(layers)])
        elif self.modality_type == "video":
            self.learnable_prompt_video = nn.ParameterList(
                [nn.Parameter(torch.randn(1, self.num_prompt, width)) for _ in range(layers)])
            self.learnable_prompt = nn.ParameterList(
                [nn.Parameter(torch.randn(self.num_frame, self.num_prompt, width)) for _ in range(layers)])
        else:
            TypeError("modality type must be text or video")

        if self.modality_type is not None:
            # Transformer
            # self.encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=8)
            # self.prompt_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)
            # self.encoder_layer = nn.LSTM(width,width, bidirectional=True)
            self.prompt_encoder = nn.Linear(width, width)
            # self.encoder_layer = nn.TransformerEncoderLayer(d_model=width, nhead=8)
            # self.prompt_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def _prompt(self, x: torch.Tensor,
                generative_prompting: bool = False,
                modality_type: str = None,
                layer_index: int = None,
                prompt_encoder_type: str = None,
                frame_wise_for_video: bool = False,
                generation_without_video_or_text: bool = False,
                resblock: nn.Module = None
                ):
        '''
        x: video/text feature
            Torch.Tensor [bs*num_frame grids**2 width] for video
            Torch.Tensor [bs, ctx, width] for text
        generative_prompting: bool  是否使用生成式prompt
        modality_type: str  video or text when 使用生成式prompt
        layer_index: int    当前层数
        prompt_encoder_type: 确定prompt encoder用的是Transformer/Linear/LSTM/BiLSTM
        frame_wise_for_video: bool  仅在video时使用, 是否给每一帧都生成n个prompt
        generation_without_video_or_text: bool 是否不使用video/text feature生成prompt
          x = self._prompt(
                    x,
                    generative_prompting=True,
                    modality_type=self.modality_type,
                    layer_index=index,
                    prompt_encoder_type="linear",
                    frame_wise_for_video=False,
                    generation_without_video_or_text=False,
                    resblock=resblock
                )
        '''

        '''
        Masking for shallow and deep layer adaptively
        shallow for frame modeling
        deep for context modeling
        '''

        assert resblock is not None, "resblock must be not None"
        x = x.permute(1, 0, 2) #bs grid width

        if modality_type == 'video':
            expand_size = x.shape[0] // self.num_frame if frame_wise_for_video else x.shape[0]
            if generative_prompting:
                if generation_without_video_or_text:  # 生成器不输入video/text
                    if prompt_encoder_type == "transformer":
                        video_prompt = self.prompt_encoder(
                            self.learnable_prompt[layer_index].permute(1, 0, 2) if frame_wise_for_video else
                            self.learnable_prompt_video[layer_index].permute(1, 0, 2)).permute(1, 0, 2)
                    elif prompt_encoder_type == "linear":
                        video_prompt = self.prompt_encoder(
                            self.learnable_prompt[layer_index] if frame_wise_for_video else self.learnable_prompt_video[
                                layer_index])
                    elif prompt_encoder_type == "LSTM" or prompt_encoder_type == "BiLSTM":
                        pass  # TODO
                    else:
                        TypeError("prompt encoder type must be transformer, linear, LSTM or BiLSTM")
                    video_prompt = video_prompt.repeat(expand_size, 1, 1)
                else:
                    temp = torch.cat([x, self.learnable_prompt[layer_index].repeat(expand_size, 1,
                                                                                   1) if frame_wise_for_video else
                    self.learnable_prompt_video[layer_index].expand(expand_size, -1, -1)], dim=1)
                    if prompt_encoder_type == "transformer":
                        video_prompt = self.prompt_encoder(temp.permute(1, 0, 2)).permute(1, 0, 2)[:, -self.num_prompt:,
                                       :]
                    elif prompt_encoder_type == "linear":
                        video_prompt = self.prompt_encoder(temp)[:, -self.num_prompt:, :]
                    elif prompt_encoder_type == "LSTM" or prompt_encoder_type == "BiLSTM":
                        pass  # TODO
                    else:
                        TypeError("prompt encoder type must be transformer, linear, LSTM or BiLSTM")
                x = torch.cat(
                    [
                        x,
                        video_prompt

                    ], dim=1)
            else:
                x = torch.cat([
                    x,
                    self.learnable_prompt[layer_index].repeat(expand_size, 1, 1) if frame_wise_for_video else
                    self.learnable_prompt_video[layer_index].expand(expand_size, -1, -1)
                ], dim=1)

        elif modality_type == 'text':
            # x = x[:, :-self.num_prompt, :]
            if generative_prompting:
                if generation_without_video_or_text:
                    if prompt_encoder_type == "transformer":
                        text_prompt = self.prompt_encoder(self.learnable_prompt[layer_index].permute(1, 0, 2)).permute(
                            1, 0, 2)
                    elif prompt_encoder_type == "linear":
                        text_prompt = self.prompt_encoder(self.learnable_prompt[layer_index])
                    else:
                        TypeError("prompt encoder type must be transformer or linear")
                    text_prompt = text_prompt.expand(x.shape[0], -1, -1)

                else:
                    temp = torch.cat([x, self.learnable_prompt[layer_index].expand(x.shape[0], -1, -1)], dim=1)
                    if prompt_encoder_type == "transformer":
                        text_prompt = self.prompt_encoder(temp.permute(1, 0, 2)).permute(1, 0, 2)[:, -self.num_prompt:,
                                      :]
                    elif prompt_encoder_type == "linear":
                        text_prompt = self.prompt_encoder(temp)[:, -self.num_prompt:, :]
                    else:
                        TypeError("prompt encoder type must be transformer or linear")
                x = torch.cat(
                    [
                        x,
                        text_prompt,

                    ], dim=1)
            else:
                x = torch.cat(
                    [
                        x,
                        self.learnable_prompt[layer_index].expand(x.shape[0], -1, -1)

                    ], dim=1)

        x = x.permute(1, 0, 2)
        x = resblock(x)
        if self.modality_type == 'video':
            x = x[:-self.num_prompt, :, :]
        else:
            x = x[:-self.num_prompt, :, :]

        return x

    def forward(self, x: torch.Tensor):

        layer_wise = True
        frame_wise = False
        if layer_wise:

            for index, resblock in enumerate(self.resblocks):
                x = self._prompt(
                    x,
                    generative_prompting=True,
                    modality_type=self.modality_type,
                    layer_index=index,
                    prompt_encoder_type="linear",
                    frame_wise_for_video=frame_wise,
                    generation_without_video_or_text=False,
                    resblock=resblock
                )

        return x



class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size


        #-------------------------------------------------------------------
        self.video_p = Transformer(4096, 4, 8, M_type="video")
        self.text_p = Transformer(2048, 4, 8, M_type="text")
        # -------------------------------------------------------------------



        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()


        input_embeds = input_embeds.permute(1,0,2)
        input_embeds = self.text_p(input_embeds)
        input_embeds = input_embeds.permute(1, 0, 2)


        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]


        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                return_dict=True).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])

        vit_embeds = vit_embeds.permute(1,0,2)
        vit_embeds = self.video_p(vit_embeds) #维度不变仍是 bs 256 4096
        vit_embeds = vit_embeds.permute(1, 0, 2)


        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (self.num_image_token * num_patches) + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            # num_image_token = self.num_image_token+16*4

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * (self.num_image_token * num_patches) + IMG_END_TOKEN

            query = query.replace('<image>', image_tokens, 1)


        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id

        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

            input_embeds = input_embeds.permute(1, 0, 2)
            input_embeds = self.text_p(input_embeds)
            input_embeds = input_embeds.permute(1, 0, 2)


            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)



            # padding_tensor = torch.zeros(B, 16 * 4).to(input_ids.device)
            # # padding_tensor2 = torch.full((B, 16 * 4), -100).to(labels.device)
            # input_ids = torch.cat([input_ids, padding_tensor], dim=1)
            # attention_mask = torch.cat([attention_mask,padding_tensor],dim=1)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
