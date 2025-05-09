# 版权 (c) Meta Platforms, Inc. 和附属公司。
# 保留所有权利。

# 此源代码根据 LICENSE 文件中找到的许可协议进行许可
# LICENSE 文件位于本源代码树的根目录。

# 导入 PyTorch 所需的库和模块
import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

# 导入自定义层归一化模块
from .common import LayerNorm2d

# 定义 MaskDecoder 类，这是一个神经网络模块
class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,  # transformer 的通道维度
        transformer: nn.Module,  # 用于预测掩码的 transformer 模型
        num_multimask_outputs: int = 3,  # 需要预测的掩码数量
        activation: Type[nn.Module] = nn.GELU,  # 上采样掩码时使用的激活函数类型
        iou_head_depth: int = 3,  # 用于预测掩码质量的 MLP 的深度
        iou_head_hidden_dim: int = 256,  # 用于预测掩码质量的 MLP 的隐藏维度
    ) -> None:
        """
        初始化 MaskDecoder 类的实例。

        参数：
          transformer_dim (int): transformer 的通道维度
          transformer (nn.Module): 用于预测掩码的 transformer 模型
          num_multimask_outputs (int): 需要预测的掩码数量
          activation (nn.Module): 上采样掩码时使用的激活函数类型
          iou_head_depth (int): 用于预测掩码质量的 MLP 的深度
          iou_head_hidden_dim (int): 用于预测掩码质量的 MLP 的隐藏维度
        """
        super().__init__()  # 调用父类的初始化方法
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        # 定义嵌入层，用于掩码和 IoU（交并比）的 token
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 定义用于上采样掩码的序列层
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # 定义用于输出掩码的多层感知机（MLP）序列
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        # 定义用于 IoU 预测的 MLP
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,  # 来自图像编码器的嵌入
        image_pe: torch.Tensor,  # 具有与图像嵌入相同形状的位置编码
        sparse_prompt_embeddings: torch.Tensor,  # 点和框的嵌入
        dense_prompt_embeddings: torch.Tensor,  # 掩码输入的嵌入
        multimask_output: bool,  # 是否返回多个掩码
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        给定图像和提示嵌入，预测掩码。

        参数：
          image_embeddings (torch.Tensor): 来自图像编码器的嵌入
          image_pe (torch.Tensor): 具有与图像嵌入相同形状的位置编码
          sparse_prompt_embeddings (torch.Tensor): 点和框的嵌入
          dense_prompt_embeddings (torch.Tensor): 掩码输入的嵌入
          multimask_output (bool): 是否返回多个掩码。

        返回：
          torch.Tensor: 批量预测的掩码
          torch.Tensor: 批量掩码质量预测
        """
        # 预测掩码
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # 选择正确的掩码输出
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # 准备输出
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测掩码。具体细节参见 'forward' 方法。"""
        # 拼接输出 token
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 在批量方向上扩展每张图像的数据，以使其每个掩码都有数据
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # 运行 transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # 上采样掩码嵌入并使用掩码 token 预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # 生成掩码质量预测
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred

# 轻微改编自
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # 输入维度
        hidden_dim: int,  # 隐藏层维度
        output_dim: int,  # 输出维度
        num_layers: int,  # 层数
        sigmoid_output: bool = False,  # 是否使用 sigmoid 激活函数
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
