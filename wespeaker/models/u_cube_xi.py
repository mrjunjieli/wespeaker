# Copyright (c) 2026 Junjie LI (mrjunjieli@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''Uncertainty-aware speaker models (U^3-xi).

These models reuse the standard WeSpeaker backbones and their pooling
layers, but the pooling layer is expected to be ``U_Cube_XI``, which
returns both the Gaussian posterior mean and its covariance. The
covariance is propagated analytically through the subsequent
BatchNorm / Linear layers, so each model returns

    (embed_a, var_diag, embed_b)

instead of the usual ``(embed_a, embed_b)``. Only ``forward`` is
overridden, therefore the parameter set -- and thus the state dict --
is identical to the corresponding standard WeSpeaker model.

Reference:
    U^3-xi: Pushing the Boundaries of Speaker Recognition by
    Incorporating Uncertainty. https://arxiv.org/abs/2601.15719
'''

import torch
import torch.nn.functional as F

import wespeaker.models.ecapa_tdnn as ecapa_tdnn
import wespeaker.models.redimnet as redimnet
import wespeaker.models.resnet as resnet


def _propagate_diag_var_bn(var_diag, bn):
    """Propagate a diagonal covariance through a BatchNorm1d layer.

    Args:
        var_diag: (B, D) diagonal covariance before the layer.
        bn: the BatchNorm1d the embedding is passed through.

    Returns:
        (B, D_out) diagonal covariance after the layer.
    """
    return var_diag / (bn.running_var + bn.eps) * bn.weight ** 2


def _propagate_diag_var_linear(var_diag, linear):
    """Propagate a diagonal covariance through a Linear layer.

    The full covariance becomes ``W diag(v) W^T``; only its diagonal is
    kept, which is what the downstream scoring uses.

    Args:
        var_diag: (B, D) diagonal covariance before the layer.
        linear: the Linear layer the embedding is passed through.

    Returns:
        (B, out_features) diagonal covariance after the layer.
    """
    var = torch.matmul(
        linear.weight,
        torch.matmul(torch.diag_embed(var_diag), linear.weight.T))
    return torch.diagonal(var, dim1=-2, dim2=-1)


class U_CUBE_XI_ECAPA_TDNN(ecapa_tdnn.ECAPA_TDNN):
    """ECAPA-TDNN that also returns the diagonal embedding covariance."""

    def forward(self, x):
        out, out4 = self._get_frame_level_feat(x)
        out = F.relu(out)

        stats, var_diag = self.pool(out)

        # mean branch
        out = self.bn(stats)
        out = self.linear(out)
        # variance branch
        var_diag = _propagate_diag_var_bn(var_diag, self.bn)
        var_diag = _propagate_diag_var_linear(var_diag, self.linear)

        if self.emb_bn:
            out = self.bn2(out)
            var_diag = _propagate_diag_var_bn(var_diag, self.bn2)

        return out4, var_diag, out


class _SegU3XiMixin:
    """U^3-xi forward for backbones built on ``seg_1``/``seg_2``.

    Shared by :class:`U_CUBE_XI_ResNet` and :class:`U_CUBE_XI_ReDimNet`,
    which expose the same segmentation head.
    """

    def forward(self, x):
        out = self._get_frame_level_feat(x)

        stats, var_diag = self.pool(out)

        # mean branch
        embed_a = self.seg_1(stats)
        # variance branch
        var_diag = _propagate_diag_var_linear(var_diag, self.seg_1)

        if self.two_emb_layer:
            out = self.seg_bn_1(embed_a)
            embed_b = self.seg_2(out)
            var_diag = _propagate_diag_var_bn(var_diag, self.seg_bn_1)
            var_diag = _propagate_diag_var_linear(var_diag, self.seg_2)
            return embed_a, var_diag, embed_b
        else:
            return torch.tensor(0.0), var_diag, embed_a


class U_CUBE_XI_ResNet(_SegU3XiMixin, resnet.ResNet):
    """ResNet that also returns the diagonal embedding covariance."""


class U_CUBE_XI_ReDimNet(_SegU3XiMixin, redimnet.ReDimNet):
    """ReDimNet that also returns the diagonal embedding covariance."""


def U_CUBE_XI_ECAPA_TDNN_c512(feat_dim,
                              embed_dim,
                              pooling_func='U_Cube_XI',
                              emb_bn=False):
    return U_CUBE_XI_ECAPA_TDNN(channels=512,
                                feat_dim=feat_dim,
                                embed_dim=embed_dim,
                                pooling_func=pooling_func,
                                emb_bn=emb_bn)


def U_CUBE_XI_ECAPA_TDNN_GLOB_c512(feat_dim,
                                   embed_dim,
                                   pooling_func='U_Cube_XI',
                                   emb_bn=False):
    return U_CUBE_XI_ECAPA_TDNN(channels=512,
                                feat_dim=feat_dim,
                                embed_dim=embed_dim,
                                pooling_func=pooling_func,
                                global_context_att=True,
                                emb_bn=emb_bn)


def U_CUBE_XI_ECAPA_TDNN_c1024(feat_dim,
                               embed_dim,
                               pooling_func='U_Cube_XI',
                               emb_bn=False):
    return U_CUBE_XI_ECAPA_TDNN(channels=1024,
                                feat_dim=feat_dim,
                                embed_dim=embed_dim,
                                pooling_func=pooling_func,
                                emb_bn=emb_bn)


def U_CUBE_XI_ECAPA_TDNN_GLOB_c1024(feat_dim,
                                    embed_dim,
                                    pooling_func='U_Cube_XI',
                                    emb_bn=False):
    return U_CUBE_XI_ECAPA_TDNN(channels=1024,
                                feat_dim=feat_dim,
                                embed_dim=embed_dim,
                                pooling_func=pooling_func,
                                global_context_att=True,
                                emb_bn=emb_bn)


def U_CUBE_XI_ResNet34(feat_dim,
                       embed_dim,
                       pooling_func='U_Cube_XI',
                       two_emb_layer=False):
    return U_CUBE_XI_ResNet(resnet.BasicBlock, [3, 4, 6, 3],
                            feat_dim=feat_dim,
                            embed_dim=embed_dim,
                            pooling_func=pooling_func,
                            two_emb_layer=two_emb_layer)


def U_CUBE_XI_ReDimNetB2(feat_dim=72,
                         embed_dim=192,
                         pooling_func='U_Cube_XI',
                         two_emb_layer=False):
    return U_CUBE_XI_ReDimNet(
        feat_dim=feat_dim,
        C=16,
        block_1d_type="conv+att",
        block_2d_type="convnext_like",
        stages_setup=[
            (1, 2, 1, [(3, 3)], 12),
            (2, 2, 1, [(3, 3)], 12),
            (1, 3, 1, [(3, 3)], 12),
            (2, 4, 1, [(3, 3)], 8),
            (1, 4, 1, [(3, 3)], 8),
            (2, 4, 1, [(3, 3)], 4),
        ],
        group_divisor=4,
        out_channels=None,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        global_context_att=True,
        two_emb_layer=two_emb_layer,
    )
