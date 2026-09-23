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
import wespeaker.models.redimnet2 as redimnet2
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
            # NOTE: this mirrors the original U^3-xi implementation and does
            # not apply the ReLU that the base class applies before seg_bn_1.
            # Kept as-is for reproducibility; every released U^3-xi config
            # uses two_emb_layer: False, so this branch is not exercised.
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


class U_CUBE_XI_ReDimNet2Wrap(redimnet2.ReDimNet2Wrap):
    """ReDimNet2 that also returns the diagonal embedding covariance.

    ``return_all_outputs`` is not supported in the U^3-xi variant; the
    intermediate 1-D outputs are silently dropped so the return signature
    stays (embed_a, var_diag, embed_b).
    """

    def forward(self, x):
        if self.pad_right_samples is not None:
            x = torch.nn.functional.pad(
                x, (0, self.pad_right_samples), mode='constant', value=None)
        if self.spec is not None and self.spec != 'fbank':
            x = self.spec(x)
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if self.return_all_outputs:
            out, _ = self.backbone(x)
        else:
            out = self.backbone(x)
        if out.ndim == 4:
            bs, C, F, T = out.size()
            out = out.reshape(bs, C * F, T)
        if self.before_pool_offset is not None:
            out = out[:, :, self.before_pool_offset:]
        stats, var_diag = self.pool(out)
        out = self.bn(stats)
        var_diag = _propagate_diag_var_bn(var_diag, self.bn)
        out = self.linear(out)
        var_diag = _propagate_diag_var_linear(var_diag, self.linear)
        if self.bn2 is not None:
            out = self.bn2(out)
            var_diag = _propagate_diag_var_bn(var_diag, self.bn2)
        return torch.tensor(0.0), var_diag, out


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


def U_CUBE_XI_ReDimNet2Custom(feat_dim,
                               embed_dim,
                               pooling_func='U_Cube_XI',
                               **kwargs):
    return U_CUBE_XI_ReDimNet2Wrap(
        feat_dim=feat_dim,
        embed_dim=embed_dim,
        pooling_func=pooling_func,
        **kwargs,
    )


def U_CUBE_XI_ReDimNet2B0(feat_dim=72,
                           embed_dim=192,
                           pooling_func='U_Cube_XI',
                           **kwargs):
    stages_setup = [
        [[1, 1], 2, 2, [[3, 3]], 36],
        [[2, 1], 3, 1, [[3, 3]], 36],
        [[1, 2], 4, 1, [[3, 3]], 36],
        [[2, 1], 5, 1, [[3, 3]], 36],
        [[1, 2], 4, 1, [[3, 3]], 18],
        [[2, 1], 3, 1, [[3, 3]], 18],
    ]
    return U_CUBE_XI_ReDimNet2Wrap(
        C=12, out_channels=64,
        feat_dim=feat_dim, embed_dim=embed_dim,
        pooling_func=pooling_func,
        stages_setup=stages_setup,
        **kwargs,
    )


def U_CUBE_XI_ReDimNet2B1(feat_dim=72,
                           embed_dim=192,
                           pooling_func='U_Cube_XI',
                           **kwargs):
    stages_setup = [
        [[1, 1], 2, 2, [[3, 3]], 32],
        [[2, 1], 3, 1, [[3, 3]], 32],
        [[1, 2], 4, 1, [[3, 3]], 32],
        [[2, 1], 5, 1, [[3, 3]], 32],
        [[1, 2], 4, 1, [[3, 3]], 16],
        [[2, 1], 3, 1, [[3, 3]], 16],
    ]
    return U_CUBE_XI_ReDimNet2Wrap(
        C=16, out_channels=64,
        feat_dim=feat_dim, embed_dim=embed_dim,
        pooling_func=pooling_func,
        stages_setup=stages_setup,
        **kwargs,
    )


def U_CUBE_XI_ReDimNet2B2(feat_dim=72,
                           embed_dim=192,
                           pooling_func='U_Cube_XI',
                           **kwargs):
    stages_setup = [
        [[1, 1], 2, 2, [[3, 5]], 40],
        [[2, 1], 3, 1, [[3, 5]], 30],
        [[1, 2], 4, 1, [[3, 5]], 30],
        [[3, 1], 5, 1, [[3, 5]], 20],
        [[1, 2], 4, 1, [[3, 7]], 20],
        [[2, 1], 3, 1, [[3, 7]], 10],
    ]
    return U_CUBE_XI_ReDimNet2Wrap(
        C=20, out_channels=64,
        feat_dim=feat_dim, embed_dim=embed_dim,
        pooling_func=pooling_func,
        stages_setup=stages_setup,
        **kwargs,
    )


def U_CUBE_XI_ReDimNet2B3(feat_dim=72,
                           embed_dim=192,
                           pooling_func='U_Cube_XI',
                           **kwargs):
    stages_setup = [
        [[1, 1], 2, 2, [[3, 3]], 36],
        [[2, 1], 3, 1, [[3, 3]], 36],
        [[1, 2], 4, 1, [[3, 3]], 36],
        [[2, 1], 5, 1, [[3, 3]], 36],
        [[1, 2], 4, 1, [[3, 3]], 18],
        [[2, 1], 3, 1, [[3, 3]], 18],
    ]
    return U_CUBE_XI_ReDimNet2Wrap(
        C=24, out_channels=64,
        feat_dim=feat_dim, embed_dim=embed_dim,
        pooling_func=pooling_func,
        stages_setup=stages_setup,
        **kwargs,
    )


def U_CUBE_XI_ReDimNet2B4(feat_dim=72,
                           embed_dim=192,
                           pooling_func='U_Cube_XI',
                           **kwargs):
    stages_setup = [
        [[1, 1], 2, 4, [[3, 3]], 24],
        [[2, 1], 3, 3, [[3, 3]], 24],
        [[1, 2], 4, 2, [[3, 3]], 24],
        [[2, 1], 5, 1, [[3, 3]], 24],
        [[1, 2], 4, 1, [[3, 3]], 24],
        [[2, 1], 3, 1, [[3, 3]], 24],
    ]
    return U_CUBE_XI_ReDimNet2Wrap(
        C=32,
        feat_dim=feat_dim, embed_dim=embed_dim,
        pooling_func=pooling_func,
        stages_setup=stages_setup,
        **kwargs,
    )


def U_CUBE_XI_ReDimNet2B5(feat_dim=72,
                           embed_dim=192,
                           pooling_func='U_Cube_XI',
                           **kwargs):
    stages_setup = [
        [[1, 1], 2, 4, [[3, 3]], 48],
        [[2, 1], 3, 3, [[3, 3]], 48],
        [[1, 2], 4, 2, [[3, 3]], 48],
        [[2, 1], 5, 1, [[3, 3]], 48],
        [[1, 2], 4, 1, [[3, 3]], 32],
        [[2, 1], 3, 1, [[3, 3]], 32],
    ]
    return U_CUBE_XI_ReDimNet2Wrap(
        C=48, out_channels=256,
        feat_dim=feat_dim, embed_dim=embed_dim,
        pooling_func=pooling_func,
        stages_setup=stages_setup,
        **kwargs,
    )


def U_CUBE_XI_ReDimNet2B6(feat_dim=72,
                           embed_dim=192,
                           pooling_func='U_Cube_XI',
                           **kwargs):
    stages_setup = [
        [[1, 1], 3, 3, [[3, 3]], 64],
        [[2, 1], 4, 2, [[3, 3]], 64],
        [[1, 2], 5, 2, [[3, 3]], 48],
        [[2, 1], 5, 1, [[3, 3]], 48],
        [[1, 2], 4, 0.75, [[3, 3]], 32],
        [[2, 1], 3, 0.5, [[3, 3]], 24],
    ]
    return U_CUBE_XI_ReDimNet2Wrap(
        C=64, out_channels=224, return_2d_output=True,
        feat_dim=feat_dim, embed_dim=embed_dim,
        pooling_func=pooling_func,
        stages_setup=stages_setup,
        **kwargs,
    )
