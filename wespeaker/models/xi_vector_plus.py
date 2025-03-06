
import torch
import torch.nn as nn
import torch.nn.functional as F
''' Res2Conv1d + BatchNorm1d + ReLU
'''


class Res2Conv1dReluBn(nn.Module):
    """
    in_channels == out_channels == channels
    """

    def __init__(self,
                 channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(
                nn.Conv1d(self.width,
                          self.width,
                          kernel_size,
                          stride,
                          padding,
                          dilation,
                          bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        sp = spx[0]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            # Order: conv -> relu -> bn
            if i >= 1:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = bn(F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


''' Conv1d + BatchNorm1d + ReLU
'''


class Conv1dReluBn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding,
                              dilation,
                              bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


''' The SE connection of 1D case.
'''


class SE_Connect(nn.Module):

    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


''' SE-Res2Block of the ECAPA-TDNN architecture.
'''


class SE_Res2Block(nn.Module):

    def __init__(self, channels, kernel_size, stride, padding, dilation,
                 scale):
        super().__init__()
        self.se_res2block = nn.Sequential(
            Conv1dReluBn(channels,
                         channels,
                         kernel_size=1,
                         stride=1,
                         padding=0),
            Res2Conv1dReluBn(channels,
                             kernel_size,
                             stride,
                             padding,
                             dilation,
                             scale=scale),
            Conv1dReluBn(channels,
                         channels,
                         kernel_size=1,
                         stride=1,
                         padding=0), SE_Connect(channels))

    def forward(self, x):
        return x + self.se_res2block(x)


class XI(torch.nn.Module):
    def __init__(self, in_dim, hidden_size=256, stddev=False,
                 train_mean=True, train_prec=True, **kwargs):
        super(XI, self).__init__()
        self.input_dim = in_dim
        self.stddev = stddev
        if self.stddev:
            self.output_dim = 2 * self.input_dim
        else:
            self.output_dim = self.input_dim
        self.prior_mean = torch.nn.Parameter(torch.zeros(1, self.input_dim),
                                             requires_grad=train_mean)
        self.prior_logprec = torch.nn.Parameter(torch.zeros(1, self.input_dim),
                                                requires_grad=train_prec)
        self.softmax = torch.nn.Softmax(dim=2)

        # Log-precision estimator
        self.lin1_relu_bn = nn.Sequential(
            nn.Conv1d(self.input_dim, hidden_size,
                      kernel_size=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_size))
        self.lin2 = nn.Conv1d(hidden_size, self.input_dim, kernel_size=1,
                              stride=1, bias=True)
        self.softplus2 = torch.nn.Softplus(beta=1, threshold=20)

    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch),
        including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim
        feat = inputs
        # Log-precision estimator
        # frame precision estimate
        logprec = self.softplus2(self.lin2(self.lin1_relu_bn(feat)))

        # Square and take log before softmax
        logprec = 2.0 * torch.log(logprec)
        # Gaussian Posterior Inference
        # Option 1: a_o (prior_mean-phi) included in variance
        weight_attn = self.softmax(
            torch.cat(
                (logprec,
                 self.prior_logprec.repeat(
                     logprec.shape[0], 1).unsqueeze(dim=2)), 2))
        # Posterior precision
        Ls = torch.sum(torch.exp(torch.cat(
            (logprec, self.prior_logprec.repeat(
                logprec.shape[0], 1).unsqueeze(dim=2)), 2)), dim=2)
        # Posterior mean
        phi = torch.sum(torch.cat(
            (feat, self.prior_mean.repeat(
                feat.shape[0], 1).unsqueeze(dim=2)), 2) * weight_attn, dim=2)

        # if self.stddev:
        #     sigma2 = torch.sum(torch.cat((
        #         feat, self.prior_mean.repeat(
        #             feat.shape[0], 1).unsqueeze(dim=2)), 2).pow(2) * weight_attn, dim=2)
        #     sigma = torch.sqrt(torch.clamp(sigma2 - phi ** 2, min=1.0e-12))
        #     return torch.cat((phi, sigma), dim=1).unsqueeze(dim=2)
        # else:
        return phi, 1/(torch.clamp(Ls,min=1.0e-12))

    def get_out_dim(self):
        return self.output_dim

    def get_prior(self):
        return self.prior_mean, self.prior_logprec

class XI_VEC_PLUS_ECAPA_TDNN(nn.Module):

    def __init__(self,
                 channels=512,
                 feat_dim=80,
                 embed_dim=192,
                 global_context_att=False,
                 emb_bn=False):
        super().__init__()

        self.layer1 = Conv1dReluBn(feat_dim,
                                   channels,
                                   kernel_size=5,
                                   padding=2)
        self.layer2 = SE_Res2Block(channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=2,
                                   dilation=2,
                                   scale=8)
        self.layer3 = SE_Res2Block(channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=3,
                                   dilation=3,
                                   scale=8)
        self.layer4 = SE_Res2Block(channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=4,
                                   dilation=4,
                                   scale=8)

        cat_channels = channels * 3
        out_channels = 512 * 3
        self.conv = nn.Conv1d(cat_channels, out_channels, kernel_size=1)
        self.pool = XI(in_dim=out_channels)
        self.pool_out_dim = self.pool.get_out_dim()
        self.bn = nn.BatchNorm1d(self.pool_out_dim)
        self.linear = nn.Linear(self.pool_out_dim, embed_dim)
        self.emb_bn = emb_bn
        if emb_bn:  # better in SSL for SV
            self.bn2 = nn.BatchNorm1d(embed_dim)
        else:
            self.bn2 = nn.Identity()

        self.cluster_for_each_spk = {}
        self.uttemb = {}

    def _get_frame_level_feat(self, x):
        # for inner class usage
        x = x.permute(0, 2, 1)  # (B,T,F) -> (B,F,T)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = self.conv(out)

        return out, out4

    def get_frame_level_feat(self, x):
        # for outer interface
        out = self._get_frame_level_feat(x)[0].permute(0, 2, 1)
        return out  # (B, T, D)

    def forward(self, x):
        out, out4 = self._get_frame_level_feat(x)
        out = F.relu(out)
        out, var_diag = self.pool(out)
        out = self.bn(out)
        var_diag = var_diag / (self.bn.running_var + self.bn.eps)
        var_diag = self.bn.weight**2 * var_diag
        out = self.linear(out)
        var = torch.matmul(
            self.linear.weight, torch.matmul(
                torch.diag_embed(var_diag), (self.linear.weight).T))
        var_diag = torch.diagonal(var,dim1=-2, dim2=-1)
        # if self.emb_bn:
        #     out = self.bn2(out)
        #     var_diag = var_diag / (self.bn2.running_var + self.bn2.eps)
        #     var_diag = self.bn2.weight**2 * var_diag
        return var_diag, out # (Batch, emb_dim)


def PLUS_XI_VEC_ECAPA_TDNN_c1024(feat_dim, embed_dim, emb_bn=False):
    return XI_VEC_PLUS_ECAPA_TDNN(channels=1024,
                                 feat_dim=feat_dim,
                                 embed_dim=embed_dim,
                                 emb_bn=emb_bn)


def PLUS_XI_VEC_ECAPA_TDNN_c512(feat_dim, embed_dim, emb_bn=False):
    return XI_VEC_PLUS_ECAPA_TDNN(channels=512,
                                 feat_dim=feat_dim,
                                 embed_dim=embed_dim,
                                 emb_bn=emb_bn)


if __name__ == '__main__':
    x = torch.zeros(1, 200, 80)
    model = PLUS_XI_VEC_ECAPA_TDNN_c512(feat_dim=80,
                                 embed_dim=256)
    model.eval()
    out = model(x)
    print(out.shape)

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))

    # from thop import profile
    # x_np = torch.randn(1, 200, 80)
    # flops, params = profile(model, inputs=(x_np, ))
    # print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
