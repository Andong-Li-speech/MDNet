import torch
import torch.nn as nn
from torch import Tensor
from utils.utils import NormSwitch


class MDNet(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: list,
                 k2: list,
                 c: int,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 q: int,
                 fft_num: int,
                 init_alpha: float,
                 intra_connect: str,
                 fusion_type: str,
                 compress_type: str,
                 norm_type: str,
                 is_u2: bool,
                 is_gate: bool,
                 is_causal: bool,
                 customed_compress=None,
                 ):
        super(MDNet, self).__init__()
        self.cin = cin
        self.k1 = tuple(k1)
        self.k2 = tuple(k2)
        self.c = c
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.q = q  # unfolding steps
        self.fft_num = fft_num
        self.init_alpha = init_alpha
        self.intra_connect = intra_connect
        self.fusion_type = fusion_type
        self.compress_type = compress_type
        self.norm_type = norm_type
        self.is_u2 = is_u2
        self.is_gate = is_gate
        self.is_causal = is_causal
        self.customed_compress = customed_compress

        # Components
        if is_u2:
            self.en = U2Net_Encoder(cin, self.k1, self.k2, c, intra_connect, norm_type)
        else:
            self.en = UNet_Encoder(cin, self.k1, c, norm_type)

        # init part
        self.init_module = InitParam_Module(kd1, cd1, d_feat, p, fft_num, is_gate, is_causal, norm_type,
                                            acti_type="sigmoid")

        # params update part
        params_update_list = []
        if q > 0:
            for i in range(q):
                params_update_list.append(UpdateParam_Module(kd1, cd1, d_feat, p, fft_num, init_alpha, is_gate,
                                                             is_causal, norm_type))
            self.params_update = nn.ModuleList(params_update_list)

        if fusion_type == "latent":
            self.fusion = FusionModule(norm_type=norm_type)


    def compress_transform(self, x):
        if self.compress_type == "sqrt":
            x_mag, x_phase = torch.norm(x, dim=1), torch.atan2(x[:,-1,...], x[:,0,...])
            x = torch.stack((((x_mag+1e-8)**0.5)*torch.cos(x_phase),
                             ((x_mag+1e-8)**0.5)*torch.sin(x_phase)), dim=1)
        elif self.compress_type == "cubic":
            x_mag, x_phase = torch.norm(x, dim=1), torch.atan2(x[:,-1,...], x[:,0,...])
            x = torch.stack((((x_mag+1e-8)**0.3)*torch.cos(x_phase),
                             ((x_mag+1e-8)**0.3)*torch.sin(x_phase)), dim=1)
        elif self.compress_type == "normal":
            x = x
        elif self.customed_compress is not None:
            assert (self.customed_compress > 0) and (self.customed_compress < 1), "the compression value should range from 0 to 1"
            x_mag, x_phase = torch.norm(x, dim=1), torch.atan2(x[:, -1, ...], x[:, 0, ...])
            x = torch.stack((((x_mag + 1e-8)**self.customed_compress)*torch.cos(x_phase),
                             ((x_mag + 1e-8)**self.customed_compress)*torch.sin(x_phase)), dim=1)
        return x


    def forward(self, inpt):
        """
            inpt: (B,2,T,F)
            return:
                esti_s_list: list, element: (B,2,F,T)
                esti_n_list: list, element: (B,2,F,T)
                esti_s: (B,2,F,T)
        """
        if inpt.ndim == 3:
            inpt = inpt.unsqueeze(dim=1)
        b_size, _, seq_num, freq_num = inpt.shape
        inpt_trans = inpt.transpose(-2, -1).contiguous()
        en_x, _ = self.en(inpt)
        en_x = en_x.transpose(-2, -1).contiguous().view(b_size, -1, seq_num)
        esti_s_list, esti_n_list = [], []
        # init params
        gs, gn, rs, rn, tilde_s, tilde_n = self.init_module(en_x, inpt_trans)
        update_s = 0.5 * (tilde_s + inpt_trans - tilde_n)
        update_n = 0.5 * (tilde_n + inpt_trans - tilde_s)
        esti_s_list.append(self.compress_transform(tilde_s))
        esti_n_list.append(self.compress_transform(tilde_n))
        # update params
        for idx in range(self.q):
            gs, gn, rs, rn, tilde_s, tilde_n = self.params_update[idx](gs, gn, rs, rn, en_x, inpt_trans, update_s,
                                                                       update_n)
            update_s = 0.5 * (tilde_s + inpt_trans - tilde_n)
            update_n = 0.5 * (tilde_n + inpt_trans - tilde_s)
            esti_s_list.append(self.compress_transform(tilde_s))
            esti_n_list.append(self.compress_transform(tilde_n))

        # compress the spectrum dynamic range, (B, 2, T, F)
        tilde_s, tilde_n = tilde_s.transpose(-2, -1), tilde_n.transpose(-2, -1)
        tilde_s_aux = inpt - tilde_n
        tilde_s, tilde_s_aux = self.compress_transform(tilde_s), self.compress_transform(tilde_s_aux)
        inpt_x = self.compress_transform(inpt)

        # signal fusion by either hard or latent
        if self.fusion_type == "hard":
            esti_s = 0.5 * (tilde_s + tilde_s_aux).transpose(-2, -1)
        elif self.fusion_type == "latent":
            esti_s = self.fusion(tilde_s, tilde_s_aux, inpt_x)
        else:
            raise RuntimeError("only support hard and latent currently")
        return esti_s_list, esti_n_list, esti_s


class InitParam_Module(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 fft_num: int,
                 is_gate: bool,
                 is_causal: bool,
                 norm_type: str,
                 acti_type: str = "sigmoid",
                 ):
        super(InitParam_Module, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.fft_num = fft_num
        self.is_gate = is_gate
        self.is_causal = is_causal
        self.norm_type = norm_type
        self.acti_type = acti_type

        # Components
        self.gain_sn = MagModule(kd1, cd1, d_feat, p, fft_num, True, is_gate, is_causal, acti_type, norm_type)
        self.res_s, self.res_n = CompModule(kd1, cd1, d_feat, p, fft_num, is_gate, is_causal, norm_type), \
                                 CompModule(kd1, cd1, d_feat, p, fft_num, is_gate, is_causal, norm_type)

    def forward(self, feat_x, x):
        """
        In the initialized stage, both pre_s and pre_n are original noisy mixture
        :param feat_x: (B, C, T)
        :param x: (B, 2, F, T)
        :return:
        """
        b_size, _, freq_num, seq_len = x.shape
        com_x = x.view(b_size, -1, seq_len).contiguous()
        mag_x = torch.norm(x, dim=1)

        # params initialization
        gs, gn = self.gain_sn(feat_x, mag_x, mag_x)
        rs, rn = self.res_s(feat_x, com_x), self.res_n(feat_x, com_x)
        tilde_s = gs * x + rs
        tilde_n = gn * x + rn
        return gs, gn, rs, rn, tilde_s, tilde_n


class UpdateParam_Module(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 fft_num: int,
                 init_alpha: float,
                 is_gate: bool,
                 is_causal: bool,
                 norm_type: str,
                 acti_type=None,
                 ):
        super(UpdateParam_Module, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.fft_num = fft_num
        self.init_alpha = init_alpha
        self.is_gate = is_gate
        self.is_causal = is_causal
        self.norm_type = norm_type
        self.acti_type = acti_type

        # Components
        self.gain_updata = GainUpdate_Module(kd1, cd1, d_feat, p, fft_num, init_alpha, is_gate, is_causal, norm_type)
        self.resi_update = ResiUpdate_Module(kd1, cd1, d_feat, p, fft_num, init_alpha, is_gate, is_causal, norm_type)

    def forward(self, gs, gn, rs, rn, feat_x, x, pre_s, pre_n):
        # update gs and gn
        gs, gn = self.gain_updata(gs, gn, rs, rn, feat_x, x, pre_s, pre_n)
        rs, rn = self.resi_update(gs, gn, rs, rn, feat_x, x, pre_s, pre_n)

        # update params
        tilde_s = gs * x + rs
        tilde_n = gn * x + rn
        return gs, gn, rs, rn, tilde_s, tilde_n


class GainUpdate_Module(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 fft_num: int,
                 init_alpha: float,
                 is_gate: bool,
                 is_causal: bool,
                 norm_type: str,
                 acti_type=None,
                 ):
        super(GainUpdate_Module, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.fft_num = fft_num
        self.init_alpha = init_alpha
        self.is_gate = is_gate
        self.is_causal = is_causal
        self.norm_type = norm_type
        self.acti_type = acti_type

        # Components
        self.alpha_gs = nn.Parameter(torch.Tensor([init_alpha]), requires_grad=True)  # for gradient descent
        self.alpha_gn = nn.Parameter(torch.Tensor([init_alpha]), requires_grad=True)
        self.gradient_net = MagModule(kd1, cd1, d_feat, p, fft_num, False, is_gate, is_causal, acti_type, norm_type)

    def forward(self, gs, gn, rs, rn, feat_x, x, pre_s, pre_n):
        """
        :param gs: (B, 1, F, T)
        :param gn: (B, 1, F, T)
        :param rs: (B, 2, F, T)
        :param rn: (B, 2, F, T)
        :param feat_x: (B, C, T)
        :param pre_s: (B, 2, F, T)
        :param pre_n: (B, 2, F, T)
        :param x: (B, 2, F, T)
        :return:
        """
        mag_s, mag_n = torch.norm(pre_s, dim=1), torch.norm(pre_n, dim=1)
        gradient_mag_s, gradient_mag_n = self.gradient_net(feat_x, mag_s, mag_n)  # (B, 1, F, T)
        delta_gs = torch.mean(((gs + gn - 1) * x + rs + rn) * x, dim=1, keepdim=True)  # (B, 1, F, T)
        delta_gn = torch.mean(((gs + gn - 1) * x + rs + rn) * x, dim=1, keepdim=True)  # (B, 1, F, T)
        gs = gs - self.alpha_gs * (gradient_mag_s + delta_gs)
        gn = gn - self.alpha_gn * (gradient_mag_n + delta_gn)
        return torch.clamp(gs, 0, 1), torch.clamp(gn, 0, 1)


class ResiUpdate_Module(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 fft_num: int,
                 init_alpha: float,
                 is_gate: bool,
                 is_causal: bool,
                 norm_type: str,
                 ):
        super(ResiUpdate_Module, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.fft_num = fft_num
        self.init_alpha = init_alpha
        self.is_gate = is_gate
        self.is_causal = is_causal
        self.norm_type = norm_type

        # Components
        self.alpha_rs = nn.Parameter(torch.Tensor([init_alpha]), requires_grad=True)
        self.alpha_rn = nn.Parameter(torch.Tensor([init_alpha]), requires_grad=True)
        self.gradient_net_s = CompModule(kd1, cd1, d_feat, p, fft_num, is_gate, is_causal, norm_type)
        self.gradient_net_n = CompModule(kd1, cd1, d_feat, p, fft_num, is_gate, is_causal, norm_type)

    def forward(self, gs, gn, rs, rn, feat_x, x, pre_s, pre_n):
        """
        :param gs: (B, 1, F, T)
        :param gn: (B, 1, F, T)
        :param rs: (B, 2, F, T)
        :param rn: (B, 2, F, T)
        :param feat_x: (B, C, T)
        :param x: (B, 2, F, T)
        :param pre_s: (B, 2, F, T)
        :param pre_n: (B, 2, F, T)
        :return:
        """
        b_size, _, freq_num, seq_len = pre_s.shape
        pre_s_1d = pre_s.view(b_size, -1, seq_len).contiguous()
        pre_n_1d = pre_n.view(b_size, -1, seq_len).contiguous()
        gradient_res_s, gradient_res_n = self.gradient_net_s(feat_x, pre_s_1d), self.gradient_net_n(feat_x, pre_n_1d)
        delta_rs, delta_rn = ((gs + gn - 1) * x + rs + rn), \
                             ((gs + gn - 1) * x + rs + rn)
        rs = rs - self.alpha_rs * (gradient_res_s + delta_rs)
        rn = rn - self.alpha_rn * (gradient_res_n + delta_rn)
        return rs, rn


class CompModule(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 fft_num: int,
                 is_gate: bool,
                 is_causal: bool,
                 norm_type: str,
                 ):
        super(CompModule, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.fft_num = fft_num
        self.is_gate = is_gate
        self.is_causal = is_causal
        self.norm_type = norm_type

        # Components
        cin = (fft_num//2+1)*2 + d_feat
        self.in_conv = nn.Sequential(
            nn.Conv1d(cin, d_feat, 1),
            NormSwitch(norm_type, "1D", d_feat),
            nn.PReLU(d_feat))
        self.tcns = TCNGroup(kd1, cd1, d_feat, p, norm_type, is_gate, is_causal)
        self.linear_com_r, self.linear_com_i = nn.Conv1d(d_feat, fft_num//2+1, 1), nn.Conv1d(d_feat, fft_num//2+1, 1)

    def forward(self, x: Tensor, com_x: Tensor):
        """
        :param x: (B, C1, T)
        :param com_x: (B, C2, T)
        :return: (B, 2, F, T)
        """
        x = self.in_conv(torch.cat((x, com_x), dim=1))
        x = self.tcns(x)
        x_r, x_i = self.linear_com_r(x), self.linear_com_i(x)
        return torch.stack((x_r, x_i), dim=1).contiguous()


class MagModule(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 fft_num: int,
                 is_init: bool,
                 is_gate: bool,
                 is_causal: bool,
                 acti_type: str,
                 norm_type: str,
                 ):
        super(MagModule, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.fft_num = fft_num
        self.is_init = is_init
        self.is_gate = is_gate
        self.is_causal = is_causal
        self.acti_type = acti_type
        self.norm_type = norm_type

        if not is_init:
            cin = (fft_num//2+1)*2 + d_feat
        else:
            cin = (fft_num//2+1) + d_feat
        self.in_conv = nn.Sequential(
            nn.Conv1d(cin, d_feat, 1),
            NormSwitch(norm_type, "1D", d_feat),
            nn.PReLU(d_feat))
        self.tcns = TCNGroup(kd1, cd1, d_feat, p, norm_type, is_gate, is_causal)

        if acti_type is not None:
            if acti_type == "relu":
                acti_func = nn.ReLU()
            elif acti_type == "tanh":
                acti_func = nn.Tanh()
            elif acti_type == "sigmoid":
                acti_func = nn.Sigmoid()
            self.linear_mag_s, self.linear_mag_n = nn.Sequential(
                nn.Conv1d(d_feat, fft_num//2+1, 1),
                acti_func
            ), nn.Sequential(
                nn.Conv1d(d_feat, fft_num//2+1, 1),
                acti_func
            )
        else:
            self.linear_mag_s, self.linear_mag_n = nn.Conv1d(d_feat, fft_num//2+1, 1), nn.Conv1d(d_feat, fft_num//2+1, 1)


    def forward(self, x: Tensor, mag_s: Tensor, mag_n: Tensor):
        """
        :param x: (B, C1, T)
        :param mag_s: (B, C2, T)
        :param mag_n: (B, C2, T)
        :return: (B, 1, F, T), (B, 1, F, T)
        """
        if not self.is_init:
            x = self.in_conv(torch.cat((x, mag_s, mag_n), dim=1))
        else:
            x = self.in_conv(torch.cat((x, mag_s), dim=1))
        x = self.tcns(x)
        mag_s, mag_n = self.linear_mag_s(x), self.linear_mag_n(x)
        return mag_s.unsqueeze(dim=1), mag_n.unsqueeze(dim=1)


class TCNGroup(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 p: int,
                 norm_type: str,
                 is_gate: bool = False,
                 is_causal: bool = True,
                 ):
        super(TCNGroup, self).__init__()
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.norm_type = norm_type
        self.is_gate = is_gate
        self.is_causal = is_causal

        self.dila_list = [1, 2, 5, 9]
        tcm_list = []
        for i in range(p):
            tcm_group = []
            for j in range(len(self.dila_list)):
                tcm_group.append(SqueezedTCM(kd1, cd1, d_feat, self.dila_list[j], norm_type, is_causal, is_gate))
            tcm_list.append(nn.Sequential(*tcm_group))
        self.tcm_list = nn.ModuleList(tcm_list)

    def forward(self, inpt_x: Tensor) -> Tensor:
        x = inpt_x
        for i in range(self.p):
            x = self.tcm_list[i](x)
        return x


class SqueezedTCM(nn.Module):
    def __init__(self,
                 kd1: int,
                 cd1: int,
                 d_feat: int,
                 dilation: int,
                 norm_type: str,
                 is_causal: bool = True,
                 is_gate: bool = False,
                 pad_type: str = "constant",
                 ):
        super(SqueezedTCM, self).__init__()
        self.d_feat = d_feat
        self.kd1 = kd1
        self.cd1 = cd1
        self.dilation = dilation
        self.norm_type = norm_type
        self.is_causal = is_causal
        self.is_gate = is_gate
        self.pad_type = pad_type

        assert pad_type in ["constant", "replicate", "reflection"], "padding should be one of the three types"

        if is_causal:
            if pad_type == "constant":
                pad = nn.ConstantPad1d(((kd1-1)*dilation, 0), value=0.)
            elif pad_type == "reflection":
                pad = nn.ReflectionPad1d(((kd1-1)*dilation, 0))
            elif pad_type == "replicate":
                pad = nn.ReplicationPad1d(((kd1-1)*dilation, 0))
        else:
            if pad_type == "constant":
                pad = nn.ConstantPad1d(((kd1-1)//2*dilation, (kd1-1)//2*dilation), value=0.)
            elif pad_type == "reflection":
                pad = nn.ReflectionPad1d(((kd1-1)//2*dilation, (kd1-1)//2*dilation))
            elif pad_type == "replicate":
                pad = nn.ReplicationPad1d(((kd1-1)//2*dilation, (kd1-1)//2*dilation))

        self.in_conv = nn.Conv1d(d_feat, cd1, kernel_size=1, bias=False)
        self.dd_conv_main = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            pad,
            nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False)
        )
        self.out_conv = nn.Sequential(
            nn.PReLU(cd1),
            NormSwitch(norm_type, "1D", cd1),
            nn.Conv1d(cd1, d_feat, kernel_size=1, bias=False)
            )
        if is_gate:
            self.dd_conv_gate = nn.Sequential(
                nn.PReLU(cd1),
                NormSwitch(norm_type, "1D", cd1),
                pad,
                nn.Conv1d(cd1, cd1, kernel_size=kd1, dilation=dilation, bias=False),
                nn.Sigmoid()
            )

    def forward(self, x: Tensor) -> Tensor:
        resi = x
        x = self.in_conv(x)
        if self.is_gate:
            x = self.dd_conv_main(x) * self.dd_conv_gate(x)
        else:
            x = self.dd_conv_main(x)
        x = self.out_conv(x)
        x = x + resi
        return x


class FusionModule(nn.Module):
    def __init__(self,
                 cin: int = 3*2,
                 k1: tuple = (1, 3),
                 c: int = 32,
                 kd1: int = 3,
                 cd1: int = 64,
                 d_feat: int = 256,
                 p: int = 2,
                 norm_type: str = "cLN",
                 is_gate: bool = False,
                 is_causal: bool = True,
                 ):
        super(FusionModule, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.kd1 = kd1
        self.cd1 = cd1
        self.d_feat = d_feat
        self.p = p
        self.norm_type = norm_type
        self.is_gate = is_gate
        self.is_causal = is_causal

        self.en = UNet_Encoder(cin, k1, c, norm_type)
        self.de = UNet_Decoder(c, k1, norm_type)
        self.tcns = TCNGroup(kd1, cd1, d_feat, p, norm_type, is_gate, is_causal)

    def forward(self, x_main, x_aux, inpt_x):
        """
        :param x_main: (B, 2, T, F)
        :param x_aux: (B, 2, T, F)
        :param inpt_x: (B, 2, T, F)
        :return:
        """
        b_size, _, seq_len, freq_num = x_main.shape
        x_in = torch.cat((x_main, x_aux, inpt_x), dim=1)
        x, x_list = self.en(x_in)
        c = x.shape[1]
        x = x.transpose(-2, -1).contiguous().view(b_size, -1, seq_len)
        x = self.tcns(x)
        x = x.view(b_size, c, -1, seq_len).transpose(-2, -1)
        x = self.de(x, x_list)
        return x_main + x


class U2Net_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 k2: tuple,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(U2Net_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.k2 = k2
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        c_last = 64
        kernel_begin = (k1[0], 5)
        stride = (1, 2)
        meta_unet = []
        meta_unet.append(
            En_unet_module(cin, c, kernel_begin, k2, intra_connect, norm_type=norm_type, scale=4, de_flag=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type=norm_type, scale=3, de_flag=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type=norm_type, scale=2, de_flag=False))
        meta_unet.append(
            En_unet_module(c, c, k1, k2, intra_connect, norm_type=norm_type, scale=1, de_flag=False))
        self.meta_unet_list = nn.ModuleList(meta_unet)
        self.last_conv = nn.Sequential(
            GateConv2d(c, c_last, k1, stride, (0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c_last),
            nn.PReLU(c_last)
        )

    def forward(self, x: Tensor) -> tuple:
        x_list = []
        for i in range(len(self.meta_unet_list)):
            x = self.meta_unet_list[i](x)
            x_list.append(x)
        x = self.last_conv(x)
        x_list.append(x)
        return x, x_list


class UNet_Encoder(nn.Module):
    def __init__(self,
                 cin: int,
                 k1: tuple,
                 c: int,
                 norm_type: str,
                 ):
        super(UNet_Encoder, self).__init__()
        self.cin = cin
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        kernel_begin = (k1[0], 5)
        stride = (1, 2)
        c_final = 64
        unet = []
        unet.append(nn.Sequential(
            GateConv2d(cin, c, kernel_begin, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c, k1, stride, padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConv2d(c, c_final, k1, (1,2), padding=(0, 0, k1[0]-1, 0)),
            NormSwitch(norm_type, "2D", c_final),
            nn.PReLU(c_final)))
        self.unet_list = nn.ModuleList(unet)

    def forward(self, x: Tensor) -> tuple:
        x_list = []
        for i in range(len(self.unet_list)):
            x = self.unet_list[i](x)
            x_list.append(x)
        return x, x_list


class UNet_Decoder(nn.Module):
    def __init__(self,
                 c,
                 k1,
                 norm_type,
                 ):
        super(UNet_Decoder, self).__init__()
        self.k1 = k1
        self.c = c
        self.norm_type = norm_type
        c_begin = 64
        kernel_end = (k1[0], 5)
        stride = (1,2)
        unet = []
        unet.append(
            nn.Sequential(
            GateConvTranspose2d(c_begin*2, c, k1, stride),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)))
        unet.append(nn.Sequential(
            GateConvTranspose2d(c*2, c, k1, stride),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)
        ))
        unet.append(nn.Sequential(
            GateConvTranspose2d(c*2, c, k1, stride),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)
        ))
        unet.append(nn.Sequential(
            GateConvTranspose2d(c*2, c, k1, stride),
            NormSwitch(norm_type, "2D", c),
            nn.PReLU(c)
        ))
        unet.append(GateConvTranspose2d(c*2, 2, kernel_end, stride))
        self.unet_list = nn.ModuleList(unet)
        self.linear_r, self.linear_i = nn.Linear(161, 161), nn.Linear(161, 161)

    def forward(self, x: Tensor, en_list: list) -> Tensor:
        for i in range(len(self.unet_list)):
            tmp = torch.cat((x, en_list[-(i+1)]), dim=1)
            x = self.unet_list[i](tmp)
        x_r, x_i = x[:,0,...], x[:,-1,...]
        x_r, x_i = self.linear_r(x_r), self.linear_i(x_i)
        return torch.stack((x_r, x_i), dim=1)


class En_unet_module(nn.Module):
    def __init__(self,
                 cin: int,
                 cout: int,
                 k1: tuple,
                 k2: tuple,
                 intra_connect: str,
                 norm_type: str,
                 scale: int,
                 de_flag: bool = False):
        super(En_unet_module, self).__init__()
        self.cin = cin
        self.cout = cout
        self.k1 = k1
        self.k2 = k2
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        self.scale = scale
        self.de_flag = de_flag
        stride = (1, 2)

        in_conv_list = []
        if not de_flag:
            in_conv_list.append(GateConv2d(cin, cout, k1, stride, (0, 0, k1[0]-1, 0)))
        else:
            in_conv_list.append(GateConvTranspose2d(cin, cout, k1, stride))
        in_conv_list.append(NormSwitch(norm_type, "2D", cout))
        in_conv_list.append(nn.PReLU(cout))
        self.in_conv = nn.Sequential(*in_conv_list)

        enco_list, deco_list = [], []
        for _ in range(scale):
            enco_list.append(Conv2dunit(k2, cout, norm_type))
        for i in range(scale):
            if i == 0:
                deco_list.append(Deconv2dunit(k2, cout, "add", norm_type))
            else:
                deco_list.append(Deconv2dunit(k2, cout, intra_connect, norm_type))
        self.enco = nn.ModuleList(enco_list)
        self.deco = nn.ModuleList(deco_list)
        self.skip_connect = SkipConnect(intra_connect)

    def forward(self, inputs: Tensor) -> Tensor:
        x_resi = self.in_conv(inputs)
        x = x_resi
        x_list = []
        for i in range(len(self.enco)):
            x = self.enco[i](x)
            x_list.append(x)

        for i in range(len(self.deco)):
            if i == 0:
                x = self.deco[i](x)
            else:
                x_con = self.skip_connect(x, x_list[-(i+1)])
                x = self.deco[i](x_con)
        x_resi = x_resi + x
        del x_list
        return x_resi


class Conv2dunit(nn.Module):
    def __init__(self,
                 k: int,
                 c: int,
                 norm_type: str,
                 ):
        super(Conv2dunit, self).__init__()
        self.k = k
        self.c = c
        self.norm_type = norm_type
        k_t = k[0]
        stride = (1,2)
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d((0, 0, k_t-1, 0), value=0.),
                nn.Conv2d(c, c, k, stride),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c)
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(c, c, k, stride),
                NormSwitch(norm_type, "2D", c),
                nn.PReLU(c)
            )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class Deconv2dunit(nn.Module):
    def __init__(self,
                 k: int,
                 c: int,
                 intra_connect: str,
                 norm_type: str,
                 ):
        super(Deconv2dunit, self).__init__()
        self.k = k
        self.c = c
        self.intra_connect = intra_connect
        self.norm_type = norm_type
        k_t = k[0]
        stride = (1, 2)
        deconv_list = []
        if intra_connect == "add":
            if k_t > 1:
                deconv_list.append(nn.ConvTranspose2d(c, c, k, stride)),
                deconv_list.append(Chomp_T(k_t-1))
            else:
                deconv_list.append(nn.ConvTranspose2d(c, c, k, stride))
        elif intra_connect == "cat":
            if k_t > 1:
                deconv_list.append(nn.ConvTranspose2d(2*c, c, k, stride))
                deconv_list.append(Chomp_T(k_t-1))
            else:
                deconv_list.append(nn.ConvTranspose2d(2*c, c, k, stride))
        deconv_list.append(NormSwitch(norm_type, "2D", c))
        deconv_list.append(nn.PReLU(c))
        self.deconv = nn.Sequential(*deconv_list)

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.dim() == 4
        return self.deconv(inputs)


class GateConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 padding: tuple,
                 ):
        super(GateConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConstantPad2d(padding, value=0.),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size, stride=stride))
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                  stride=stride)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(dim=1)
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class GateConvTranspose2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 ):
        super(GateConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        k_t = kernel_size[0]
        if k_t > 1:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                   stride=stride),
                Chomp_T(k_t-1))
        else:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels*2, kernel_size=kernel_size,
                                           stride=stride)

    def forward(self, inputs: Tensor) -> Tensor:
        assert inputs.dim() == 4
        x = self.conv(inputs)
        outputs, gate = x.chunk(2, dim=1)
        return outputs * gate.sigmoid()


class SkipConnect(nn.Module):
    def __init__(self,
                 intra_connect: str,
                 ):
        super(SkipConnect, self).__init__()
        self.intra_connect = intra_connect

    def forward(self, x_main: Tensor, x_aux: Tensor) -> Tensor:
        if self.intra_connect == "add":
            x = x_main + x_aux
        elif self.intra_connect == "cat":
            x = torch.cat((x_main, x_aux), dim=1)
        return x


class Chomp_T(nn.Module):
    def __init__(self,
                 t: int,
                 ):
        super(Chomp_T, self).__init__()
        self.t = t

    def forward(self, x: Tensor) -> Tensor:
        return x[..., :-self.t, :]


if __name__ == "__main__":
    net = MDNet(cin=2,
                k1=[1,3],
                k2=[2,3],
                c=64,
                kd1=3,
                cd1=64,
                d_feat=256,
                p=2,
                q=3,
                fft_num=320,
                init_alpha=0.01,
                intra_connect="cat",
                is_u2=True,
                is_gate=False,
                is_causal=True,
                compress_type="sqrt",
                norm_type="cLN",
                customed_compress=None,
                fusion_type="latent"
                ).cuda()
    net.eval()
    from utils.utils import numParams
    print(f"The number of trainable parameters:{numParams(net)}")
    x = torch.rand([3, 2, 101, 161]).cuda()
    _, _, y = net(x)
    print(f"{x.shape}->{y.shape}")
    from ptflops.flops_counter import get_model_complexity_info
    macs, params = get_model_complexity_info(net, (2, 101, 161))
