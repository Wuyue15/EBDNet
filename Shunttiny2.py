import torch
from torch import nn
from backbone.Shunted.SSA import *
import torch.nn.functional as F

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )

def DOWN1(in_, out_):
    return nn.Sequential(
        nn.Conv2d(in_, out_, 3, 1, 1),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
    )

class EA(nn.Module):
    def __init__(self, c, k = 128):
        super(EA, self).__init__()

        self.conv1 = nn.Conv1d(c, c, 1)

        self.k = k

        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.date = self.linear_0.weight.permute(1, 0, 2)
        self.conv2 = nn.Sequential(nn.Conv1d(c, c, 1, bias=False),
                                   nn.BatchNorm1d(c))

        self.softmax = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        idn = x
        x = self.conv1(x)
        # b, c, h, w = x.size()
        # x = x.view(b, self.k, h * w)                       # b * c * n
        attn = self.linear_0(x)                            # b * k * n
        attn = self.softmax(attn)                          # b * k * n
        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))  # b, k, n
        x = self.linear_1(attn)  # b, c, n
        # x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)

        return x

class FGCN(nn.Module):

    def __init__(self, in_ch, ratio=4):
        super(FGCN, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch // 2)
        self.conv2 = nn.Conv2d(in_ch, in_ch // 4, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_ch // 4)

        #  Interaction Space
        #  Adjacency Matrix: (-)A_g
        self.ea1 = EA(128)
        self.conv_adj1 = nn.Conv1d(in_ch // 4, in_ch // 4, kernel_size=1, bias=False)
        self.conv_adj2 = nn.Conv1d(in_ch // 4, in_ch // 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.bn_adj = nn.BatchNorm1d(in_ch // 4)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(in_ch // 2, in_ch // 2, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(in_ch // 2)

        #  last fc
        self.conv3 = nn.Conv2d(in_ch // 2, in_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_ch)

    def to_matrix(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        return x

    def forward(self, x):

        x_sqz, b = x, x

        x_sqz = self.conv1(x_sqz)
        x_sqz = self.bn1(x_sqz)
        x_sqz = self.to_matrix(x_sqz)      #1, 256,100

        b = self.conv2(b)
        b = self.bn2(b)
        b = self.to_matrix(b)               #1,128,100

        # Project
        z_idt = torch.matmul(x_sqz, b.transpose(1, 2))     #1,256,128

        # # # # Interaction Space # # # #
        z = z_idt.transpose(1, 2).contiguous()            #1,128,256

        z1 = self.ea1(z)
        z2 = self.conv_adj1(z)
        z3 = self.conv_adj2(z)
        z3 = self.relu(z3)
        z = z2 * z3 + z1
        # z2 = self.conv_adj1(z)
        # z = self.bn_adj(z2)

        z = z.transpose(1, 2).contiguous()
        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = self.conv_wg(z)
        z = self.bn_wg(z)

        # # # # Re-projection Space # # # #
        # Re-project
        y = torch.matmul(z, b)

        n, _, h, w = x.size()
        y = y.view(n, -1, h, w)

        y = self.conv3(y)
        y = self.bn3(y)

        g_out = F.relu_(x+y)

        # cat or sum, nearly the same results

        return g_out

class shunttiny(nn.Module):
    def __init__(self):
        super(shunttiny, self).__init__()
        self.rgb_net = shunted_t(pretrained=True)
        self.d_net = shunted_t(pretrained=True)

        self.gcn = FGCN(512)

        self.dwa1 = DOWN1(256, 128)
        self.dwa2 = DOWN1(128, 64)
        self.dwb1 = DOWN1(256, 128)
        self.dwb2 = DOWN1(128, 64)

        self.downgi = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )
        #################          监督     ###########################################
        self.s1 = nn.Sequential(
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )
        self.jdgi = nn.Sequential(
            nn.Conv2d(512, 1, 3, 1, 1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True),

        )
        ##############################################################################
        self.upa3 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True),
        )
        self.upb3 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True),
        )
        self.cb1 = convblock(256,256,3,1,1)
        self.cb2 = convblock(256, 256, 3, 1, 1)
        self.cb3 = convblock(128, 128, 3, 1, 1)
        self.cb4 = convblock(128, 128, 3, 1, 1)
        self.dwc1 = DOWN1(256, 128)
        self.dwc2 = DOWN1(256, 128)
        self.dwc3 = DOWN1(128, 64)
        self.dwc4 = DOWN1(128, 64)

        self.avgd3 = nn.AdaptiveAvgPool2d(1)
        self.avgd2 = nn.AdaptiveAvgPool2d(1)
        self.avgd1 = nn.AdaptiveAvgPool2d(1)

    def forward(self, rgb, d):
        d = torch.cat((d, d, d), dim=1)
        rgb_list = self.rgb_net(rgb)
        depth_list = self.d_net(d)

        r1 = rgb_list[0]
        r2 = rgb_list[1]
        r3 = rgb_list[2]
        r4 = rgb_list[3]

        d1 = depth_list[0]
        d2 = depth_list[1]
        d3 = depth_list[2]
        d4 = depth_list[3]

        gi = r4 * d4 + r4 + d4
        gi = self.gcn(gi)
        r4_g = r4 * gi + gi
        d4_g = d4 * gi + gi
        gi = gi + r4_g + d4_g
        gi_1 = self.downgi(gi)
        outgi = self.jdgi(gi)

        a1 = r3 * gi_1 + gi_1
        a1_1 = self.dwa1(a1)
        b1 = self.avgd3(d3) * gi_1 + gi_1
        b1_1 = self.dwb1(b1)

        a2 = r2 * a1_1 + b1_1
        a2_1 = self.dwa2(a2)
        b2 = self.avgd2(d2) * b1_1 + a1_1
        b2_1 = self.dwb2(b2)

        a3 = r1 * a2_1 + b2_1
        a3_3 = self.upa3(a3)
        b3 = self.avgd1(d1) * b2_1 + a2_1
        b3_3 = self.upb3(b3)

        c1 = self.dwc1(self.cb1(b3_3 + a1)) + b2
        c2 = self.dwc2(self.cb2(a3_3 + b1)) + a2

        c3 = self.dwc3(self.cb3(c1)) + a3
        c4 = self.dwc4(self.cb4(c2)) + b3

        s = c3 + c4
        outs = self.s1(s)


        return  outs, outgi,c1,c2,c3,c4,b3,a3
        # return outs, outgi

    def load_pre(self, pre_model):
        save_model = torch.load(pre_model)
        model_dict_r = self.rgb_net.state_dict()
        state_dict_r = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_r.update(state_dict_r)
        self.rgb_net.load_state_dict(model_dict_r)

        model_dict_d = self.d_net.state_dict()
        state_dict_d = {k: v for k, v in save_model.items() if k in model_dict_r.keys()}
        model_dict_d.update(state_dict_d)
        self.d_net.load_state_dict(model_dict_d)
        ############################################################
        # sk = torch.load(pre_model)
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in sk.items():
        #     name = k[9:]
        #     new_state_dict[name] = v
        # self.rgb_net.load_state_dict(new_state_dict, strict=False)
        # self.t_net.load_state_dict(new_state_dict, strict=False)
        # # self.rgb_depth.load_state_dict(new_state_dict, strict=False)
        print('self.rgb_uniforr loading', 'self.depth_unifor loading')

if __name__ == '__main__':
    rgb = torch.randn([1, 3, 320, 320])                                      # batch_size=1，通道3，图片尺寸320*320
    depth = torch.randn([1, 1, 320, 320])
    model = shunttiny()
    a = model(rgb, depth)
    print(a[0].shape)
    print(a[1].shape)
    print(a[2].shape)
    print(a[3].shape)
    print(a[4].shape)
    print(a[5].shape)
    print(a[6].shape)
    print(a[7].shape)
    print(a[8].shape)
    print(a[9].shape)
    print(a[10].shape)
    print(a[11].shape)

