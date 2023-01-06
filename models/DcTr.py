import torch
from torch import nn

from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1
from .Transformer import PCTransformer
from .build import MODELS
import numpy
from .utils_skip import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, Transformer
from .skip_transformer import SkipTransformer


def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc

class SkipAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = k
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step
        self.SkipAttention = SkipAttention(512, 512, num_heads=8, qkv_bias=False,
                                               qk_scale=None, attn_drop=0, proj_drop=0)
        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.stage2_layer1 = nn.Linear(128, 336)
        self.stage1_layer1 = nn.Linear(512, 672)


        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding3 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim // 2, 1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim // 2, 3, 1),
        )

    def forward(self, x, f_stage1, f_stage2):
        #print('x.shape:',x.shape)
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        #print('features.shape:', features.shape)

        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        #print('x.shape:', x.shape)
        fd1 = self.folding1(x)
        #print('fd1.shape:', fd1.shape)
        f_stage2 = self.stage2_layer1(f_stage2)
        f_stage2 = f_stage2.view(bs, 3, -1)
        # print('f_stage2_fold.shape:', f_stage2.shape)
        #f_stage2 = self.stage2_layer2(f_stage2)
        fd1 = self.SkipAttention(fd1, f_stage2, fd1)
        x = torch.cat([fd1, features], dim=1)
        #print('x.shape:', x.shape)

        fd2 = self.folding2(x)
        f_stage1 = self.stage1_layer1(f_stage1)
        f_stage1 = f_stage1.view(bs, 3, -1)
        fd2 = self.SkipAttention(fd2, f_stage1, fd2)
        #print('fd2.shape:', fd2.shape)
        x = torch.cat([fd2, features], dim=1)
        #print('x.shape:', x.shape)
        fd3 = self.folding3(x)
        #print('fd3.shape:', fd3.shape)

        return fd3


class SeedGenerator(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(SeedGenerator, self).__init__()
        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat):
        """
        Args:
            feat: Tensor (b, dim_feat, 1)
        """
        x1 = self.ps(feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, 256)
        return completion


class SPD(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1):
        """Snowflake Point Deconvolution"""
        super(SPD, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])

        self.skip_transformer = SkipTransformer(self.i, in_channel=128, dim=64)

        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)   # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, pcd_prev, feat_global, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_global.repeat(1, 1, feat_1.size(2))], 1)
        Q = self.mlp_2(feat_1)

        H = self.skip_transformer(pcd_prev, K_prev if K_prev is not None else Q, Q)

        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / self.radius**self.i  # (B, 3, N_prev * up_factor)
        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta

        return pcd_child, K_curr


class Foldnew(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=None):
        super(Foldnew, self).__init__()
        self.num_p0 = num_p0
        self.decoder_coarse = SeedGenerator(dim_feat=dim_feat, num_pc=num_pc)
        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors
        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(SPD(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius))
        self.SkipAttention = SkipAttention(512, 512, num_heads=8, qkv_bias=False,
                                           qk_scale=None, attn_drop=0, proj_drop=0)
        self.stage2_layer1 = nn.Linear(128, 4)
        self.stage1_layer1 = nn.Linear(512, 8)
        self.uppers = nn.ModuleList(uppers)
        #print(self.uppers)
    def forward(self, feat, partial, f_stage1, f_stage2, return_P0=False):
        """
        Args:
            feat: Tensor, (b, dim_feat, n)
            partial: Tensor, (b, n, 3)
        """
        bs, _, _ = feat.shape
        #print('feat_shape:', feat.shape)
        #print('f_stage1_shape:', f_stage1.shape)
        #print('f_stage2_shape:', f_stage2.shape)
        arr_pcd = []
        pcd = self.decoder_coarse(feat).permute(0, 2, 1).contiguous()  # (B, num_pc, 3)
        arr_pcd.append(pcd)
        pcd = fps_subsample(torch.cat([pcd, partial], 1), self.num_p0)
        if return_P0:
            arr_pcd.append(pcd)
        K_prev = None
        pcd = pcd.permute(0, 2, 1).contiguous()
        i = 0
        for upper in self.uppers:
            #print('upper:',upper)
            pcd, K_prev = upper(pcd, feat, K_prev)
            if i == 0:
                f_stage2 = self.stage2_layer1(f_stage2)
                f_stage2 = f_stage2.view(bs, 512, -1).transpose(1,2)
                feat = feat.transpose(1,2)
                #print('feat_shape:', feat.shape)
                #print('f_stage2_shape:', f_stage2.shape)
                feat = self.SkipAttention(feat, f_stage2, f_stage2)
                feat = feat.transpose(1,2)
                #print('feat_shape:', feat.shape)
            if i == 1:
                f_stage1 = self.stage1_layer1(f_stage1)
                f_stage1 = f_stage1.view(bs, 512, -1).transpose(1,2)
                feat = feat.transpose(1, 2)
                #print('feat_shape:', feat.shape)
                #print('f_stage1_shape:', f_stage1.shape)
                feat = self.SkipAttention(feat, f_stage1, f_stage1)
                feat = feat.transpose(1, 2)
                #print('feat_shape:', feat.shape)
            #print('i + pcd_shape:', i, ' ', pcd.shape)
            i += 1
            #arr_pcd.append(pcd.permute(0, 2, 1).contiguous())
        pcd = pcd.permute(0, 2, 1).contiguous()
            #print(arr_pcd)
        arr_pcd = pcd
        return arr_pcd


@MODELS.register_module()
class DcTr(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim
        self.knn_layer = config.knn_layer
        self.num_pred = config.num_pred
        self.num_query = config.num_query
        self.layer1 = nn.Linear(1024, 512)
        self.fold_step = int(pow(self.num_pred//self.num_query, 0.5) + 0.5)
        self.base_model = PCTransformer(in_chans = 3, embed_dim = self.trans_dim, depth = [6, 8], drop_rate = 0., num_query = self.num_query, knn_layer = self.knn_layer)
        
        self.foldingnet = Fold(self.trans_dim, step = self.fold_step, hidden_dim = 256)  # rebuild a cluster point
        self.decoder = Foldnew(dim_feat=512, num_pc=256, num_p0=512, radius=1, up_factors=[4, 8])
        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def forward(self, xyz):
        q, coarse_point_cloud, f_stage1, f_stage2 = self.base_model(xyz) # B M C and B M 3
    
        B, M ,C = q.shape
        #print('B:',B)
        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024
        global_feature = self.layer1(global_feature)
        global_feature = global_feature.unsqueeze(-1)
        #print('global_feature:', global_feature.shape)

        #rebuild_feature = torch.cat([
        #    global_feature.unsqueeze(-2).expand(-1, M, -1),
        #    q,
        #    coarse_point_cloud], dim=-1)  # B M 1027 + C

        #rebuild_feature = self.reduce_map(rebuild_feature.reshape(B*M, -1)) # BM C
        # # NOTE: try to rebuild pc
        # coarse_point_cloud = self.refine_coarse(rebuild_feature).reshape(B, M, 3)

        # NOTE: foldingNet
        #relative_xyz = self.foldingnet(rebuild_feature, f_stage1, f_stage2).reshape(B, M, 3, -1)    # B M 3 S
        #rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)  # B N 3

        # NOTE: fc
        # relative_xyz = self.refine(rebuild_feature)  # BM 3S
        # rebuild_points = (relative_xyz.reshape(B,M,3,-1) + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)

        # cat the input
        #inp_sparse = fps(xyz, self.num_query)
        #coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        #rebuild_points = torch.cat([rebuild_points, xyz],dim=1).contiguous()
        #print(rebuild_points)
        rebuild_points = self.decoder(global_feature, xyz, f_stage1, f_stage2, return_P0=False)
        ret = (coarse_point_cloud, rebuild_points)
        return ret

