#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @Author: Peng Xiang

import torch
from torch import nn, einsum
from .utils_skip import MLP_Res, grouping_operation, query_knn


class SkipTransformer(nn.Module):
    def __init__(self, i,  in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(SkipTransformer, self).__init__()
        self.mlp_v = MLP_Res(in_dim=in_channel*2, hidden_dim=in_channel, out_dim=in_channel)
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )
        if i == 2:
            self.attn_mlp_N = nn.Sequential(
                nn.Conv2d(2048, 4096, 1),
                nn.BatchNorm2d(4096),
                nn.ReLU(),
                nn.Conv2d(4096, 2048, 1)
            )
        elif i == 1:
            self.attn_mlp_N = nn.Sequential(
                nn.Conv2d(512, 2048, 1),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.Conv2d(2048, 512, 1)
            )
        else:
            self.attn_mlp_N = nn.Sequential(
                nn.Conv2d(512, 2048, 1),
                nn.BatchNorm2d(2048),
                nn.ReLU(),
                nn.Conv2d(2048, 512, 1)
            )

        self.conv_end = nn.Conv1d(dim, in_channel, 1)

        self.layer1 = nn.Conv1d(dim*4, in_channel, 1)

    def forward(self, pos, key, query, include_self=True):
        """
        Args:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
            include_self: boolean

        Returns:
            Tensor: (B, in_channel, N), shape context feature
        """
        value = self.mlp_v(torch.cat([key, query], 1))
        identity = value
        key = self.conv_key(key)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = value.shape

        pos_flipped = pos.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped, include_self=include_self)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)

        ''''''
        qk_rel_N = qk_rel.permute(0, 2, 1, 3)
        pos_embedding_N = pos_embedding.permute(0, 2, 1, 3)
        attention_N = self.attn_mlp_N(qk_rel_N + pos_embedding_N) # b, n, dim, n_knn
        attention_N = torch.softmax(attention_N, -1)
        value_N = value
        value_N = value_N.reshape((b, -1, n, 1))
        value_N = value_N.permute(0, 2, 1, 3)
        #print('value_N', value_N.shape)
        #print('pos_embedding_N', pos_embedding_N.shape)
        value_N = value_N + pos_embedding_N
        #print('value_N', value_N.shape)
        #print('attention_N', attention_N.shape)
        agg_N = einsum('b c i j, b c i j -> b c i', attention_N, value_N) # b, n, dim
        agg_N = agg_N.permute(0, 2, 1) # b, dim, n
        y_N = self.conv_end(agg_N)


        ''''''
        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, dim, n, n_knn
        attention = torch.softmax(attention, -1)

        value = value.reshape((b, -1, n, 1)) + pos_embedding  #
        #print('value', value.shape)
        #print('attention', attention.shape)
        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.conv_end(agg)


        res = torch.cat((y_N, y), dim=1)
        #print('res.shape:', res.shape)
        res = self.layer1(res)
        #print('res.shape:', res.shape)
        #print('identity.shape:', identity.shape)

        #return y + identity
        return res + identity
