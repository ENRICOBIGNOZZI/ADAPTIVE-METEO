#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch_geometric.nn import GATv2Conv

class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_h, dim_out, heads=1):
    super().__init__()
    self.gat1 = GATv2Conv(in_channels=dim_in,out_channels=dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)

  def forward(self, x, edge_index):
    x = x.view(-1, x.size(-1))
    h = self.gat1(x, edge_index.permute(1,0))
    h = torch.nn.functional.elu(h)
    #h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index.permute(1,0))
    return h
