#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File: node2vec_ano.py
@Author: Wang Yang
@Date:   2024/12/15 17:17 
@Last Modified by: Wang Yang
@Descirption : node2vec for anomaly detection
'''
from typing import List, Optional, Tuple, Union

import torch
from torch_geometric.nn import Node2Vec
from torch import Tensor


class Node2Vec4AD(Node2Vec):

    def __init__(
        self,
        edge_index: Tensor,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        walks_per_node: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        num_nodes: Optional[int] = None,
        sparse: bool = False,
    ):
        super().__init__(edge_index, embedding_dim, walk_length, context_size,
                         walks_per_node, p, q, num_negative_samples, num_nodes, sparse)

    @torch.jit.export
    def anomaly_loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        pass

    def anomaly_score(self) -> Tensor:
        pass
