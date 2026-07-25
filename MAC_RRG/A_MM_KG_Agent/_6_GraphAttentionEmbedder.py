import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

# from src.Config.train_sentences_config import TrainSentencesConfig
# from src.LLM.LLM import LLM
from AIU._5_Embedder import Embedder


#  GraphAttentionEmbedder 的类，
# 它是一个基于图注意力机制的嵌入模型，用于在图数据中进行节点嵌入学习。它继承自 Embedder 类
class GraphAttentionEmbedder(Embedder):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, num_layers, num_neighbors, activation='gelu', num_pseudo_words=1):
        super(GraphAttentionEmbedder, self).__init__(node_dim, edge_dim, hidden_dim, output_dim, num_layers, num_neighbors, activation)
        self.node_dim = node_dim # 节点特征的维度
        self.edge_dim = edge_dim #  边特征的维度
        self.d = num_pseudo_words # 伪词数量（即注意力头数）

        # Define separate linear transformations for each head
        # 为每个头定义单独的线性变换层（用于 Query, Key, Value 和输出）
        self.query_transforms = nn.ModuleList([nn.Linear(node_dim, node_dim) for _ in range(self.d)])
        self.key_transforms = nn.ModuleList([nn.Linear(node_dim, node_dim) for _ in range(self.d)])
        self.value_transforms = nn.ModuleList([nn.Linear(node_dim, node_dim) for _ in range(self.d)])
        self.output_transforms = nn.ModuleList([nn.Linear(node_dim, node_dim) for _ in range(self.d)])

        # 根据用户传入的激活函数选择激活层
        if self.activation == 'gelu':
            self.nonlinear_layer = nn.GELU
        elif self.activation == 'relu':
            self.nonlinear_layer = nn.ReLU
        elif self.activation == 'leaky_relu':
            self.nonlinear_layer = nn.LeakyReLU
        else:
            raise ValueError('Unknown activation')

        # Final neural network
        # 定义最终的神经网络（MLP），由多个线性层和激活函数组成
        self.final_network = nn.ModuleList()
        self.final_network.append(nn.Linear(node_dim, hidden_dim))  # 第一层：节点维度到隐藏层维度
        self.final_network.append(self.nonlinear_layer())  # 激活函数
        # self.final_network.append(nn.BatchNorm1d(hidden_dim))
        # 添加剩余的层
        for _ in range(num_layers - 1):
            self.final_network.append(nn.Linear(hidden_dim, hidden_dim))  # 每一层的隐藏层
            self.final_network.append(self.nonlinear_layer())  # 激活函数
            # self.final_network.append(nn.BatchNorm1d(hidden_dim))               # BatchNorm after each linear layer
        
        # 最后一层输出层
        self.final_network.append(nn.Linear(hidden_dim, output_dim))

# 义了模型的前向传播计算过程。
    def forward(self, central_node_features, neighbor_node_features, edge_features):
        """
        central_node_features: 形状为 (batch_size, 1, embedding_size) 的张量，表示中心节点的特征。
        :param central_node_features: Tensor(batch_size, 1, embedding_size)
        
        neighbor_node_features: 形状为 (batch_size, num_neighbors, embedding_size) 的张量，表示邻居节点的特征。
        :param neighbor_node_features: Tensor(batch_size, num_neighbors, embedding_size)
        
        edge_features: 形状为 (batch_size, num_neighbors, embedding_size) 的张量，表示连接中心节点和邻居节点的边的特征。
        :param edge_features: Tensor(batch_size, num_neighbors, embedding_size)
       
        子类应该在实现中根据需求返回一个形状为 (batch_size, embedding_size) 的张量，表示嵌入后的节点特征。
        :return: Tensor(batch_size, 1, embedding_size)
        """
        logging.debug(f"GraphAttentionEmbedder.central_node_features: {central_node_features.shape}")
        logging.debug(f"GraphAttentionEmbedder.neighbor_node_features: {neighbor_node_features.shape}")
        logging.debug(f"GraphAttentionEmbedder.edge_features: {edge_features.shape}")

        batch_size, num_neighbors, _ = neighbor_node_features.size() # 获取批大小和邻居数量

        multi_head_context = []  # 用于存储多个注意力头的输出

        # 对每个伪词（或注意力头）进行处理
        for i in range(self.d):
            # Prepare Query, Key, Value for each head
            # 准备每个头的查询（Query）、键（Key）和值（Value）
            Q = self.query_transforms[i](central_node_features)  # (batch_size, 1, node_dim)
            K = self.key_transforms[i](neighbor_node_features) + edge_features  # (batch_size, num_neighbors, node_dim)
            V = self.value_transforms[i](neighbor_node_features)  # (batch_size, num_neighbors, node_dim)

            # Attention mechanism for each head
            # 对每个头计算注意力机制
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.node_dim ** 0.5)
            logging.debug(f"GraphAttentionEmbedder.attention_scores: {attention_scores.shape}")

            # 计算每个邻居节点的注意力概率（softmax）
            attention_probs = F.softmax(attention_scores, dim=2)  # Softmax over neighbors
            logging.debug(f"GraphAttentionEmbedder.attention_probs: {attention_probs.shape}")

            # Weighted sum of values for each head
            # 基于计算出的注意力概率对值（V）进行加权求和，得到上下文表示
            context = torch.matmul(attention_probs, V)  # (batch_size, 1, node_dim)

            logging.debug(f"GraphAttentionEmbedder.context: {context.shape}")

            # Output transformation for each head
            # 对每个头的上下文进行输出变换
            head_output = self.output_transforms[i](context)
            logging.debug(f"GraphAttentionEmbedder.head_output: {head_output.shape}")

            # 将输出通过最终的神经网络（MLP）得到最终的输出
            output = head_output
            for module in self.final_network:
                output = module(output)

            logging.debug(f"GraphAttentionEmbedder.output: {output.shape}")

            multi_head_context.append(output)

        # Concatenate all head outputs
        # 将所有头的输出拼接起来，得到最终的多头上下文表示
        multi_head_context = torch.cat(multi_head_context, dim=1)  # (batch_size, d, node_dim)

        logging.debug(f"GraphAttentionEmbedder.multi_head_context: {multi_head_context.shape}")

        return multi_head_context

    # @staticmethod
    # def from_config(config: TrainSentencesConfig, llm: LLM):
    #     # 使用配置文件和 LLM 实例化一个 GraphAttentionEmbedder 对象
    #     return GraphAttentionEmbedder(
    #         node_dim=llm.embedding_length,
    #         edge_dim=llm.embedding_length,
    #         hidden_dim=int(llm.embedding_length * config.model_layer_width_multiplier),
    #         output_dim=llm.embedding_length,
    #         num_layers=config.model_layer_depth,
    #         num_neighbors=config.number_of_neighbors,
    #         activation=config.model_layer_activation,
    #         num_pseudo_words=config.num_pseudo_words,
    #     )