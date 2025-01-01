# -*- coding: utf-8 -*-
# todo 优化目标为 “adv2” + v2的同时决策。+ Critic估计的为“一辆uav路径”且LR是5倍
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # todo 记得修改不同的gpu编号
import time
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
print("优化目标为“一辆uav路径的长度”并且网络参数共享版本 + v2的同时决策+Critic估计的为“一辆uav路径 且LR是10倍”")
# from Greedy_VRP import run_greedy_VRP
if torch.cuda.is_available():  # 一定要放在import matplotlib.pyplot之前
    print("using cuda")
    device = torch.device("cuda")
    import matplotlib
    matplotlib.use('Agg')  # 防止尝试使用图形界面，允许在没有图形界面的环境中运行。
else:
    print("using cpu")
    device = torch.device("cpu")
import matplotlib.pyplot as plt

# %matplotlib inline


class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution.
    本质一维卷积（因为地点的坐标和顺序无关，所以用一维卷积当成embed工具，替代RNN编码器来用
    等价于 Conv1d，有hidden_size个大小为【1*input_size的】的卷积核"""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1) # 1*input_size的卷积核，有hidden_size个

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)


class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        # W processes features from static decoder elements
        self.W = nn.Parameter(torch.zeros((1, hidden_size, 3 * hidden_size),  # (1,H,3H)×(B,3H,L)=(B,H,L)
                                          device=device, requires_grad=True))

        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))  # (1,1,H)×tanh(B,H,L)=(B,1,L)=注意力a

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden):
        batch_size, hidden_size, _ = static_hidden.size()  # B H L

        hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)  # 这是小车当前位置的静态信息，本是1BH，扩展成BHL
        hidden = torch.cat((static_hidden, dynamic_hidden, hidden), 1)  # 拼接，B,3H,L

        # Broadcast some dimensions so we can do batch-matrix-multiply
        # 为了能够和batch进行相乘，进行expand操作。expand的参数是目标size，是以复制填充的方式实现的。
        v = self.v.expand(batch_size, 1, hidden_size)  # B,1,H
        W = self.W.expand(batch_size, hidden_size, -1)  # B,H,L

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, hidden)))
        attns = F.softmax(attns, dim=2)  # (batch, seq_len)=(B,L)=注意力a
        return attns


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings.
    使用GRU部分+指针，根据给定的上一个状态和input embed（当前位置静态信息的embed），
    返回所有节点作为下一个点的概率的分布，以及GRU当前隐状态ht"""

    def __init__(self, hidden_size, num_layers=1, dropout=0.2):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Used to calculate probability of selecting next state
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, 2 * hidden_size),
                                          device=device, requires_grad=True))

        # Used to compute a representation of the current decoder output
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.encoder_attn = Attention(hidden_size)  # 是注意力块。

        self.drop_rnn = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, static_hidden, dynamic_hidden, decoder_hidden, last_hh):
        rnn_out, last_hh = self.gru(decoder_hidden.transpose(2, 1), last_hh)  # 输出RNN output 和当前隐状态last_hh。
        rnn_out = rnn_out.squeeze(1)

        # Always apply dropout on the RNN output 对RNN输出进行dropout……
        rnn_out = self.drop_rnn(rnn_out)
        if self.num_layers == 1:  # 上面init函数里面规定，如果>1就自动drop out，如果=1则需要在这里手动drop out
            # If > 1 layer dropout is already applied
            last_hh = self.drop_hh(last_hh)

        # Given a summary of the output, find an  input context
        enc_attn = self.encoder_attn(static_hidden, dynamic_hidden, rnn_out)  # 在这里计算获得注意力a，B 1 L
        context = enc_attn.bmm(static_hidden.permute(0, 2, 1))  # 只和静态信息相乘！B 1 L × B L H  = B 1 H

        # Calculate the next output using Batch-matrix-multiply ops
        context = context.transpose(1, 2).expand_as(static_hidden)  # 转置成B H 1之后扩展成 B H L
        energy = torch.cat((static_hidden, context), dim=1)  # 和静态信息拼接在一起，(B, num_feats, seq_len) B H L

        v = self.v.expand(static_hidden.size(0), -1, -1)  # 第一维度扩展batch
        W = self.W.expand(static_hidden.size(0), -1, -1)  # 第一维度扩展batch

        probs = torch.bmm(v, torch.tanh(torch.bmm(W, energy))).squeeze(1)  # 得到还没有mask的概率，大小B L

        return probs, last_hh  # 返回每一个点被选取为下一个点的概率，以及隐状态


class DRL4TSP(nn.Module):
    """[注意这个类就是Actor]Defines the main Encoder, Decoder, and Pointer combinatorial models.

    Parameters:
    ----------
    static_size: int
        Defines how many features are in the static elements of the model
        (e.g. 2 for (x, y) coordinates)
    dynamic_size: int > 1
        Defines how many features are in the dynamic elements of the model
        (e.g. 2 for the VRP which has (load, demand) attributes. The TSP doesn't
        have dynamic elements, but to ensure compatility with other optimization
        problems, assume we just pass in a vector of zeros.
    hidden_size: int
        Defines the number of units in the hidden layer for all static, dynamic,
        and decoder output units.
    update_fn: function or None
        If provided, this method is used to calculate how the input dynamic
        elements are updated, and is called after each 'point' to the input element.
    mask_fn: function or None
        Allows us to specify which elements of the input sequence are allowed to
        be selected. This is useful for speeding up training of the networks,
        by providing a sort of 'rules' guidelines to the algorithm. If no mask
        is provided, we terminate the search after a fixed number of iterations
        to avoid tours that stretch forever
    num_layers: int
        Specifies the number of hidden layers to use in the decoder RNN
    dropout: float
        Defines the dropout rate for the decoder
    """

    def __init__(self, static_size, dynamic_size, hidden_size, car_load, depot_num,
                 update_fn=None, mask_fn=None, node_distance_fn=None, num_layers=1,
                 dropout=0.):  ##########################################################
        super(DRL4TSP, self).__init__()

        self.static_hidden = None
        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn  # 动态信息的更新函数
        self.mask_fn = mask_fn  # mask函数
        self.node_distance_fn = node_distance_fn

        # Define the encoder & decoder models [可训练的模块]
        self.static_encoder = Encoder(static_size, hidden_size)  # 静态信息embed编码器
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)  # 动态信息embed编码器
        self.decoder = Encoder(static_size, hidden_size)  # 当前位置信息编码器（为什么叫做decoder
        self.pointer = Pointer(hidden_size, num_layers, dropout)  # 指针网络（含GRU）

        self.car_load = car_load
        self.depot_num = depot_num

        for p in self.parameters():  # 对参数进行初始化。
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

        # Used as a proxy initial state in the decoder when not specified 这是小车的初始位置。

        # self.x0 = torch.zeros((1, static_size, 1), requires_grad=True, device=device)

    def forward(self, static, dynamic, decoder_input=None,
                last_hh=None):
        """
        Parameters
        ----------
        static: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the TSP, this could be
            things like the (x, y) coordinates, which won't change
        dynamic: Array of size (batch_size, feats, num_cities)
            Defines the elements to consider as static. For the VRP, this can be
            things like the (load, demand) of each city. If there are no dynamic
            elements, this can be set to None
        decoder_input: Array of size (batch_size, num_feats)
            Defines the outputs for the decoder. Currently, we just use the
            static elements (e.g. (x, y) coordinates), but this can technically
            be other things as well.
        last_hh: Array of size (batch_size, num_hidden)
            Defines the last hidden state for the RNN
        """
        # ---这是同时决策版本。------------------------------------------------------
        # 这里已经是坐标的形式了！！！
        batch_size, input_size, sequence_size = static.size()
        distance = self.node_distance_fn(static, self.depot_num)  # (B,num_node,num_node) # node 2 node 的距离
        if decoder_input is None:
            raise ValueError("DRL4TSP decoder input is None!")

        num_nodes = static.size(2)  # =总节点=city+depot
        mask = torch.ones(batch_size, sequence_size, device=device)  # 0是mask掉。

        #car_id = 0
        car_load = [torch.full((batch_size,), self.car_load) for _ in range(self.depot_num)]

        # 使用 arange 生成 depot 数字，然后扩展为 (depot_num, 1, batch_size)
        initial_depot = torch.arange(self.depot_num).view(self.depot_num, 1, 1).expand(self.depot_num, 1,
                                                                                       batch_size).tolist()
        # 给每个无人机tour里记录初始仓库。tour idx 的大小是num_car,tour length,B,
        tour_idx = [[torch.tensor(initial_depot[i][0]).unsqueeze(1).to(device)] for i in
                    range(self.depot_num)]  # num_car, 1, batch_size
        #tour_logp = []# 最终的访问序列。
        tour_logp =[[] for _ in range(self.depot_num)] # 为了把优化目标改为每个无人机的路径，需要添加维度

        # # ptr 大小B 1。
        # current_idx = tour_idx[0][-1].clone().detach()  # 当前第0辆无人机的位置。从第一辆无人机开始，取最后一个（其实只有一个元素）所在下标。

        max_steps = sequence_size if self.mask_fn is None else 500  # 如果设置mask函数，为了避免死循环，这是最大步数。

        self.static_hidden = self.static_encoder(static) # 只需要计算一次

        probs_home= [None] * self.depot_num
        mask_home = [None] * self.depot_num
        for _ in range(max_steps):  # 主循环
            tem_next_ptr=[] # 维度是list:num_depot,内部元素(B,)
            tem_next_logp=[]# 维度是list:num_depot,内部元素(B,)

            # 所有batch的里面，uav归位+所有city无需求。退出
            if dynamic.data[:, 1].eq(0).all():
                break

            for car_id in range(self.depot_num):

                # 替换dynamic里的load！把load换成下一个无人机的load，但demand不变，因为是同一个环境。。
                dynamic = dynamic.clone()
                dynamic[:, 0] = car_load[car_id].unsqueeze(1).expand(-1, num_nodes) # safe

                # current_idx 更新,car id 更新。
                current_idx = tour_idx[car_id][-1].clone().detach()# 当前所在id

                if self.mask_fn is not None:
                    # 根据当前无人机结束新的访问，结合下一台无人机的.
                    mask = self.mask_fn(self.depot_num, dynamic, distance, current_idx.data, car_id).detach()
                if not mask.byte().any():  # 如果全mask掉了就退出
                    break

                # 更新decoder_input当前位置xy，(B, 2)
                decoder_input = torch.gather(static, 2, current_idx.view(-1, 1, 1).expand(-1, input_size, 1).to(
                    dynamic.device)).detach()

                dynamic_hidden = self.dynamic_encoder(dynamic)  # 更新编码当前动态信息
                decoder_hidden = self.decoder(decoder_input)  # 更新编码当前位置xy静态信息

                # 指针网络。里面包含GRU。
                probs, last_hh = self.pointer(self.static_hidden, dynamic_hidden, decoder_hidden,
                                              last_hh)  # 得到下一个点的（未mask）概率分布和隐状态。

                probs_home[car_id] = probs
                mask_home[car_id] = mask
                probs = F.softmax(probs + mask.log(), dim=1)  # mask操作+softmax Softmax的原因：因为π(a,s)是必须大于0的，用softmax把0映射成很小的数字

                if self.training:
                    try:
                        m = torch.distributions.Categorical(probs)
                    except:
                        raise ValueError("Error: m = torch.distributions.Categorical(probs)")
                    # Sometimes an issue with Categorical & sampling on GPU; See:
                    # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                    ptr = m.sample()  # 根据上面的概率分布，采样一个点。大小B

                    while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                        ptr = m.sample()

                    logp = m.log_prob(ptr) # 记录这个点的概率……# 是因为在这里用ptr获得了logp，所以后续直接修改ptr会报错。
                else:
                    prob, ptr = torch.max(probs, 1)  # Greedy
                    logp = prob.log()  # B,1

                # 记录这个无人机选的点ptr 和logp……………………
                tem_next_ptr.append(ptr)
                tem_next_logp.append(logp)

                # -------------------结束for every uav循环。-----------------

            # 快速跳转用
            # ---------处理冲突----------------
                duplicates_mask = self.get_duplicate_mask(tem_next_ptr, tem_next_logp).detach()  # tensor(B,num_depot)
                tem_count = 0
                while duplicates_mask.any():
                    if tem_count > 100:
                        print("duplicate", tem_count)
                    tem_count += 1
                    # 找出本行里面是否有重复元素
                    if duplicates_mask.any():  # 如果有，遍历选到重复点的car：
                        # √ 在duplicates_mask里面为True的元素就是 “ptr重复且logp不是同数字的top1的元素”
                        if tem_count > 100:
                            print("row_ptr修改前：", tem_next_ptr)
                        for duplicate_car in duplicates_mask.any(dim=0).nonzero():  # 纵轴比较，找出有重复点的car id
                            duplicate_car = duplicate_car.item()  # duplicate_car是本行中需要重新选点的car id（等价于下标）
                            new_ptr, new_logp = self.get_next_ptr_by_mask2_whole(tem_next_ptr, duplicates_mask,
                                                                                 duplicate_car, mask_home, probs_home,
                                                                                 tour_idx)
                            mask_car = duplicates_mask[:, duplicate_car]  # duplicates_mask形状B，num_car
                            tem_next_ptr[duplicate_car] = tem_next_ptr[duplicate_car] * torch.logical_not(
                                mask_car) + new_ptr * mask_car
                            tem_next_logp[duplicate_car] = tem_next_logp[duplicate_car] * torch.logical_not(
                                mask_car) + new_logp * mask_car
                        if tem_count > 100:
                            print("row_ptr修改后：", tem_next_ptr)
                    duplicates_mask = self.get_duplicate_mask(tem_next_ptr, tem_next_logp)  # 刷新重复元素

            # ---------处理冲突结束----------------

            # -------------for every uav循环：储存本轮的访问计划、更新地图-----------------
            for car_id in range(self.depot_num):
                # 读取当前位置；更新dynamic；保存现在的load；换成下一个车的load；更新isdone；（ptr和logp保存过了）；更新下一车ptr
                # 选择好了下一个访问的点，访问，更新动态信息。
                if self.update_fn is not None:
                    last_visited = tour_idx[car_id][-1] # 获取当前位置
                    # 获得下一步的点（已解决冲突）
                    ptr = tem_next_ptr[car_id].clone().detach()  # 无人机下一步去往的id
                    # 把本uav的load装进dynamic
                    dynamic=dynamic.clone() #(B,2,num_node) 【反向传播应该没问题，就是这样的。】
                    dynamic[:, 0] = car_load[car_id].unsqueeze(1).expand(-1, num_nodes)

                    # 更新dynamic。B 2 L 这里还是当前车的load和新一步ptr，更新地图里的load和demand
                    dynamic = self.update_fn(self.depot_num, self.car_load, dynamic, distance, ptr.data,
                                             last_visited)

                    # dynamic中的旧load储存，更新新load。所以此后dynamic都是当前无人机访问下一个点后的新环境
                    car_load[car_id] = dynamic[:, 0, 0].clone().detach()  # 随便取第一个仓库的load就好了，存到car_load数组里面。

                    # 意思应该是batch里面有些情况下，已经遍历完了，就让车强制留在仓库）
                    is_done = dynamic[:, 1].sum(1).eq(0).float()  # 如果所有点的需求加和=0，就说明done
                    logp= tem_next_logp[car_id]
                    logp = logp * (1. - is_done)  # 如果完成了，logp 也是0, 梯度就不会更新了

                    # 每个无人机的logp和ptr是单独存放，以便单独反向传播。
                    #tour_logp.append(logp.unsqueeze(1))  # 每个时间t都要储存：因为为了计算整条路径出现的概率. T B 1
                    tour_logp[car_id].append(logp.unsqueeze(1)) # 为了把优化目标改为每个无人机的路径，需要添加维度
                    tour_idx[car_id].append(ptr.data.unsqueeze(1))  # T B 1 把当前无人机的新访问的点保存起来。

        else:
            print(f"达到最大迭代次数{max_steps}退出")

        if (dynamic[:,1,:]>0).any():# 如果仍然有需求
            a=torch.nonzero(dynamic[:,1,:])
            print(a)
            raise ValueError("仍然有需求尚未满足：",a)
        else:
            # print("完成所有需求。")
            pass

        # tour_idx 大小：最外层是list，包含num depot 个元素，每个元素是batch*无人机飞行过node个数。
        tour_idx = [torch.cat(tour_idx[i], dim=1) for i in range(self.depot_num)]  # 包含了每一辆无人机的轨迹【这个不用改】
        # tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)
        tour_logp = [torch.cat(tour_logp[i], dim=1) for i in range(self.depot_num)] #外层是list，包含num depot 个元素(batch_size, seq_len)
        return tour_idx, tour_logp

    # def get_next_ptr_by_mask(self,row_logp, row_ptr, duplicates_mask, car_id, mask_home, probs_home, batch_id,tour_list):
    #     '''
    #     在这里面更新mask、、、结合原始的probs分布重新选一个点？
    #     mask:旧的mask【看来又要mask home了或者时间换空间
    #     返回值：最新的单个batch、单个飞机的ptr和logp
    #     '''
    #     probs=probs_home[car_id][batch_id].clone() # (num_node,)
    #     mask=mask_home[car_id][batch_id].clone() # 目前这是一个元素……# # (num_node,)
    #
    #     # 用旧的mask叠加现在新的需要mask掉的重复点。
    #     duplicates_mask_row = duplicates_mask[batch_id]  # duplicates_mask True为重复元素
    #     visited_id = row_ptr[torch.logical_not(duplicates_mask_row)]  # 标记处那些点是非重复元素。（把非重复元素作为访问过的点，mask掉，
    #     mask[visited_id] = False  # 把其他人访问过的改为False
    #
    #     if not mask.byte().any():   # 是因为在最后阶段，，其他点都被访问了，所以不得不【留在仓库】
    #         current_pos = tour_list[car_id][-1][batch_id].clone().detach()  # 当前所在id
    #         mask[current_pos]=True
    #
    #     if not mask.byte().any():  # 如果全mask掉了就退出
    #         raise ValueError(f"not mask.byte().any() in get_next_ptr()：{mask}")
    #
    #     probs = F.softmax(probs + mask.log(), dim=0)
    #     # probs = F.softmax(probs, dim=1) + mask.log() # todo？有没有更合理的？【使用原始prob的概率会如何？】
    #
    #     if self.training:
    #
    #         try:
    #             m = torch.distributions.Categorical(probs)
    #         except:
    #             raise ValueError("Error: m = torch.distributions.Categorical(probs)")
    #         ptr = m.sample()  # 根据上面的概率分布，采样一个点。大小B
    #
    #         while not torch.gather(mask, 0, ptr.data.unsqueeze(0)).byte().all():
    #             ptr = m.sample()
    #
    #         logp = m.log_prob(ptr)  # 记录这个点的概率……
    #
    #     else:
    #         # prob, ptr = torch.max(probs, 1)  # Greedy
    #         prob, ptr = torch.max(probs,0)
    #         logp = prob.log()  # 1
    #         # print(f"self.test in get_next_ptr_by_mask choose {ptr} with prob {prob}")
    #
    #     return ptr,logp
    def get_next_ptr_by_mask2_whole(self, ptr_matrix, duplicates_mask, car_id, mask_home, probs_home, tour_list):
        '''
        比起旧的函数，只是多了一个B的维度……
        使用新mask结合原始的probs分布重新选一个点
        mask:旧的mask
        返回值：最新的【整个】batch、【所有】飞机的ptr和logp
        '''
        probs = probs_home[car_id].clone()  # (B,num_node)
        mask = mask_home[car_id].clone()  # # (B,num_node)
        ptr_matrix_tran = torch.stack(ptr_matrix).detach()  # tensor(B,num_depot)
        # 用旧的mask叠加现在新的需要mask掉的重复点。
        for batch_id in range(probs.size(0)):
            if not duplicates_mask[batch_id].any():
                continue
            # print(f"batch_id={batch_id}")
            duplicates_mask_row = duplicates_mask[batch_id]  # duplicates_mask True为重复元素
            row_ptr = ptr_matrix_tran[:, batch_id].clone().detach()
            visited_id = row_ptr[torch.logical_not(duplicates_mask_row)]  # 标记处那些点是非重复元素。（把非重复元素作为访问过的点，mask掉，
            mask[batch_id][visited_id] = False  # 把其他人访问过的改为False

            if not mask[batch_id].byte().any():  # 是因为在最后阶段，，其他点都被访问了，所以不得不【留在仓库】
                current_pos = tour_list[car_id][-1][batch_id].clone().detach()  # 当前所在id
                mask[batch_id][current_pos] = True

            if not mask[batch_id].byte().any():  # 如果全mask掉了就退出
                raise ValueError(f"not mask[{batch_id}].byte().any() in get_next_ptr()：{mask}")

        probs = F.softmax(probs + mask.log(), dim=1)  # (B,num_node)
        # probs = F.softmax(probs, dim=1) + mask.log() # fixme ？？？？？？？有没有更合理的？不知道如果使用这种会怎么样……

        if self.training:
            try:
                m = torch.distributions.Categorical(probs)
            except:
                raise ValueError("Error: m = torch.distributions.Categorical(probs)")
            ptr = m.sample()  # 根据上面的概率分布，采样一个点。大小B

            while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                ptr = m.sample()

            logp = m.log_prob(ptr)  # 记录这个点的概率……todo 还有一个不合理的地方是：这个去的logp不是旧的probs的logp……呃呃！

        else:
            # prob, ptr = torch.max(probs, 1)  # Greedy
            prob, ptr = torch.max(probs, 1)
            logp = prob.log()  # 1
            # print(f"self.test in get_next_ptr_by_mask choose {ptr} with prob {prob}")

        return ptr, logp

    def get_duplicate_mask(self,next_ptr_matrix, next_logp_matrix):

        next_ptr_matrix = torch.transpose(torch.stack(next_ptr_matrix), 1, 0).detach()  # tensor(B,num_depot)
        next_logp_matrix = torch.transpose(torch.stack(next_logp_matrix), 1, 0).detach()  # tensor(B,num_depot)

        # 创建一个和 tensor 同样大小的布尔 tensor，
        duplicates_mask = torch.zeros_like(next_ptr_matrix, dtype=torch.bool)
        for i in range(next_ptr_matrix.size(0)):
            ptr_row = next_ptr_matrix[i]
            logp_row = next_logp_matrix[i]
            unique_elements, counts = torch.unique(ptr_row, return_counts=True)  # 统计每行元素的出现次数
            duplicate_elements = unique_elements[counts > 1]  # 找到重复的元素 (出现次数大于 1)
            for element in duplicate_elements:  # 对于每一种重复元素，标记为 True
                duplicates_this_num = (ptr_row == element)  # 找到本重复元素
                max_logp = torch.max(logp_row[duplicates_this_num])  # 最大的logp
                max_logp_car = (logp_row == max_logp) & duplicates_this_num  # 找到最大元素的位置 （bool）
                if sum(max_logp_car) != 1:
                    max_logp_car[max_logp_car.nonzero()[1:]] = False
                duplicates_mask[i] = duplicates_mask[i] | (
                        duplicates_this_num & (torch.logical_not(max_logp_car)))  # 只有logp并非最大且重复元素需要True

        return duplicates_mask


from torch.utils.data import Dataset


class VehicleRoutingDataset(Dataset):
    def __init__(self, num_samples, num_city, max_load=10, car_load=0., max_demand=1., seed=None,
                 num_depots=-1):  # 增加了 num_depots参数 ###
        super(VehicleRoutingDataset, self).__init__()

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.num_samples = num_samples
        self.num_depots = num_depots  # 保存仓库数量
        #self.max_load = max_load  # 原本max load是未归一化的，carload是归一化的1
        self.max_demand = max_demand

        # 修改位置生成逻辑以支持多仓库地图
        # locations = torch.rand((num_samples, 2, input_size + 1))
        self.static = torch.rand((num_samples, 2, num_city + self.num_depots))  # 需要生成飞机数量+city数量个节点（前面的是飞机）
        self.car_load = car_load  # 这个是回仓库的时候恢复的值。含义上是 maxload

        # 所有状态都将广播司机当前的载重量
        # 注意，我们只在 [0, 1] 范围内使用载重量，以防止大数字输入神经网络
        # dynamic_shape = (num_samples, 1, input_size + 1)
        dynamic_shape = (num_samples, 1, num_city + self.num_depots)
        # loads = torch.full(dynamic_shape, 5.)
        loads = torch.full(dynamic_shape, self.car_load)

        # demands = torch.randint(1, self.max_demand + 1, dynamic_shape)
        # demands = demands / float(self.max_load)  # 取消归一化。
        demands = torch.full(dynamic_shape, self.max_demand)

        # # # 设置仓库的需求量为 0
        # for depot_idx in range(self.num_depots):
        #     demands[:, 0, depot_idx] = 0
        # 所有仓库的需求量设置为 0.测试一下效率优化。
        demands[:, 0, :self.num_depots] = 0  # 所有仓库的需求量设置为 0

        # self.dynamic = torch.cat((loads, demands), dim=2)
        self.dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 返回 (静态信息, 动态信息, 起始位置)
        return self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1]


def node_distance_shared(static, num_depots):
    '''
   是【完全共享仓库】版本.（可以在别人的仓库充电、最终停放）
    修改了：目前版本是返回【点a-点b-点b最【远】的仓库】因为存在最近仓库被占用的情况。
    static的维度:应该是B，2，num_depots+num_city
    '''
    depot_positions = static[:, :, :num_depots]  # 维度应该是B，2，num_depots
    city_positions = static[:, :, num_depots:]  # 维度应该是B，2，num_city
    depot_positions_expanded = depot_positions.unsqueeze(2).expand(-1, -1, city_positions.size(2),
                                                                   -1)  # B,2,num_city, num_depots
    distances_to_depot = torch.sqrt(
        torch.sum((city_positions.unsqueeze(3) - depot_positions_expanded) ** 2, dim=1))  # B,num_city,num_depots
    # 取每行最小值作为每个节点到最近仓库的距离
    min_distances2depot, _ = torch.min(distances_to_depot, dim=2)  # B,num_city,
    min_distances2depot = torch.cat(  # 拼回 B，num_depot+ numcity的大小（因为仓库到最近的仓库距离=0，所以直接用zero矩阵）
        (torch.zeros(min_distances2depot.size(0), num_depots).to(static.device), min_distances2depot), dim=1)  #
    ######### 改成最远仓库
    max_distances2depot, _ = torch.max(distances_to_depot, dim=2)
    max_distances2depot = torch.cat(  # 拼回 B，num_depot+ numcity的大小（因为仓库到最近的仓库距离=0，所以直接用zero矩阵）
        (torch.zeros(max_distances2depot.size(0), num_depots).to(static.device), max_distances2depot), dim=1)  #

    # 计算所有节点之间的距离 【需求是：每一个点到下一个点+下一个点回仓库。所以我需要的是：n2n 和 n2depot】
    distances_between_node = torch.sqrt(
        torch.sum((static.unsqueeze(2) - static.unsqueeze(3)) ** 2, dim=1))  # 计算欧式距离
    distances = torch.cat((distances_between_node, max_distances2depot.unsqueeze(1)), dim=1)  # 前面是n2n，后面是n2depot
    # distances大小为 (betch_size,seq_len+1,seq_len)加上了最近仓库张量
    return distances


def update_mask_shared(num_depots, dynamic, distance, current_idx, car_id):
    """更新用于隐藏非有效状态的掩码。用于【共享仓库】的功能

    参数
    ----------
    dynamic: torch.autograd.Variable 的大小为 (1, num_feats, seq_len)
    distance: 在这里是node distance 1函数给的距离矩阵。-1位置存放着每个点到其最近仓库的距离。
    chosen_idx:[非常重要] 是当前的无人机的坐标。需要根据当前坐标，mask下一个可能的点。（已经改名为current_idx）
    car_id:并没有用。只是为了对齐update_mask2_outside的参数。
    """

    # 将浮点数转换为整数进行计算
    loads = dynamic.data[:, 0]  # (batch_size, seq_len)
    demands = dynamic.data[:, 1]  # (batch_size, seq_len)
    n2n2depot_dis = distance + distance[:, -1, :].unsqueeze(1)  # 节点之间的距离加上与最近仓库的距离。
    n2n2depot_dis = n2n2depot_dis[:, :-1, :]  # 第二维度的意思：前面的值是节点之间的距离加上与最近仓库的距离。

    # 计算 current_idx 到仓库点的距离【注意：共享仓库】
    # depot_distances = distance[torch.arange(distance.size(0)), current_idx.squeeze(1), :num_depots]
    # 根据batch，取当前位置-其他所有点的距离。
    chosen_distance = n2n2depot_dis[torch.arange(distance.size(0)), current_idx.squeeze(1)]
    # chosen_distance[:, :num_depots] = depot_distances # 什么无效代码。

    # 如果没有剩余的正需求量，我们可以结束行程。

    if demands.eq(0).all():  # 即所有batch的里面，地图里面每一个点都没有需求了. 外界检测到全0的mask会退出vrp流程。
        return demands * 0.

    # 这个 demand ne 0 会筛选出：所有有需求的city+所有空的仓库节点
    #  第二项三项筛选出并且loads-chosen_distance需要大于0的city【虽然这一条会涉及到仓库，但是先假设后续会单独对仓库进行处理，这里怎么样都不管】
    new_mask = demands.ne(0) * demands.lt(loads - chosen_distance) * (loads - chosen_distance > 0)

    # 我们应该避免连续两次前往仓库
    # at_depot = chosen_idx.ne(0)
    # if at_depot.any(): # 不在仓库的可以回去。
    #     new_mask[at_depot.nonzero(), 0] = 1.
    # if (~at_depot).any(): # 在仓库的不能继续访问仓库
    #     new_mask[(~at_depot).nonzero(), 0] = 0.
    ############################# 避免连续两次前往仓库（"如果在仓库，下一个就不可以访问任何仓库"）
    at_depot = current_idx < num_depots
    if at_depot.any():
        new_mask[at_depot.squeeze().nonzero(), :num_depots] = 0

    ##############################

    # ...除非我们在等待小批量中的其他样本完成
    # has_no_load = loads[:, 0].eq(0).float() # 仓库load=0 说明无人机归位。
    # has_no_demand = demands[:, 1:].sum(1).eq(0).float() # 这里的1要改/所有city都没有demand
    has_no_demand = demands[:, num_depots:].sum(1).eq(0).float()  # 所有city都没有demand，转1和0 【避免本无人机在仓库但是其他无人机还没回去）

    # combined = (has_no_load + has_no_demand).gt(0) # combine zero：该样本有的 city有demand 并且 车load不等于0
    combined = has_no_demand.gt(0)  # combined应该是B 1 吧？
    if combined.any():  # 如果该batch里面存在city没有demand（也就是说有的batch结束了，需要让无人机允许留在原地）
        # 首先，我们将所有节点的掩码设置为0，防止访问
        # new_mask[combined.nonzero(), :] = 0.
        # 对于每个样本，如果它已经在仓库中，我们只允许它访问当前所在的仓库
        # for sample_idx in combined.nonzero().squeeze():
        for sample_idx in combined.nonzero():  # 找到全部无需求的batch id
            current_location = current_idx[sample_idx]  # 找到当前位置
            if current_location < num_depots:  # 如果当前在仓库
                # 仅允许访问当前所在的仓库
                # new_mask[combined.nonzero(), :] = 0.
                new_mask[sample_idx, current_location] = 1.
            # else:
            #     # 如果不在仓库，但没有需求或者载重为0，则允许访问【demand不为0的仓库】
            #     #new_mask[sample_idx, :self.num_depots] = 1.
            #     new_mask[sample_idx * torch.ones_like(demands[sample_idx], dtype=torch.long), demands[sample_idx] != 0] = 1.

    # 判断是否存在某一行的mask全为0 ####################
    all_zero_mask = (new_mask == 0).all(dim=1)
    if all_zero_mask.any():
        # 找到全为0的行的索引
        all_zero_index = all_zero_mask.nonzero(as_tuple=True)[0]
        # 检查是否是留在仓库。如果是留在城市就报错
        if any(current_idx[all_zero_index] >= num_depots):
            raise ValueError(f"uav {car_id} 留在原地:{current_idx[all_zero_index]}")
        # 将这些行中chosen_idx对应位置的mask设为1
        new_mask[all_zero_index, current_idx[all_zero_index]] = 1  # 因为存在最远的有需求的城市正好去不了的情况

    return new_mask.float()


def update_dynamic_shared(num_depots, max_car_load, dynamic, distance, next_idx, current_idx):  # 加了参数：访问的前一个点。
    """
    用于更新当前地图的dynamic的函数。啊要用到distance是因为dynamic里面的load需要减去距离……
    这个函数是【共享仓库】版本
    """
    # print("update_dynamic:网络预测下一步next_idx 是：\n",next_idx)
    # print("update_dynamic:当前所在位置current_idx 是：\n",current_idx)
    """更新 (load, demand) 的值。"""
    current_idx = current_idx.squeeze()
    # 根据是访问仓库还是城市，以不同方式更新动态元素
    ##############################
    visit = next_idx.ge(num_depots)  # 访问的是城市还是仓库
    depot = next_idx.lt(num_depots)
    # 如果 chosen_idx 小于 num_depots，则表示访问的是仓库
    ##############################

    # 克隆动态变量，以免破坏图
    all_loads = dynamic[:, 0].clone()
    all_demands = dynamic[:, 1].clone()
    load = torch.gather(all_loads, 1, next_idx.unsqueeze(1))  # 获得batch里每一个样本，下一个节点的load
    demand = torch.gather(all_demands, 1, next_idx.unsqueeze(1))  # 获得batch里每一个样本，下一个节点的demand
    distance_matrix = distance[:, :-1, :]  # 距离矩阵 取到-1是因为最后一个是“每个点到最远的仓库的距离
    # 在小批量中 - 如果我们选择访问一个城市，尽可能满足其需求量
    if visit.any():
        diff_distances = distance_matrix[
            torch.arange(distance_matrix.size(0)), current_idx, next_idx.squeeze()].unsqueeze(1)

        # 检查上一次选择的节点与这次选择的节点的差值
        check_load = load - demand - diff_distances
        if (check_load < 0).any():
            print(check_load)
            raise ValueError("Error: 存在负载为负数.")

        # 上一次选择的节点与这次选择的节点的差值
        new_load = torch.clamp(check_load, min=0)

        check_demand = demand - load + diff_distances
        if (check_demand>0).any():
            raise ValueError("Error:无法满足下一个城市的需求。")
        new_demand = torch.clamp(check_demand, min=0)

        # 将载重量广播到所有节点，但单独更新需求量
        visit_idx = visit.nonzero().squeeze()

        all_loads[visit_idx] = new_load[visit_idx]
        all_demands[visit_idx, next_idx[visit_idx]] = new_demand[visit_idx].view(-1)
        # all_demands[visit_idx, 0] = -1. + new_load[visit_idx].view(-1) # 改了

    # -----------测试把上一个访问节点是仓库的时候，条件扩展为"当前访问节点可以是任何点（即允许连续两次访问仓库）
    # 使用布尔索引来找出上一个访问的是仓库的样本
    # is_depot = last_visited.lt(self.num_depots).to('cuda')
    depot_visited_idx = current_idx.lt(num_depots).to(dynamic.device)

    # depot_visited_idx = is_depot # 找出同时访问城市且上一次访问的是仓库的样本（不用了，需要可以连续访问仓库）

    # 1，原始的visit idx是【数组下标】不是城市序号！！
    depot_visited_idx = depot_visited_idx.nonzero().squeeze()

    # 使用布尔索引和高效的张量操作来更新all_demands
    # all_demands[depot_visited_idx.to('cuda'), last_visited.to('cuda')[depot_visited_idx]] = -1. #+ new_load[depot_visited_idx].reshape(-1,2)
    all_demands[depot_visited_idx.to(dynamic.device), current_idx.to(dynamic.device)[
        depot_visited_idx]] = -1.  # + new_load[depot_visited_idx].reshape(-1,2)

    # 返回仓库以填充车辆载重量
    ##############################
    # if depot.any():
    #     all_loads[depot.nonzero().squeeze()] = 1.
    #     all_demands[depot.nonzero().squeeze(), 0] = 0.
    if depot.any():
        # all_loads[depot.nonzero().squeeze()] = float(self.max_load)
        all_loads[depot.nonzero().squeeze()] = float(max_car_load)
        depot_indices = next_idx[depot.squeeze()]
        all_demands[depot.squeeze(), depot_indices] = 0.

    new_dynamic = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
    # return torch.tensor(tensor.data, device=dynamic.device)
    # 避免额外的计算开销和不必要的内存使用
    return new_dynamic.clone().detach().to(device=dynamic.device)


def node_distance_independent(static, num_depots):
    """
    static: 所有点（包括仓库和tower）的坐标
    num_depots: 没有用，是为了对齐node_distance_shared 的参数。
    本函数：返回每个点到另一个点之间的距离。
    之前的版本：return 的维度是B，num_node+1,num_node.属于是强行在第二个维度的最后面上添加了一个大小为num_node的矩阵……
    所以返回值的[b][-1][i]表示在第b个batch里面，第i个node离最近的仓库的距离。（如果第i是仓库则距离=0）
    """

    # depot_positions = static[:, :, :self.num_depots]# 维度应该是B，2，num_depots
    # city_positions = static[:, :, self.num_depots:]# 维度应该是B，2，num_city
    # depot_positions_expanded = depot_positions.unsqueeze(2).expand(-1, -1, city_positions.size(2), -1)# B,2,num_city, num_depots
    # # 计算每个节点到最近仓库的距离
    # distances_to_depot = torch.sqrt(torch.sum((city_positions.unsqueeze(3) - depot_positions_expanded) ** 2, dim=1))# B,num_city,num_depots
    # # 取每行最小值，作为每个city节点到最近仓库的距离 这个应该不用再取最小值了，这行要删掉。
    # min_distances, _ = torch.min(distances_to_depot, dim=2) # (B,num_city)
    # min_distances = torch.cat( # 这一行本意是在前面加上“每个仓库距离最近的仓库（自己）距离为0”的意思）
    #     (torch.zeros(min_distances.size(0), self.num_depots).to(static.device), min_distances), dim=1)

    # 计算所有节点之间的距离
    distances_between_node = torch.sqrt(
        torch.sum((static.unsqueeze(2) - static.unsqueeze(3)) ** 2, dim=1))  # 计算欧式距离
    # distances = torch.cat((distances_between_node, min_distances.unsqueeze(1)), dim=1)

    return distances_between_node  # distances大小为 (betch_size,seq_len+1,seq_len).加上了最近仓库张量


def update_mask_independent(num_depots, dynamic, n2n_distance, current_idx, car_id):
    """和上一个相比是只允许无人机返回自己的仓库。

    dynamic: torch.autograd.Variable 的大小为 (1, num_feats, seq_len)
    n2n_distance：每两个点之间的距离。
    chosen_idx:[非常重要] 大小(B,1)是当前的无人机的坐标。需要根据当前坐标，mask下一个可能的点。
    """
    # 将浮点数转换为整数进行计算
    loads = dynamic.data[:, 0]  # (batch_size, seq_len)
    demands = dynamic.data[:, 1]  # (batch_size, seq_len)

    '''
    9.28修改：这里只考虑mask city的逻辑：(就算是影响到仓库也没关系，后续会处理仓库进行覆盖。)
        找到当前的位置，并且取出当前点-所有点的距离+所有点回自己的仓库（已经给了carid）的距离 
        （所以我需要node-node 矩阵就够了。因为可以转换成：D(当前点~所有点) +D(自己仓库~所有点)
    '''
    # n2n_distance 是维度为(B,node_num,node_num)的
    dis_cur2all = n2n_distance[torch.arange(n2n_distance.size(0)), current_idx.squeeze(1)]
    dis_depot2all = n2n_distance[torch.arange(n2n_distance.size(0)), car_id]  # 仓库id=当前car id

    dis_cur2node2depot = dis_cur2all + dis_depot2all  # D(当前点~所有点) +D(自己仓库~所有点) 总距离

    # 如果没有剩余的正需求量，我们可以结束行程。
    if demands.eq(0).all():  # 即所有batch的里面，地图里面每一个点都没有需求了：
        return demands * 0.

    # 这个 demand ne 0 会筛选出：所有有需求的city+所有空的仓库节点
    # 第二项三项筛选出并且loads-chosen_distance-demand大于0的city【虽然这一条会涉及到仓库，但是先假设后续会单独对仓库进行处理，这里怎么样都不管】
    new_mask = demands.ne(0) * demands.lt(loads - dis_cur2node2depot) * (loads - dis_cur2node2depot > 0)

    ##############################
    # 9.28修改方案：任何时刻兜底把所有仓库mask掉.
    new_mask[:, :num_depots] = 0
    # 然后判断让不在仓库的可以回到自己的仓库。
    in_city = (current_idx >= num_depots)
    new_mask[in_city.squeeze(), car_id] = 1

    # ...除非我们在等待小批量中的其他样本完成
    # has_no_load = loads[:, 0].eq(0).float() # 仓库load=0 说明无人机归位。
    # has_no_demand = demands[:, 1:].sum(1).eq(0).float() # 这里的1要改/所有city都没有demand
    has_no_demand = demands[:, num_depots:].sum(1).eq(0).float()  # 所有city都没有demand，转1和0 【避免本无人机在仓库但是其他无人机还没回去）

    # combined = (has_no_load + has_no_demand).gt(0) # combine zero：该样本有的 city有demand 并且 车load不等于0
    combined = has_no_demand.gt(0)  # combined应该是B 1 吧？
    if combined.any():  # 如果该batch里面存在city没有demand（也就是说有的batch结束了，需要让无人机允许留在原地）
        # 首先，我们将所有节点的掩码设置为0，防止访问
        # new_mask[combined.nonzero(), :] = 0.
        # 对于每个样本，如果它已经在仓库中，我们只允许它访问当前所在的仓库
        # for sample_idx in combined.nonzero().squeeze():
        for sample_idx in combined.nonzero():  # 找到全部无需求的batch id
            current_location = current_idx[sample_idx]  # 找到当前位置
            if current_location < num_depots:  # 如果当前在仓库
                # 仅允许访问当前所在的仓库
                new_mask[sample_idx, current_location] = 1. # 其实current_location等价于car id


    # 判断是否存在某一行的mask全为0####################
    all_zero_mask = (new_mask == 0).all(dim=1)
    if all_zero_mask.any():
        # 以下是打底的补丁，无补丁会报错【因为存在最远的有需求的城市正好去不了的情况】
        # 找到全为0的行的索引
        all_zero_indices = all_zero_mask.nonzero(as_tuple=True)[0]
        if any(current_idx[all_zero_indices] >= num_depots):  # 检查是否是留在仓库。如果是留在城市就报错（
            raise ValueError(f"uav {car_id} 留在城市:{current_idx[all_zero_indices]}")

        # 将这些行中chosen_idx(当前位置即仓库位置？）对应位置的mask设为1
        new_mask[all_zero_indices, current_idx[all_zero_indices]] = 1

    return new_mask.float()


def update_dynamic_independent(num_depots, max_car_load, dynamic, distance, next_idx, current_idx):  # 加了参数：访问的前一个点。

    """
    这是用于更新当前地图的dynamic的函数。此函数用于【非共享仓库】。
    要用到distance是因为dynamic里面的load需要减去距离……
    current_idx：当前所在的点id（即离开的点）。如果为仓库，则需要更新该点demand=-1以说明已经为空。
    next_idx：访问的下一个节点。认为【已经去了】 。所以下一个节点的需求、飞机的load要根据节点类型而更新。
    """

    current_idx = current_idx.squeeze()
    # 根据【下一个节点】是访问仓库还是城市，以不同方式更新dynamic
    visit = next_idx.ge(num_depots)  # 下一个节点访问的是城市还是仓库
    depot = next_idx.lt(num_depots)

    # 克隆动态变量，以免破坏图
    all_loads = dynamic[:, 0].clone()
    all_demands = dynamic[:, 1].clone()
    load = torch.gather(all_loads, 1, next_idx.unsqueeze(1))  # 获得batch里每一个样本，下一个节点load【实际含义是当前carload】
    demand = torch.gather(all_demands, 1, next_idx.unsqueeze(1))  # 获得batch里每一个样本，下一个节点的demand
    n2n_distance = distance
    # 在batch中 - 如果我们下一个点选择访问一个城市：
    if visit.any():
        distance = n2n_distance[  #取对应batch的对应两点间的距离值（毕竟只要减去两个点之间的真实距离，无需考虑仓库距离。
            torch.arange(n2n_distance.size(0)), current_idx, next_idx.squeeze()].unsqueeze(1)
        # 上一次选择的节点与这次选择的节点的差值
        check_load = load - demand - distance
        if (check_load < 0).any():
            print("check_load",check_load)
            raise ValueError("Error: 存在负载为负数.")

        new_load = torch.clamp(check_load, min=0)  # 当前负载-下一个点需求-路程损耗

        check_demand =demand - load+distance # 下一个点需求减去车辆走路后的负载。
        if (check_demand>0).any():
            raise ValueError("Error:无法满足下一个城市的需求。") # 注意只有需求无法分割的时候，这个报错才是必要的
        new_demand = torch.clamp(check_demand, min=0)

        # 将载重量广播到所有节点，但单独更新需求量
        visit_idx = visit.nonzero().squeeze()

        all_loads[visit_idx] = new_load[visit_idx]
        all_demands[visit_idx, next_idx[visit_idx]] = new_demand[visit_idx].view(-1)

    #  dynamic会标记仓库是非空（-1空、0非空），所以update的时候，如果无人机离开仓库（当前点=仓库），就要更新该仓库demand为
    #  注意，认为飞机【已经去了】下一个节点 next_idx。所以下一个节点的需求、飞机的load也要根据节点类型而更新。
    # 使用布尔索引来找出当前节点的是仓库的样本
    depot_visited = current_idx.lt(num_depots).to(dynamic.device)

    # depot_visited_idx是【数组下标】不是城市序号！！
    depot_visited_idx = depot_visited.nonzero().squeeze()

    # 使用布尔索引操作来更新all_demands【把当前节点是depot的样本，的对应depot的demand更新为-1】
    # all_demands[depot_visited_idx.to('cuda'), last_visited.to('cuda')[depot_visited_idx]] = -1. #+ new_load[depot_visited_idx].reshape(-1,2)
    all_demands[depot_visited_idx.to(dynamic.device), current_idx.to(dynamic.device)[
        depot_visited_idx]] = -1.  # + new_load[depot_visited_idx].reshape(-1,2)

    # 在batch中 - 如果我们下一个选择访问一个仓库，则load回满，仓库的demand标记为0
    # if depot.any():
    #     all_loads[depot.nonzero().squeeze()] = 1.
    #     all_demands[depot.nonzero().squeeze(), 0] = 0.
    if depot.any():
        all_loads[depot.nonzero().squeeze()] = float(max_car_load)  # 恢复满格
        all_demands[depot.squeeze(), next_idx[depot.squeeze()]] = 0.

    # 把load和demand拼接会dynamic
    new_dynamic = torch.cat((all_loads.unsqueeze(1), all_demands.unsqueeze(1)), 1)
    return new_dynamic.clone().detach().to(device=dynamic.device)  # 避免额外的计算开销和不必要的内存使用


def reward(static, tour_indices):  # 这个是旧的reward，,depot_number参数没用
    """
    根据 tour_indices 给出的所有城市/节点之间的欧几里得距离

    参数:
    static: 包含所有城市/节点位置的静态信息张量。
    tour_indices: 由城市/节点索引构成的序列，表示访问的顺序。

    返回:
    旅行总长度: 计算得到的旅行的总欧几里得距离。
    """
    total_len = []
    for tour_indices_item in tour_indices:# tour_indices是长度为depot num的列表……
        # 将索引转换回旅行路线。idx：B*2*飞机路过的node数量。
        idx = tour_indices_item.unsqueeze(1).expand(-1, static.size(1), -1)
        tour = torch.gather(static.data, 2, idx).permute(0, 2, 1) # tour大小：B*飞机路过的node数量*2（XY）#

        # 确保总是返回到仓库 - 注意额外的 concat不会增加任何额外的损失，因为连续点之间的欧几里得距离是 0
        # tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
        tour_len = torch.sqrt(torch.sum(torch.pow(tour[:, :-1] - tour[:, 1:], 2), dim=2))
        total_len.append(tour_len.sum(1))
    # 返回旅行总长度
    # total_len.sum(1)
    # return total_len#tour_len.sum(1)

    # 将列表中的所有元素堆叠成一个新的张量。# total_len是大小为depot num的列表，元素为tensor(B,)
    total_len_tensor = torch.stack(total_len) # total_len_tensor大小depot num * B
    # 计算所有旅行的总长度之和
    #total_distance = total_len_tensor.sum(0) # 大小(B,),是把同一个Batch内部，所有飞机的总路程求和。
    return total_len_tensor #total_distance


def render(static, tour_indices, num_depots, save_path):
    """绘制找到的解决方案。"""
    # 关闭所有潜在的之前的绘图
    plt.close('all')

    # 确定绘制的子图数量，至少绘制一张图
    num_plots = 1  # 注意子图个数是num_plots*num_plots

    # 创建子图
    _, axes = plt.subplots(nrows=num_plots, ncols=num_plots,
                           sharex='col', sharey='row')
    # 如果只有一个子图，确保 axes 是二维的
    if num_plots == 1:
        axes = [[axes]]
    # 将 axes 从二维列表转换为一维列表
    axes = [a for ax in axes for a in ax]
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta', 'brown', 'gray']  # 创建颜色列表
    # 遍历每个子图，绘制路径
    for i, ax in enumerate(axes):
        for j in range(len(tour_indices)): # j是飞机维度
            # Convert the indices back into a tour
            idx = tour_indices[j][i] # 第二个维度是Batch size 的维度。
            if len(idx.size()) == 1:
                idx = idx.unsqueeze(0)
            idx = idx.expand(static.size(1), -1)
            data = torch.gather(static[i].data, 1, idx).cpu().numpy()

            x = np.hstack(data[0])
            y = np.hstack(data[1])
            # Assign each subtour a different colour & label in order traveled
            idx = np.hstack(tour_indices[j][i].cpu().numpy().flatten())
            where = np.where(idx < num_depots)[0]
            for k in range(len(where) - 1):
                low = where[k]
                high = where[k + 1]
                ax.plot(x[low: high + 1], y[low: high + 1], zorder=1, color=colors[j % 10], label=j)

            # 给每个点加上它的序号
            for point_idx, (px, py) in enumerate(zip(x, y)):
                ax.text(px, py, str(idx[point_idx]), fontsize=8, ha='right', va='bottom')

        ax.scatter(static[i, 0, num_depots:].cpu(), static[i, 1, num_depots:].cpu(), s=10, c='r', zorder=2)
        ax.scatter(static[i, 0, :num_depots].cpu(), static[i, 1, :num_depots].cpu(), s=50, c='k', marker='*', zorder=3)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    # plt.show()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    plt.close('all')

def render_dynamic(static, tour_indices, num_depots, save_path):
    """绘制找到的解决方案。"""
    # 关闭所有潜在的之前的绘图
    plt.close('all')
    import matplotlib.animation as animation
    # 确定绘制的子图数量，至少绘制一张图
    num_plots = 1  # 注意子图个数是num_plots*num_plots
    points_set=[]
    # 遍历每个子图，绘制路径
    for i in range(num_plots):
        for j in range(len(tour_indices)):  # j是飞机维度
            # Convert the indices back into a tour
            idx = tour_indices[j][i]  # 第二个维度是Batch size 的维度。
            if len(idx.size()) == 1:
                idx = idx.unsqueeze(0)
            idx = idx.expand(static.size(1), -1)
            data = torch.gather(static[i].data, 1, idx).cpu().numpy() #2 路程长度
            data = data.transpose(1,0)
            points_set.append(data)

    # # 创建子图

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    plt.tight_layout()

    # ----------------------
    x = [np.array([p[0] for p in track]) for track in points_set]
    y = [np.array([p[1] for p in track]) for track in points_set]

    tower_x = [x1[1:] for x1 in x]
    tower_y = [y1[1:] for y1 in y]
    for xt, yt in zip(tower_x, tower_y):
        plt.scatter(xt, yt, s=10)  # 画出城市的点 # s是大小
    # 用于画出仓库点。拼接在一起
    plt.scatter([x1[0] for x1 in x], [y1[0] for y1 in y], marker="*", s=90)  # 仓库

    lines = [ax.plot([], [], lw=1)[0] for _ in range(len(points_set))]
    plt.tight_layout()
    # ----------------------

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(N):
        for line, x1, y1 in zip(lines, x, y):
            line.set_data(x1[:N], y1[:N])
        return lines

    ani = animation.FuncAnimation(fig, animate, 50, init_func=init, interval=200)
    title = f"RL T{static.size(2)-num_depots} UAV{num_depots} share={str(share)}"
    plt.title(title)
    ani.save(os.path.join(save_path, f"{title}.gif"), writer='pillow',dpi=500)  # 保存
    # plt.show()
    plt.close('all')



"""
主要组件
* StateCritic类：估计给定问题的复杂性。它接受静态和动态输入，并通过一系列的全连接层（Conv1d）来估计问题的复杂性或成本。
* Critic类：一个简化版的StateCritic，用于估计问题的复杂性。它通过全连接层处理输入，输出问题复杂性的估计值。
"""

class StateCritic(nn.Module):
    # 简单的使用model里的Encoder方法以及三次一维卷积得到Critic的结果
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    # todo 如何修改Critic：我希望它变成预测一个无人机的路程长度。
    #   actor是不是也要同步修改输入？
    #   critic的输入：
    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1) # 卷积核大小kernel_size*in_channel，个数是out_channels=输出第二维
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:  # 张量的行数应该大于1，从而满足xavier的使用条件
                nn.init.xavier_uniform_(p)  # xavier预防一些参数过大或过小的情况，再保证方差一样的情况下进行缩放，便于计算

    def forward(self, static, dynamic):
        '''
        static:B,2,num_node
        dynamic:B,2,num_node
        '''
        # Use the probabilities of visiting each
        # 静态和动态数据编码
        static_hidden = self.static_encoder(static)# B 128,num_node (?num_noode维度没变是因为什么
        dynamic_hidden = self.dynamic_encoder(dynamic) # B 128,num_node
        # 讲两种编码后第二个维度上拼接，就是按照Conv1d的输出结果个数(hidden_size)进行拼接，得到2*hidden_size
        hidden = torch.cat((static_hidden, dynamic_hidden), 1)# B 256,num_node (256是两个128拼在一起)

        output = F.relu(self.fc1(hidden)) # B，20，num_node (num_node没变是因为kernel size=1，
        output = F.relu(self.fc2(output)) # B，20，num_node
        # 提取特征后，第二维求和，得到结果
        output = self.fc3(output).sum(dim=2) # B,1
        return output


class StateCritic4UAV(nn.Module):
    #todo【试图修改critic，使之预测的是一条无人机的轨迹。】
    # 输入：原始的static信息、dynamic，新增：无人机的tour idx信息（？在这里面取出无人机的路径坐标）（在外面也行）
    #

    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    # todo 如何修改Critic：我希望它变成预测一个无人机的路程长度。
    #   actor是不是也要同步修改输入？
    #   critic的输入：
    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic4UAV, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.tour_encoder = Encoder(static_size, hidden_size) # todo test

        # Define the encoder & decoder models
        self.fc1 = nn.Conv1d(hidden_size * 2, 20, kernel_size=1) # 卷积核大小kernel_size*in_channel，个数是out_channels=输出第二维
        self.fc2 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc3 = nn.Conv1d(20, 1, kernel_size=1)
        # todo 新增.这是给tour hidden用的。
        self.fc4 = nn.Conv1d(hidden_size, 20, kernel_size=1) # 卷积核大小kernel_size*in_channel，个数是out_channels=输出第二维
        self.fc5 = nn.Conv1d(20, 20, kernel_size=1)
        self.fc6 = nn.Conv1d(20, 1, kernel_size=1)

        for p in self.parameters():
            if len(p.shape) > 1:  # 张量的行数应该大于1，从而满足xavier的使用条件
                nn.init.xavier_uniform_(p)  # xavier预防一些参数过大或过小的情况，再保证方差一样的情况下进行缩放，便于计算

    def forward(self, static, dynamic,tour_indices):
        '''
        static:B,2,num_node
        dynamic:B,2,num_node
        返回值：对每一个无人机线路的预估长度？(depot num, B)
        '''

        # todo 新增####################
        # total_len = []
        # for tour_indices_item in tour_indices:  # tour_indices是长度为depot num的列表……
        #     # 将索引转换回旅行路线。idx：B*2*飞机路过的node数量。
        #     idx = tour_indices_item.unsqueeze(1).expand(-1, static.size(1), -1)
        #     tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)  # tour大小：B*飞机路过的node数量*2（XY）#
        #     tour_hidden = self.tour_encoder(tour)# todo ????????为什么是所有无人机的路程啊！！不应该每个actor judge 一次吗。

        # 将索引转换回旅行路线。idx：B*2*飞机路过的node数量。
        idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
        tour = torch.gather(static.data, 2, idx)# 好像无需转置。.permute(0, 2, 1)  # tour大小：B*飞机路过的node数量*2（XY）#
        tour_hidden = self.tour_encoder(tour)
        output_2 = F.relu(self.fc4(tour_hidden))
        output_2 = F.relu(self.fc5(output_2))
        output_2 = self.fc6(output_2).sum(dim=2)
        ###################
        # todo 然后把本critic的loss计算为它对每一个uav的estimate的差的均方误差“？？？/

        # # Use the probabilities of visiting each
        # # 静态和动态数据编码
        # static_hidden = self.static_encoder(static)# B 128,num_node (num_noode维度没变是因为是一维卷积。
        # dynamic_hidden = self.dynamic_encoder(dynamic) # B 128,num_node
        # # 讲两种编码后第二个维度上拼接，就是按照Conv1d的输出结果个数(hidden_size)进行拼接，得到2*hidden_size
        # hidden = torch.cat((static_hidden, dynamic_hidden), 1)# B 256,num_node (256是两个128拼在一起)
        #
        # output = F.relu(self.fc1(hidden)) # B，20，num_node (num_node没变是因为kernel size=1，
        # output = F.relu(self.fc2(output)) # B，20，num_node
        # # 提取特征后，第二维求和，得到结果
        # output = self.fc3(output).sum(dim=2) # B,1

        return output_2 #output

def validate(data_loader, actor, reward_fn, render_fn=None, save_dir='.',
             num_plot=1, depot_number=-1):
    """Used to monitor progress on a validation set & optionally plot solution.
    在训练阶段，每个epoch结束的时候会使用validate set来调用此函数进行评估。
    在所有训练完成，会使用test set来调用此函数进行最终的模型效果计算。
    返回：reward平均值，reward集合。
    """
    # 将actor设置为评估模式，确保不会应用随机性或梯度计算
    actor.eval()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    rewards = []
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic, x0 = batch

        static = static.to(device)  # 复制变量到GPU上
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():  # 把requires_grad设置为False,避免反向传播时自动求导，节约了显存

            # 关键步骤！！！，把静态数据和动态数据进行actor操作，获得访问节点的索引。
            tour_indices, _ = actor.forward(static, dynamic, x0)

        # 使用vrp奖励函数 reward_fn 计算预测的旅游索引的奖励。取奖励的均值，并使用 item() 提取标量值，添加到rewards列表中
        # reward = reward_fn(static, tour_indices).mean().item()
        reward = reward_fn(static, tour_indices) # 已修改reward fn，现在大小为tensor(num_depot,B)
        reward = reward.sum(0)
        batch_reward_mean = reward.mean().item()  # 本batch的均值。
        # rewards.append(batch_reward_mean)
        rewards.extend(reward.tolist())
        # 控制vrp的渲染函数，主要是于作图有关
        if render_fn is not None and batch_idx < num_plot:
            name = 'batch%d_reward%2.4f.png' % (batch_idx, batch_reward_mean)
            path = os.path.join(save_dir, name)
            render_fn(static, tour_indices,depot_number, path)
            # render_dynamic(static, tour_indices,depot_number, "GIF")
    # 将模型 actor 设置回训练模式
    actor.train()
    # 返回平均奖励
    return np.mean(rewards), rewards


def train(actor, critic4uav, task, num_city, train_data, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,
          depot_num, **kwargs):
    # 搭建主要的AC网络，进行全部的训练
    """Constructs the main actor & critic networks, and performs all training."""
    # 时间，保存路径获取
    #google_drive_path = '/content/drive/MyDrive/' # 这都什么啊居然是绝对路径。
    current_dir = os.path.dirname(os.path.abspath(__file__))
    now = datetime.datetime.now()
    format_now = '%s' % now.month + "_" + '%s' % now.day + "_" + '%s' % now.hour + "_" + '%s' % now.minute + "_" + '%s' % now.second

    save_dir = os.path.join(current_dir, task + "parallel_train_log_uav_goal_times C2", '%d' % num_city, format_now)  # ./vrp/numnode/time
    # 创建能够保存训练中checkpoint的文件夹
    checkpoint_dir = os.path.join(save_dir, 'train_checkpoints')  # /vrp/numnode/time/checkpoints
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # 定义AC优化器
    actor_optim = optim.Adam(actor.parameters(), lr=actor_lr)
    # critic_optim = optim.Adam(critic.parameters(), lr=critic_lr)

    # todo 参数新增了critic4uav.在此创建优化器
    critic4uav_optim = optim.Adam(critic4uav.parameters(), lr=critic_lr)

    # DataLoader理解：只是一个容器类，方便训练数据实现各种分batch、打散、多线程的功能。实际上的数据内容是自己重写里面的getitem函数。
    train_loader = DataLoader(train_data, batch_size, True, num_workers=0)  # 读取训练数据，一次读取batch_size个，无序
    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)  # 读取测试数据，一次读取batch_size个，有序

    best_reward = np.inf  # 正无穷大
    all_epoch_loss, all_epoch_reward = [], []
    for epoch in range(20):  # 执行20轮训练 fixme!!!!!!!!能不能把这个写到外面。

        actor.train()
        # critic.train()
        critic4uav.train()

        # loss放每个batch的损失
        times, losses, rewards, critic_rewards = [], [], [], []

        epoch_start = time.time()
        start = epoch_start

        for batch_idx, batch in enumerate(train_loader):

            static, dynamic, x0 = batch

            static = static.to(device)
            dynamic = dynamic.to(device)
            x0 = x0.to(device) if len(x0) > 0 else None

            # 用每条无人机路径来训练同一个网络。
            # Full forward pass through the dataset(使用actor的前向传播)
            tour_indices, tour_logp = actor(static, dynamic, x0)  # 调用的是forward函数。

            reward = reward_fn(static, tour_indices) # (depot num, B)

            # 新增：为了把优化目标改为独立的uav的路径长度，tour_logp：list[depot_num:tensor(T,B,)]
            tour_logp_per_uav=[tour_logp[i].sum(dim=1) for i in range (depot_num)]

            # Query the critic for an estimate of the reward(向评论家询问奖励的估计值)
            # critic_est = critic(static, dynamic).view(-1)  # (B,) #
            # critic_est_per_uav=critic_est / depot_num  # (B,) # todo test 暂时使用critic/depot num进行测试。
            # advantage_per_uav=(reward-critic_est_per_uav).detach() # (depot num, B)
            critic_est=[critic4uav(static, dynamic, tour_indices[i]) for i in range(depot_num)]
            actor_optim.zero_grad()
            critic4uav_optim.zero_grad()
            actor_losses=[]
            # 新增：对每辆无人机的路径进行训练。
            for uav_id in range(depot_num):

                # 最原始的uav goal，只是把critic预测的东西改了。
                # advantage = reward[uav_id] - critic_est[uav_id].detach()

                # 尝试一下adv2？对半开。
                advantage = 0.5*(reward[uav_id] - critic_est[uav_id].detach()) + 0.5*(reward.sum(0)- sum(critic_est))

                actor_loss = torch.mean(advantage.detach() *  tour_logp_per_uav[uav_id])  # tour_logp:(B，路程长度）.之所以要sum是为了求路径出现概率
                actor_loss.backward(retain_graph=True)
                actor_losses.append(actor_loss.detach().item()) # 保存一下,用于绘图

                critic_loss = torch.mean((reward[uav_id] - critic_est[uav_id]) ** 2)
                critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            actor_optim.step()

            torch.nn.utils.clip_grad_norm_(critic4uav.parameters(), max_grad_norm)
            critic4uav_optim.step()

            # 将奖励估计值，真实奖励值，actor损失平均求和后写入空列表中
            # critic_rewards.append(torch.mean(critic_est.detach()).item())

            # rewards.append(torch.mean(reward.detach()).item())# todo 已修改。
            rewards.append(torch.mean(reward.sum(0).detach()).item())
            actor_losses=sum(actor_losses)/len(actor_losses)
            # losses.append(torch.mean(actor_losses.detach()).item())# todo 已修改。
            losses.append(actor_losses)

            # 每100次输出
            if (batch_idx + 1) % 100 == 0:
                end = time.time()
                times.append(end - start)
                start = end

                mean_loss = np.mean(losses[-100:])
                mean_reward = np.mean(rewards[-100:])

                print('  Batch %d/%d, reward: %2.3f, loss: %2.4f, took: %2.4fs' %
                      (batch_idx, len(train_loader), mean_reward, mean_loss,
                       times[-1]))

        ###【画出当前epoch内的图像：每100个batch计算loss和reward，】
        averages_loss = [np.mean(np.array(losses)[i:i + 100]) for i in
                         range(0, len(losses), 100)]  # # loss是每个batch的损失集合
        averages_reward = [np.mean(np.array(rewards)[i:i + 100]) for i in range(0, len(rewards), 100)]
        # 如果样本很小的话画不出averages_loss和averages_reward

        # x = np.arange(len(averages_loss))
        # plt.figure(1)
        # plt.plot(x, averages_loss)
        # plt.title(f'Epoch {epoch} Averages_loss')
        # plt.grid(True)
        # plt.savefig(os.path.join(save_dir,"Averages_loss"))
        # # plt.show() # debug 模式似乎不能show
        # plt.figure(2)
        # plt.plot(x, averages_reward)
        # plt.title(f'Epoch {epoch} Averages_reward')
        # plt.grid(True)
        # plt.savefig(os.path.join(save_dir, "Averages_reward"))

        # 保存当前epoch的每100个bach的平均loss和reward （若一个epoch里面有100个batch，则每10个之间计算一次mean，有10个值）
        all_epoch_loss.extend(averages_loss)
        all_epoch_reward.extend(averages_reward)

        # 保存目前为止的loss曲线图。
        plt.figure(3)
        plt.plot(np.arange(len(all_epoch_loss)), all_epoch_loss)
        plt.title('All_epoch_loss')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "All_epoch_loss.jpg"))

        plt.figure(4)
        plt.plot(np.arange(len(all_epoch_reward)), all_epoch_reward)
        plt.title('All_epoch_reward')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "All_epoch_reward.jpg"))
        # plt.show()
        #############

        mean_loss = np.mean(losses)
        mean_reward = np.mean(rewards)

        # Save the weights
        epoch_dir = os.path.join(checkpoint_dir, '%s' % epoch)  #./vrp_numnode_time//checkpoints/0
        if not os.path.exists(epoch_dir):
            os.makedirs(epoch_dir)

        actor_save_path = os.path.join(epoch_dir, 'actor.pt')  #./vrp_numnode_time//checkpoints/0/actor.pt
        torch.save(actor.state_dict(), actor_save_path)
        critic_save_path = os.path.join(epoch_dir, 'critic.pt')  #./ vrp_numnode_time//checkpoints/0/critic.pt
        torch.save(critic4uav.state_dict(), critic_save_path)

        # -------------------Valid------------------
        print(f"Train {epoch} ends. Start valid {epoch}.")
        valid_dir = os.path.join(save_dir, "valid_picture")  # test去掉后面的epoch

        # 每一个epoch validate一次。
        mean_valid, _ = validate(valid_loader, actor, reward_fn, render_fn,
                                 valid_dir, num_plot=5, depot_number=depot_num)

        # Save best model parameters(保存最佳奖励) 是每个epoch检查!!!使用valid set的reward来选！！！
        if mean_valid < best_reward:  # reward 就是路程总长度。暂定无惩罚项(好像确实不需要额外的乘法）
            best_reward = mean_valid

            actor_save_path = os.path.join(save_dir, 'actor.pt')
            torch.save(actor.state_dict(), actor_save_path)

            critic_save_path = os.path.join(save_dir, 'critic.pt')
            torch.save(critic4uav.state_dict(), critic_save_path)
        # 输出平均(actor)损失，平均奖励，平均验证奖励，运行时间
        print(f"Epoch {epoch} ", end="")
        print('Mean: Loss %2.4f, Reward %2.4f, Valid reward %2.4f, took: %2.4fs ' \
              '(%2.4fs / 100 batches)\n' % \
              (mean_loss, mean_reward, mean_valid, time.time() - epoch_start,
               np.mean(times)))  # 这里 np f，会为空，显示nan但是不影响运行。

    #
    plt.figure(3)
    plt.plot(np.arange(len(all_epoch_loss)), all_epoch_loss)
    # print(f"len(all_epoch_loss)={len(all_epoch_loss)}")
    plt.title('All_epoch_loss')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "All_epoch_loss.jpg"))
    # plt.show()

    plt.figure(4)
    plt.plot(np.arange(len(all_epoch_reward)), all_epoch_reward)
    # print(f"len(all_epoch_reward)={len(all_epoch_reward)}")
    plt.title('All_epoch_reward')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "All_epoch_reward.jpg"))
    # plt.show()


def run_RL_exp(share_depot, args):
    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 30: 35, 50: 40, 100: 50, 200: 80}  # todo 已经废弃
    # MAX_DEMAND = 1
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 2  # (load, demand)
    max_load = -1  #LOAD_DICT[args.num_city] # todo没事现在maxload已经废弃了
    # car_load = 30.  #原版、

    map_size = 1  # fixme 原代码地图大小默认1，不需要指定。而且本来产生坐标的时候就是0-1范围的浮点。
    car_load = 2 * map_size * 1.4  # 测试
    MAX_DEMAND = 0.1  # 测试 # todo 目前是固定值。

    if share_depot:
        print("Shared depot.")
        actor = DRL4TSP(STATIC_SIZE,
                        DYNAMIC_SIZE,
                        args.hidden_size,
                        car_load,
                        args.depot_num,
                        update_dynamic_shared,
                        update_mask_shared,
                        node_distance_shared,
                        args.num_layers,
                        args.dropout).to(device)
    else:
        print("Not Shared depot.")
        actor = DRL4TSP(STATIC_SIZE,
                        DYNAMIC_SIZE,
                        args.hidden_size,
                        car_load,
                        args.depot_num,
                        update_dynamic_independent,
                        update_mask_independent,
                        node_distance_independent,
                        args.num_layers,
                        args.dropout).to(device)

    # 实例化critic
    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)
    critic4uav=StateCritic4UAV(STATIC_SIZE, DYNAMIC_SIZE, args.hidden_size).to(device)

    if args.checkpoint:  # 读取之前保存的模型。
        print(f"args.checkpoint:已经有ckpt,读取:{args.checkpoint}")
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))  # load_state_dict：加载模型参数
        path = os.path.join(args.checkpoint, 'critic.pt')
        critic.load_state_dict(torch.load(path, device))  # 加载模型参数
    else:
        print("No args.checkpoint：模型从0初始化。")

    print(f"args.num_city={args.num_city}")
    print(f"args.depot_num={args.depot_num}")

    if not args.test:
        # 生成随机训练数据集(1000000)，验证数据集(1000)
        train_data = VehicleRoutingDataset(args.train_size,
                                           args.num_city,
                                           max_load,
                                           car_load,
                                           MAX_DEMAND,
                                           args.seed,
                                           args.depot_num)

        valid_data = VehicleRoutingDataset(args.valid_size,
                                           args.num_city,
                                           max_load,
                                           car_load,
                                           MAX_DEMAND,
                                           args.seed + 1,
                                           args.depot_num)

        # 转换为字典，并且添加训练数据，验证数据，reward奖励函数，render渲染函数
        kwargs = vars(args)
        kwargs['train_data'] = train_data
        kwargs['valid_data'] = valid_data
        kwargs['reward_fn'] = reward
        kwargs['render_fn'] = render

        print("No args.test：模型开始训练")
        print(f"args.train_size={args.train_size}")
        print(f"args.batch_size={args.batch_size}")

        train(actor,critic4uav, **kwargs)  # 训练！!!!!!!!!!!!!!!!!!!!!
        print("训练结束。")

    print(f"开始测试：args.valid_size={args.valid_size}")
    # 生成测试数据，大小于验证数据一致(1000)
    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_city,
                                      max_load,
                                      car_load,
                                      MAX_DEMAND,
                                      args.seed + 2,
                                      args.depot_num)

    if share_depot:
        test_dir = 'test_picture_shared_depot'
    else:
        test_dir = 'test_picture'

    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)

    test_out, test_reward = validate(test_loader, actor, reward, render, test_dir, num_plot=1,
                                     depot_number=args.depot_num)
    # print("DRL:Average tour length in test set: ", test_out)
    # reward_greedy = run_greedy_VRP(test_data.static, args.num_city, args.depot_num,share=share_depot)

    # print("\nRun data analysis:")
    # analysis = data_analysis.Reward_Collect()
    # analysis.reward_RL=test_reward
    # analysis.reward_greedy = reward_greedy
    # analysis.run_analysis()
    return test_reward #, reward_greedy

def run_multi_alg_test(share_depot, args, algorithm):
    '''
    本函数内部生成测试数据集。
    返回的是{算法：测试结果} 字典
    '''
    if share_depot:
        print("Shared depot.")
    else:
        print("Independent depot.")
    print(f"args.num_city={args.num_city}")
    print(f"args.depot_num={args.depot_num}")
    print(f"开始测试：args.valid_size={args.valid_size}")

    max_load = -1  #LOAD_DICT[args.num_city] # todo没事现在maxload已经废弃了
    map_size = 1  # fixme 原代码地图大小默认1，不需要指定。而且本来产生坐标的时候就是0-1范围的浮点。
    car_load = 2 * map_size * 1.4  # 测试
    MAX_DEMAND = 0.1  # 测试 # todo 目前是固定值。

    # 生成测试数据，大小于验证数据一致(1000)
    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_city,
                                      max_load,
                                      car_load,
                                      MAX_DEMAND,
                                      args.seed + 2,
                                      args.depot_num)

    # print('DRL:Average tour length in test set: ', test_out)

    algorithm_result={}
    from Greedy_VRP_share import run_greedy_VRP
    # -----------------------------------------------------------------Greedy
    if "greedy" in algorithm or "Greedy" in algorithm:
        reward_greedy = run_greedy_VRP(test_data.static, args.num_city, args.depot_num,share=share_depot)
        algorithm_result["Greedy"]=reward_greedy

    #------------------------------------------------------------------RL
    if "RL" in algorithm:
        STATIC_SIZE = 2  # (x, y)
        DYNAMIC_SIZE = 2  # (load, demand)
        if share_depot:
            actor = DRL4TSP(STATIC_SIZE,
                            DYNAMIC_SIZE,
                            args.hidden_size,
                            car_load,
                            args.depot_num,
                            update_dynamic_shared,
                            update_mask_shared,
                            node_distance_shared,
                            args.num_layers,
                            args.dropout).to(device)
        else:
            actor = DRL4TSP(STATIC_SIZE,
                            DYNAMIC_SIZE,
                            args.hidden_size,
                            car_load,
                            args.depot_num,
                            update_dynamic_independent,
                            update_mask_independent,
                            node_distance_independent,
                            args.num_layers,
                            args.dropout).to(device)

        print(f"RL测试模式：读取ckpt:{args.checkpoint}")
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))  # load_state_dict：加载模型参数

        if share_depot:
            test_dir = 'test_picture_shared_depot'
        else:
            test_dir = 'test_picture'

        test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)

        RL_test_out, RL_test_reward = validate(test_loader, actor, reward, render, test_dir, num_plot=5,
                                         depot_number=args.depot_num)
        algorithm_result["RL"]= RL_test_reward

    return  algorithm_result


def test_generalization_uav_change(shared, run_alg_name):
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_city', default=200, type=int)
    # parser.add_argument('--actor_lr', default=5e-4, type=float)
    # parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--actor_lr', default=1e-4, type=float)  # 学习率，现在在训练第4epoch，我手动改了一下
    parser.add_argument('--critic_lr', default=1e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size', default=-1, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)
    parser.add_argument('--depot_num', default=-1, type=int)
    # 解析为args
    args = parser.parse_known_args()[0]  # colab环境跑使用

    if shared:
        args.checkpoint = os.path.join("trained_model", "total_shared_w200")
        # args.checkpoint = os.path.join("trained_model", "total_shared_w200_w250")
    else:  # not share
        args.checkpoint = os.path.join("trained_model", "trained_w200")
    print("比较算法：",run_alg_name)
    reward_list_dict={}

    uav_list=list(range(2, 4))
    for uav_n in uav_list:
        args.depot_num=uav_n
        reward_dict = run_multi_alg_test(shared, args, algorithm=run_alg_name)
        for key,val in reward_dict.items():
            reward_list_dict.setdefault(key,[])
            reward_list_dict[key].append(np.mean(val))# 对每一个算法的reward集合计算平均值。

    plt.close('all')
    for alg in reward_list_dict.keys():
        plt.plot(uav_list, reward_list_dict[alg], label=f"{alg} average path")
    # plt.plot(uav_list, avg_R_Greedy, label="Greedy average path")
    plt.legend()
    dir = os.path.join("generalization_test_picture")
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, f"Greedy_VS_RL on {args.num_city} tower share={shared}.png"))
    # plt.show()

    # 储存csv
    reward_filename = f"Generalization_compare on {args.num_city} T share={str(shared)}.csv"
    txt = ",".join(reward_list_dict.keys())+ "\n"
    # for greedy, RL in zip(avg_R_Greedy, avg_R_RL):
    for rs in zip(*reward_list_dict.values()):
        rs = list(map(str, list(rs)))
        txt += ",".join(rs)
        txt+="\n"
    with open(os.path.join(dir,reward_filename), "w") as f2:
        f2.write(txt)
    print(txt)

def test_generalization_tower_change(share,run_alg_name):
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_city', default=-1, type=int)
    # parser.add_argument('--actor_lr', default=5e-4, type=float)
    # parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--actor_lr', default=1e-4, type=float)  # 学习率，现在在训练第4epoch，我手动改了一下
    parser.add_argument('--critic_lr', default=1e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size', default=-1, type=int)
    parser.add_argument('--valid-size', default=1000, type=int)
    parser.add_argument('--depot_num', default=20, type=int)
    # 解析为args
    args = parser.parse_known_args()[0]  # colab环境跑使用

    args.test = True
    if share:
        args.checkpoint = os.path.join("trained_model", "total_shared_w200")
    else:  # not share
        args.checkpoint = os.path.join("trained_model", "trained_w200")

    # run_alg_name = ["Greedy", "RL"]
    print("比较算法：", run_alg_name)
    reward_list_dict = {}

    tower_list = list(range(200, 301, 50))
    for tower_n in tower_list:
        args.num_city = tower_n
        # reward_rl, reward_greedy = run_exp(share, args)
        reward_dict= run_multi_alg_test(share, args, algorithm=run_alg_name)
        for key, val in reward_dict.items():
            reward_list_dict.setdefault(key, [])
            reward_list_dict[key].append(np.mean(val))  # 对每一个算法的reward集合计算平均值。
    for alg in reward_list_dict.keys():
        plt.plot(tower_list, reward_list_dict[alg], label=f"{alg} average path")
    plt.legend()
    dir = os.path.join("generalization_test_picture")
    if not os.path.exists(dir):
        os.makedirs(dir)
    share=str(share)
    plt.savefig(os.path.join(dir, f"Greedy_VS_RL on {args.depot_num} UAV share={share}.png"))
    # plt.show()

    # 储存csv
    reward_filename = f"Generalization_on {args.depot_num} UAV share={str(share)}.csv"
    txt = ",".join(reward_list_dict.keys()) + "\n"
    for rs in zip(*reward_list_dict.values()):
        rs = list(map(str, list(rs)))
        txt += ",".join(rs)
        txt+="\n"
    with open(os.path.join(dir,reward_filename), "w") as f2:
        f2.write(txt)
    print(txt)


if __name__ == '__main__':
    # 命令行参数解析器对象 parser.参数是按顺序的。。。。
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_city', default=50, type=int)  #todo 对齐#########
    # parser.add_argument('--actor_lr', default=5e-4, type=float)
    # parser.add_argument('--critic_lr', default=5e-4, type=float)
    # parser.add_argument('--actor_lr', default=1e-4, type=float)  # 学习率，现在在训练第4epoch，我手动改了一下
    # parser.add_argument('--critic_lr', default=1e-4, type=float)
    parser.add_argument('--actor_lr', default=1e-3, type=float)  # 学习率，现在在训练第4epoch，我手动改了一下
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=64, type=int)  #fixme#########################
    # parser.add_argument('--batch_size', default=8, type=int) ##########
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size', default=100000, type=int)  #fixme!!!!!!!!!!!!
    parser.add_argument('--valid-size', default=1000, type=int)
    parser.add_argument('--depot_num', default=5, type=int)  # todo ###############

    # 解析为args
    args = parser.parse_known_args()[0]  # colab环境跑使用
    # --------------------------------------------------------------------
    args.test = False
    # --------------------------------------------------------------------
    # 设置checkpoint路径
    share = False           # todo 检查#############
    # if share:
    #     args.checkpoint = os.path.join("trained_model", "total_shared_w200")
    # else:
    #     # args.checkpoint = os.path.join("trained_model", "trained_w200")
    #     args.checkpoint = os.path.join("vrpparallel_train_log", "50","11_22_11_17_8")# 服务器训练用。

    print("---------这是同时决策版本 py---------")
    #  【是不是应该把没量无人机的last hh存起来？但是不解决冲突版本也没有这样】
    # todo ：其他改进方案：输入编码器的静态信息只能是本无人机的xy吗？能不能加入其他无人机的xy？【太复杂的话是不是还要把网络加深】

    import time
    # 开始计时
    start_time = time.time()
    reward_rl=run_RL_exp(share, args)
    mean_rl = np.mean(reward_rl)
    print('DRL:Average tour length in test set: ', mean_rl)
    #--------------------------------------------------------------
    # test_generalization_uav_change(shared=share, run_alg_name=["RL","Greedy"])
    # test_generalization_tower_change(share=share,run_alg_name=["Greedy","RL"])
    # 结束计时
    end_time = time.time()
    # 计算并打印执行时间
    print(f"执行时间：{end_time - start_time}秒")
    print("Running ends.")

