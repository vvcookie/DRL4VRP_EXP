# todo 统计一下share 和 independent的不同情况下的电池的利用率：
# 去程和返程：当前点或者下一个点是电塔。
# 巡检路程：当前点和下一个点都是Tower
# 充电次数。
import numpy as np
import torch
from google.protobuf.internal.encoder import FloatSizer

from DRL_TW_random import *
print("记得检查import 的文件\n"*3)

def get_tower_path(tour_indices, depot_num):
    filtered_tours = []

    for tour in tour_indices:
        # tour is a tensor of shape (B, tour_len)
        B, tour_len = tour.shape

        # List to store filtered tours for this depot
        depot_filtered_tours = []

        # Iterate over each batch
        for b in range(B):
            # Mask for tower nodes (id >= depot_num)
            mask = tour[b] >= depot_num
            # Extract the tower nodes
            tower_nodes = tour[b][mask]
            depot_filtered_tours.append(tower_nodes)

        # Stack the filtered tours into a tensor of shape (B, filtered_tour_len)
        max_len = max(len(t) for t in depot_filtered_tours)
        padded_tours = [torch.cat([t, torch.full((max_len - len(t),), t[-1])]) for t in depot_filtered_tours]
        filtered_tours.append(torch.stack(padded_tours))

    return filtered_tours


def count_charging_visits(tour_indices, depot_num):
    charging_counts = []
    for tour in tour_indices:
        # tour is a tensor of shape (B, tour_len)
        B, tour_len = tour.shape

        # List to store charging counts for each batch
        batch_counts = []

        # todo 写的怎么是错的啊啊啊去死。。。哪有你这么实现的。
        # Iterate over each batch
        for b in range(B):
            end_depot=tour[b][-1]
            p=tour_len-1
            while tour[b][p-1]==end_depot:
                p-=1
            mask = tour[b][1:p] < depot_num # todo 注意掐头去尾了。开头和结尾的一次都不算
            count=sum(mask).item()
            batch_counts.append(count)

        # Convert to a tensor
        charging_counts.append(torch.tensor(batch_counts))
    charging_counts2=torch.sum(torch.stack(charging_counts,dim=1),1)

    return charging_counts2

def batch_path_analysis(test_loader, actor, reward_fn, depot_number):
    '''
    todo 在函数内部进行batch的forward。所以在这个函数里面完成统计。
    '''
    rewards = []
    tower_dis=[] # 收集每个样本的tower dis最后取平均（不能按照batch 平均）
    charge_times=[]# 收集每个样本的, 最后取平均
    for batch_idx, batch in enumerate(test_loader):

        static, dynamic, x0 = batch

        static = static.to(device)  # 复制变量到GPU上
        dynamic = dynamic.to(device)
        x0 = x0.to(device) if len(x0) > 0 else None

        with torch.no_grad():
            tour_indices, _ = actor.forward(static, dynamic, x0)
        reward = reward_fn(static, tour_indices)  # 本batch的所有reward 列表
        rewards.extend(reward.tolist())

        # tour_indices :长度为depot num list，内部大小为B，tour_len的tensor
        # 我要统计的是所有样本里面 1“去程返程reward”   2“巡检reward”   3“充电次数

        # 电塔之间巡检的路程的统计
        filtered_tours = get_tower_path(tour_indices, depot_number)
        tower_check_dis=reward_fn(static,filtered_tours)
        tower_dis.extend(tower_check_dis.tolist())

        # 充电次数（不计算开头结尾）的统计
        charge_time=count_charging_visits(tour_indices, depot_number)
        charge_times.extend(charge_time.tolist())

    # 电塔之间巡检的路程的统计
    print(f"mean of tower distance is {np.mean(tower_dis)}")
    print(f"mean of total distance is {np.mean(rewards)}")
    print(f"Percentage of tower distance ↑ = {round(np.mean(tower_dis)/np.mean(rewards)*100,2)}%")

    # 充电次数（不计算开头结尾）的统计
    print(f"mean of charge_time is {np.mean(charge_times)}")

    # 返回平均奖励
    return  rewards, tower_dis, charge_times

def path_analysis(upper_, lower_, share_depot, args):
    '''
    todo 需要在这里面完成数据的本地保存（？如果需要多组实验保存的话，也要在这里写）
    '''
    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 30: 35, 50: 40, 100: 50, 200: 80}  # todo 已经废弃
    # MAX_DEMAND = 1
    STATIC_SIZE = 2  # (x, y)
    DYNAMIC_SIZE = 2  # (load, demand)
    max_load = -1  #LOAD_DICT[args.num_city] # todo没事现在maxload已经废弃了

    map_size = 1  # fixme 原代码地图大小默认1，不需要指定。而且本来产生坐标的时候就是0-1范围的浮点。
    car_load = 2 * map_size * 1.4  # 测试
    MAX_DEMAND = 0.1  # 测试 # todo 目前是固定值。????????????????????
    print(f"目前是固定值。如果想要非固定值请import\n"*3)

    # from DRL_TW_random_unfix_demand import VehicleRoutingDataset_unfix_demand

    if share_depot:
        print("Shared depot.")
        actor = DRL4TSP(STATIC_SIZE,
                        DYNAMIC_SIZE,
                        args.hidden_size,
                        car_load,
                        args.depot_num,
                        update_dynamic_shared,
                        update_mask_shared_TW,
                        node_distance_shared,
                        args.num_layers,
                        args.dropout,
                        upper_, lower_).to(device)
    else:
        print("Not Shared depot.")
        actor = DRL4TSP(STATIC_SIZE,
                        DYNAMIC_SIZE,
                        args.hidden_size,
                        car_load,
                        args.depot_num,
                        update_dynamic_independent,
                        update_mask_independent_TW,
                        node_distance_independent,
                        args.num_layers,
                        args.dropout,
                        upper_, lower_).to(device)

    if args.checkpoint:  # 读取之前保存的模型。
        print(f"args.checkpoint:已经有ckpt,读取:{args.checkpoint}")
        path = os.path.join(args.checkpoint, 'actor.pt')
        actor.load_state_dict(torch.load(path, device))  # load_state_dict：加载模型参数
    else:
        print("No args.checkpoint：模型从0初始化。")

    print(f"args.num_city={args.num_city}")
    print(f"args.depot_num={args.depot_num}")

    print(f"开始测试：args.valid_size={args.valid_size}")
    # 生成测试数据，大小于验证数据一致(1000)
    test_data = VehicleRoutingDataset(args.valid_size,
                                      args.num_city,
                                      max_load,
                                      car_load,
                                      MAX_DEMAND,
                                      args.seed + 2,
                                      args.depot_num)


    test_loader = DataLoader(test_data, args.batch_size, False, num_workers=0)

    # 得到test set中每一个样本的统计指标的list（长度=test set）。todo 后续可以进行不同参数的指标的比较。。不过感觉很不好
    rewards, tower_dis, charge_times = batch_path_analysis(test_loader, actor, reward, depot_number=args.depot_num)
    # ----------------------

    return  rewards, tower_dis, charge_times

def batter_efficiency():
    pass

if __name__ == '__main__':
    # 命令行参数解析器对象 parser.参数是按顺序的。。。。
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_city', default=200, type=int)  # todo 对齐#########
    # parser.add_argument('--actor_lr', default=5e-4, type=float)
    # parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--actor_lr', default=1e-4, type=float)  # 学习率，现在在训练第4epoch，我手动改了一下
    parser.add_argument('--critic_lr', default=1e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=64, type=int)  # fixme#########################
    # parser.add_argument('--batch_size', default=8, type=int) ##########
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size', default=-1, type=int)  #fixme!!!!!!!!!!!!
    parser.add_argument('--valid-size', default=1000, type=int)
    parser.add_argument('--depot_num', default=10, type=int)  # todo ###############

    # 解析为args
    args = parser.parse_known_args()[0]  # colab环境跑使用
    # --------------------------------------------------------------------
    args.test = False
    # --------------------------------------------------------------------
    upper, lower = 1.1, 0.9
    # 设置checkpoint路径
    share = True   # todo 检查#############
    args.checkpoint = os.path.join( "TW_random_train_log_share", "200", "2_14_23_17_39")
    # todo 需要在这里面完成数据的本地保存（？如果需要多组实验保存的话，也要在这里写）
    path_analysis(upper, lower, share, args)
    # ------------------------------
    share = False
    args.checkpoint = os.path.join("TW_random_train_log", "200", "2_14_23_16_34")
    path_analysis(upper, lower, share, args)

    '''
    est_upper= 1.1
    est_lower= 0.9
    SHARE:
    args.num_city=50
    args.depot_num=5
    开始测试：args.valid_size=1000
    mean of tower distance is 7.733287194252014
    mean of total distance is 12.123580431938171
    Percentage of tower distance ↑ = 63.79%
    mean of charge_time is 8.564
    
    INDEPENDENT:
    args.num_city=50
    args.depot_num=5
    开始测试：args.valid_size=1000
    mean of tower distance is 7.8749401898384095
    mean of total distance is 12.770562036037445
    Percentage of tower distance ↑ = 61.66%
    mean of charge_time is 9.313
    '''

    '''
    
    
    Not Shared depot.
    est_upper= 1.1
    est_lower= 0.9
    args.num_city=200
    args.depot_num=20
    开始测试：args.valid_size=1000
    mean of tower distance is 18.750715662002563
    mean of total distance is 29.040897727966307
    Percentage of tower distance ↑ = 64.57%
    mean of charge_time is 20.597
    '''
    #--------------------------------------------------------------

    print("Running ends.")