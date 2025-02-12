import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # todo 记得修改不同的gpu编号
import argparse
import numpy as np
import matplotlib.pyplot as plt
# -------------------------------------------------------
from 时间窗口TW.DRL_TW_random_constrain_home_share import *
from 时间窗口TW.Greedy_VRP_TW_random import run_greedy_VRP

def test_generalization_uav_change(upper,lower,shared, run_alg_name,_save_dir):
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_city', default=100, type=int)
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
    parser.add_argument('--valid-size', default=10, type=int)
    parser.add_argument('--depot_num', default=-1, type=int)
    # 解析为args
    args = parser.parse_known_args()[0]  # colab环境跑使用

    if share:
        args.checkpoint = os.path.join("TW_random_Cons_train_log_share","50","2_11_0_21_59")
    else:
        args.checkpoint=os.path.join("TW_random_Cons_train_log","50","2_11_0_20_17")

    print("比较算法：",run_alg_name)
    reward_list_dict={}

    uav_list=list(range(3, 10))
    for uav_n in uav_list:
        args.depot_num=uav_n
        reward_dict = run_multi_alg_test(upper,lower,shared, args, algorithm=run_alg_name)
        for key,val in reward_dict.items():
            reward_list_dict.setdefault(key,[])
            reward_list_dict[key].append(np.mean(val))# 对每一个算法的reward集合计算平均值。
            print(f"{key}:{reward_list_dict[key]}")

    plt.close('all')
    for alg in reward_list_dict.keys():
        plt.plot(uav_list, reward_list_dict[alg], label=f"{alg} average path")
    # plt.plot(uav_list, avg_R_Greedy, label="Greedy average path")
    plt.xlabel('Num of UAV')
    plt.ylabel('Total distance')
    plt.legend()
    plt.title(f"Greedy_VS_RL on {args.num_city} tower (share={shared})")
    dir = os.path.join(_save_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, f"Greedy_VS_RL on {args.num_city} Tower share={shared}.png"))
    # plt.show()

    # 储存csv
    reward_filename = f"Generalization_compare on {args.num_city} T (share={str(shared)}).csv"
    txt = ",".join(reward_list_dict.keys())+ "\n"
    # for greedy, RL in zip(avg_R_Greedy, avg_R_RL):
    for rs in zip(*reward_list_dict.values()):
        rs = list(map(str, list(rs)))
        txt += ",".join(rs)
        txt+="\n"
    with open(os.path.join(dir,reward_filename), "w") as f2:
        f2.write(txt)
    print(txt)

def test_generalization_tower_change(upper,lower,share,run_alg_name,_save_dir):
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
    parser.add_argument('--valid-size', default=10, type=int)
    parser.add_argument('--depot_num', default=5, type=int)
    # 解析为args
    args = parser.parse_known_args()[0]  # colab环境跑使用

    args.test = True

    if share:
        args.checkpoint = os.path.join("TW_random_Cons_train_log_share","50","2_11_0_21_59")
    else:
        args.checkpoint=os.path.join("TW_random_Cons_train_log","50","2_11_0_20_17")

    # run_alg_name = ["Greedy", "RL"]
    print("比较算法：", run_alg_name)
    reward_list_dict = {}

    tower_list = list(range(20, 101, 10))
    for tower_n in tower_list:
        args.num_city = tower_n
        # reward_rl, reward_greedy = run_exp(share, args)
        reward_dict= run_multi_alg_test(upper,lower,share, args, algorithm=run_alg_name)
        for key, val in reward_dict.items():
            reward_list_dict.setdefault(key, [])
            reward_list_dict[key].append(np.mean(val))  # 对每一个算法的reward集合计算平均值。
            print(f"{key}:{reward_list_dict[key]}")

    plt.close('all')
    for alg in reward_list_dict.keys():
        plt.plot(tower_list, reward_list_dict[alg], label=f"{alg} average path")
    plt.xlabel('Num of Tower')
    plt.ylabel('Total distance')
    plt.legend()
    plt.title(f"Greedy_VS_RL on {args.depot_num} UAV (share={share})")

    dir = os.path.join(_save_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    share=str(share)
    plt.savefig(os.path.join(dir, f"Greedy_VS_RL on {args.depot_num} UAV (share={share}).png"))
    # plt.show()

    # 储存csv
    reward_filename = f"Generalization_on {args.depot_num} UAV (share={str(share)}).csv"
    txt = ",".join(reward_list_dict.keys()) + "\n"
    for rs in zip(*reward_list_dict.values()):
        rs = list(map(str, list(rs)))
        txt += ",".join(rs)
        txt+="\n"
    with open(os.path.join(dir,reward_filename), "w") as f2:
        f2.write(txt)
    print(txt)

def run_multi_alg_test(upper,lower,share_depot, args, algorithm):
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

    # -----------------------------------------------------------------Greedy
    if "greedy" in algorithm or "Greedy" in algorithm:
        # from Greedy_VRP_TW_random import run_greedy_VRP

        reward_greedy = run_greedy_VRP(test_data.static, args.num_city, args.depot_num,upper,lower,share=share_depot)
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
                            update_mask_shared_TW_constraint,
                            node_distance_shared,
                            args.num_layers,
                            args.dropout,
                            upper,lower).to(device)
        else:
            actor = DRL4TSP(STATIC_SIZE,
                            DYNAMIC_SIZE,
                            args.hidden_size,
                            car_load,
                            args.depot_num,
                            update_dynamic_independent,
                            update_mask_independent_TW_constraint,
                            node_distance_independent,
                            args.num_layers,
                            args.dropout,
                            upper,lower).to(device)

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

if __name__ == '__main__':
    save_dir = "Generalization_test_TW_random_constrain"
    upper, lower = 1.1, 0.9
    share=True

    test_generalization_uav_change(upper, lower, shared=share, run_alg_name=["Greedy"], _save_dir=save_dir)
    test_generalization_tower_change(upper, lower, share=share, run_alg_name=["Greedy"], _save_dir=save_dir)
    print("Running ends.")
