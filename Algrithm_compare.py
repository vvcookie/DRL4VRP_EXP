import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2" # todo 记得修改不同的gpu编号
import argparse
import numpy as np
import matplotlib.pyplot as plt
# ----------------------RL---------------------------------
from 时间窗口TW.DRL_TW_random import run_RL_test_exp
from 时间窗口TW.DRL_TW_random_Over_Est import run_RL_test_exp as run_RL_overest
from 时间窗口TW.DRL_TW_random import VehicleRoutingDataset
# --------------------Greedy-----------------------------------
from 时间窗口TW.Greedy_VRP_TW import run_greedy_VRP
from 时间窗口TW.Greedy_over_est import run_greedy_VRP as run_greedy_overestimate
# ----------------------VNS-GVNS--------------------------------
from TW_Baseline.VNS距离矩阵版 import run_VNS_vrp
# ------------------------GA-----------------------------
from TW_Baseline.GA import run_GA_vrp

def test_generalization_uav_change(upper,lower,shared, run_alg_name,_save_dir):
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=True)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_city', default=200, type=int)
    parser.add_argument('--actor_lr', default=1e-4, type=float)  # 学习率，现在在训练第4epoch，我手动改了一下
    parser.add_argument('--critic_lr', default=1e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size', default=-1, type=int)
    parser.add_argument('--valid-size', default=2, type=int)
    parser.add_argument('--depot_num', default=-1, type=int)
    # 解析为args
    args = parser.parse_known_args()[0]  # colab环境跑使用


    if share:
        args.checkpoint = os.path.join("时间窗口TW","TW_random_train_log_share","200","2_14_23_17_39")

    else:
        args.checkpoint = os.path.join("时间窗口TW","TW_random_train_log","200","2_14_23_16_34")

    print("比较算法：",run_alg_name)
    reward_list_dict={}
    uav_list=list(range(4, 21,2))
    print(f"样本数量={args.valid_size} Tower={args.num_city},UAV_list={uav_list}")


    for uav_n in uav_list:
        args.depot_num=uav_n
        reward_dict,avg_time_dict = run_multi_alg_test(upper,lower,shared, args, algorithm=run_alg_name)
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
    alg_vs_name=" VS ".join(run_alg_name)
    plt.title(f"{alg_vs_name} on {args.num_city} tower (share={shared})")
    dir = os.path.join(_save_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(os.path.join(dir, f"{alg_vs_name} on {args.num_city} Tower (share={shared}).png"))
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

    try:
        # 存平均时间 avg_time_dict
        time_filename = f"Time_compare on {args.num_city} T (share={str(shared)}).csv"
        txt = ",".join(avg_time_dict.keys()) + "\n"
        # for greedy, RL in zip(avg_R_Greedy, avg_R_RL):
        for rs in zip(*avg_time_dict.values()):
            rs = list(map(str, list(rs)))
            txt += ",".join(rs)
            txt += "\n"
        with open(os.path.join(dir, time_filename), "w") as f2:
            f2.write(txt)
        print(txt)
    except:
        print("存时间有错。")
        pass

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
    parser.add_argument('--valid-size', default=100, type=int)
    parser.add_argument('--depot_num', default=20, type=int)
    # 解析为args
    args = parser.parse_known_args()[0]  # colab环境跑使用

    args.test = True

    if share:
        args.checkpoint = os.path.join("时间窗口TW", "TW_random_train_log_share", "200", "2_14_23_17_39")
    else:
        args.checkpoint = os.path.join("时间窗口TW", "TW_random_train_log", "200", "2_14_23_16_34")

    print("比较算法：", run_alg_name)
    reward_list_dict = {}
    time_list_dict={}
    tower_list = list(range(40, 201, 40))
    print(f"样本数量={args.valid_size} UAV={args.depot_num},Tower_list={tower_list}")

    dir = os.path.join(_save_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)

    for tower_n in tower_list:
        args.num_city = tower_n
        # reward_rl, reward_greedy = run_exp(share, args)
        reward_dict,avg_time_dict= run_multi_alg_test(upper,lower, share, args, algorithm=run_alg_name)
        for key, val in reward_dict.items():
            reward_dict[key]=np.array(val)*10
        for key, val in reward_dict.items():
            reward_list_dict.setdefault(key, [])
            reward_list_dict[key].append(np.mean(val))  # 对每一个算法的当前数据组合的reward集合计算平均值。
            print(f"{key}:{reward_list_dict[key]}")

            time_list_dict.setdefault(key, [])  # 对每一个算法的当前数据组合的时间的保存
            time_list_dict[key].append(avg_time_dict[key])

        # # 储存不同算法的每一个instance的reawrd
        filename = f"reward on {tower_n}T {args.depot_num} UAV (share={str(share)}).csv"
        txt = ",".join(reward_list_dict.keys()) + "\n"  # 标题……。
        for rewardss in zip(*reward_dict.values()):
            rs = list(map(str, list(rewardss)))
            txt += ",".join(rs)
            txt += "\n"
        with open(os.path.join(dir, filename), "w") as f:
            f.write(txt)
        f.close()


    # plot画图
    plt.close('all')
    for alg in reward_list_dict.keys():
        plt.plot(tower_list, reward_list_dict[alg], label=f"{alg} average path")
    plt.xlabel('Num of Tower')
    plt.ylabel('Total distance')
    plt.legend()
    alg_vs_name=" VS ".join(run_alg_name)
    plt.title(f"{alg_vs_name} on {args.depot_num} UAV (share={share})")
    # 保存plot画图
    share=str(share)
    plt.savefig(os.path.join(dir, f"{alg_vs_name} on {args.depot_num} UAV (share={share}).png"))
    # plt.show()

    # 储存不同算法的均值比较csv
    reward_filename = f"Generalization_on {args.depot_num} UAV (share={str(share)}).csv"
    txt = ",".join(reward_list_dict.keys()) + "\n"
    for rs in zip(*reward_list_dict.values()):
        rs = list(map(str, list(rs)))
        txt += ",".join(rs)
        txt+="\n"
    with open(os.path.join(dir,reward_filename), "w") as f2:
        f2.write(txt)
    print(txt)
    f2.close()


    # 存平均时间 time_list_dict
    time_filename = f"Time_compare on {args.depot_num} UAV.csv"
    txt = ",".join(time_list_dict.keys()) + "\n"
    # for greedy, RL in zip(avg_R_Greedy, avg_R_RL):
    for rs in zip(*time_list_dict.values()):
        rs = list(map(str, list(rs)))
        txt += ",".join(rs)
        txt += "\n"
    with open(os.path.join(dir, time_filename), "w") as f2:
        f2.write(txt)
    print("--算法平均一组所用时间--")
    print(txt)


def run_multi_alg_test(upper,lower, share_depot, args, algorithm):
    '''
    本函数内部生成测试数据集。
    返回的是{算法：测试结果} 字典
    '''
    if share_depot:
        print("Shared depot.")
    else:
        print("Independent depot.")
    print(f"args.num_city={args.num_city},args.depot_num={args.depot_num}")

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
    algorithm_time={}

    # -----------------------------------------------------------------Greedy
    if "Greedy" in algorithm or "Greedy" in algorithm:
        # from Greedy_VRP_TW_random import run_greedy_VRP
        Greedy_test_reward,greedy_avgtime = run_greedy_VRP(test_data.static, args.num_city, args.depot_num,
                                                    upper, lower, share=share_depot)

        algorithm_result["Greedy"]=Greedy_test_reward
        algorithm_time["Greedy"]=greedy_avgtime

    if "Greedy_Over" in algorithm:
        algorithm_result["Greedy_Over"] = run_greedy_overestimate(test_data.static, args.num_city, args.depot_num,upper,lower,share=share_depot)

    #------------------------------------------------------------------RL
    if "RL" in algorithm:
        # RL_test_reward = run_RL_exp(upper,lower,share_depot,args)
        RL_test_reward,rlavg_time = run_RL_test_exp(upper, lower, share_depot, test_data, args)
        algorithm_result["RL"]= RL_test_reward
        algorithm_time["RL"] = rlavg_time

    if "RL_Over" in algorithm:
        RL_test_reward,rloveravg_time = run_RL_overest(upper, lower, share_depot, test_data, args)
        algorithm_result["RL_Over"] = RL_test_reward
        algorithm_time["RL_Over"] = rloveravg_time

    # ------------------------------------------------------------------VNS
    if "VNS" in algorithm:
        max_iter=alg_para["VNS"]["max_iteration"]
        opt=alg_para["VNS"]["operator"]
        VNS_test_reward,avg_time = run_VNS_vrp(max_iter,test_data.static, args.num_city, args.depot_num,
                                    upper, lower,opt=opt,share_=share_depot)
        algorithm_time["VNS"] = avg_time
        algorithm_result["VNS"]=VNS_test_reward


    if "G_VNS" in algorithm:
        max_iter=alg_para["VNS"]["max_iteration"]
        opt=alg_para["VNS"]["operator"]
        GVNS_test_reward,gavg_time = run_VNS_vrp(max_iter,test_data.static, args.num_city, args.depot_num,
                                    upper, lower,opt=opt,share_=share_depot,initialG=True)
        algorithm_time["G_VNS"] = gavg_time
        algorithm_result["G_VNS"]=GVNS_test_reward

    if "GA" in algorithm:
        GA_args=alg_para["GA"]
        GA_test_reward, GAavg_time =  run_GA_vrp(GA_args, test_data.static, args.depot_num, args.num_city, upper, lower,share_= share)

        algorithm_time["GA"] = GAavg_time
        algorithm_result["GA"] = GA_test_reward


    return  algorithm_result,algorithm_time

def get_alg_parameters(alg_name):
    total={}

    if "VNS" or "G_VNS" in alg_name:
        total["VNS"]={
                        "max_iteration":100,
                        "operator":[2,3]
                      }
    if "GA" in alg_name:
        total["GA"]= {
            "POP_SIZE": 50,  # 种群大小
            "GENERATIONS": 100,  # 迭代次数
            "STAGNATION_LIMIT":10, # 迭代多少次不改进就停止
            "CROSSOVER_RATE": 0.8,  # 交叉概率
            "MUTATION_RATE": 0.08,  # 变异概率  （典型值0.01~0.1）
            "TOURNAMENT_SIZE": 5}

    return total

if __name__ == '__main__':

    upper, lower = 1.1, 0.9
    # 0324是1.2 0.8 固定需求……。

    compare_alg= [
        "RL",
        # "RL_Over",
        # "Greedy",
        # "Greedy_Over"
        # "VNS"
        # "G_VNS"
        # "GA"
    ]

    alg_para=get_alg_parameters(compare_alg)
    print(f"alg_para=\n{alg_para}")
    save_dir = "0601 Generalization_TW_random "+" VS ".join(compare_alg) # todo 记得改！！……
    print(f"Save dir:{save_dir}")

    # share = False
    # test_generalization_uav_change(upper, lower, shared=share, run_alg_name=compare_alg, _save_dir=save_dir)
    # test_generalization_tower_change(upper, lower, share=share, run_alg_name=compare_alg, _save_dir=save_dir)
    #--------------------------------------------------------------------------
    share = True
    # test_generalization_uav_change(upper, lower, shared=share, run_alg_name=compare_alg, _save_dir=save_dir)
    test_generalization_tower_change(upper, lower, share=share, run_alg_name=compare_alg, _save_dir=save_dir)

    print(f"Save dir:{save_dir}")
    print("Running ends.")
