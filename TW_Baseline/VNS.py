import random

import matplotlib.pyplot as plt
import numpy as np

from TW_Baseline.草稿 import calculate_cost_path_share


# todo：添加注释、检查约束是否正确、调整为我的问题的约束、
# todo 写share!!!！
# todo: 嵌入对比代码!!!!和greedy速速比较（注意随机种子
# todo 统计指标的对比、输出format
# todo share 和indepoent 合并。

def VNS_VRP_Problem(MAX_ITER, uav_num, tower_num, position, opt, share_,draw_path):
    # 参数设置
    BATTERY_CAPACITY = 2.8  # 电池容量
    # DEMAND_RANGE = (0.1, 0.2)  # 需求范围
    DEMAND=0.1
    SHAKE_STRENGTH = 3  # 扰动强度（交换客户次数）
    upper_bound = 1.1
    lower_bound = 0.9



    # 随机生成客户和仓库坐标（地图1x1）
    depots_pos = {i: position[i][:] for i in range(uav_num)}
    tower_pos = {i: position[uav_num+i][:] for i in range(tower_num)}

    # demands = {i: round(random.uniform(*DEMAND_RANGE), 2) for i in tower_pos} # 动态的需求版本
    # print("动态的需求！")
    demands = {i: DEMAND for i in tower_pos}  # 静态的需求版本
    print("静态的需求！")

    # 计算距离矩阵
    def get_distance(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    # 初始化解：每个仓库分配一辆车，随机分配客户
    def initial_solution():
        solution = {depot_id: [] for depot_id in depots_pos}
        customer_list = list(tower_pos.keys())
        random.shuffle(customer_list)

        # 平均分配客户到各仓库车辆
        chunk_size = len(customer_list) // uav_num
        for i, depot_id in enumerate(solution.keys()):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i != uav_num - 1 else len(customer_list)
            solution[depot_id] = customer_list[start:end]
        return solution

    def shake(solution, k):
        new_solution = {key: v.copy() for key, v in solution.items()}
        for _ in range(k):
            # 随机选择两个仓库的两个客户，交换
            depot1, depot2 = random.sample(list(new_solution.keys()), 2)
            if len(new_solution[depot1]) == 0 or len(new_solution[depot2]) == 0:
                continue
            idx1 = random.randint(0, len(new_solution[depot1]) - 1)
            idx2 = random.randint(0, len(new_solution[depot2]) - 1)
            new_solution[depot1][idx1], new_solution[depot2][idx2] = new_solution[depot2][idx2], new_solution[depot1][
                idx1]
        return new_solution


    def calculate_cost_path_independent(solution, get_path=False):
        """
        会在这个函数里面对解进行评估。注意solution={depot_i:客户点list}，
        需要按顺序检查电量并且插入充电桩。
        需要检查 “去下一个点+回仓库” 是否足够

        input: 解
        output: 解的评估的质量，包含仓库的路径点xy列表。
        """
        total_distance = 0
        visited_plan = {cu: False for p in solution.values() for cu in p}

        if get_path:
            path = {depot_id: [] for depot_id in range(uav_num)}

        for depot_id, route in solution.items():
            current_pos = depots_pos[depot_id]  # 初始状态是从仓库出发
            if get_path:
                path[depot_id].append(depots_pos[depot_id])
            battery = BATTERY_CAPACITY  # 满电

            for tower_id in route:
                distance2c = get_distance(current_pos, tower_pos[tower_id])  # 去下一个点。
                distance2c2depot = distance2c + get_distance(tower_pos[tower_id], depots_pos[depot_id]) #下一个点回自己仓库

                if battery < upper_bound * distance2c2depot + demands[tower_id]:  # 如果电不够去下一个电塔后再充电桩 # todo 已波动修改
                    # 返回仓库充电
                    return_distance = get_distance(current_pos, depots_pos[depot_id]) # 立刻回自己仓库
                    # rand_factor = (lower_bound + np.random.rand() * (upper_bound - lower_bound)) # todo 已波动修改
                    battery -= return_distance * upper_bound#  todo 已波动修改
                    if battery < 0:
                        raise ValueError(f"返回仓库的时候 battery {battery}<0 ")

                    total_distance += return_distance
                    battery = BATTERY_CAPACITY # 恢复满电
                    current_pos = depots_pos[depot_id]


                    if get_path:
                        path[depot_id].append(depots_pos[depot_id])
                    distance2c = get_distance(current_pos, tower_pos[tower_id])
                    distance2c2depot = distance2c + get_distance(tower_pos[tower_id], depots_pos[depot_id])
                    if battery < upper_bound * distance2c2depot + demands[tower_id]:
                        # print("无法出发")
                        break #如果无法出发：直接break。会因为没访问完城市被标记cost=inf

                # rand_factor = (lower_bound + np.random.rand() * (upper_bound - lower_bound))  # todo 已波动修改
                total_distance += distance2c
                battery -= (distance2c * upper_bound + demands[tower_id]) # todo 已波动修改
                if battery < 0:
                    raise ValueError(f"battery={battery} <0")
                visited_plan[tower_id] = True
                current_pos = tower_pos[tower_id]
                if get_path:
                    path[depot_id].append(tower_pos[tower_id])

            # 返回仓库
            # rand_factor = (lower_bound + np.random.rand() * (upper_bound - lower_bound))# todo 已波动修改
            dis2depot = get_distance(current_pos, depots_pos[depot_id])
            total_distance += dis2depot
            battery -= dis2depot * upper_bound # todo 已波动修改
            if battery < 0:
                raise ValueError(f"返回仓库的时候 battery {battery}<0 ")
            if get_path:
                path[depot_id].append(depots_pos[depot_id])

        if not all(visited_plan.values()):
            # print("visited_plan.values=", visited_plan.values())
            total_distance = np.inf

        if get_path:
            return total_distance, path
        else:
            return total_distance

    def calculate_cost_path_share(solution, get_path=False):
        """
        [share]函数！！！！！！！
        会在这个函数里面对解进行评估。注意solution={depot_i:客户点list}，
        需要按顺序检查电量并且插入充电桩。
        需要检查 “去下一个点+回仓库” 是否足够

        input: 解
        output: 解的评估的质量，包含仓库的路径点xy列表。

        伪代码：
         弹出第一个需要决策的无人机：取出当前的位置、电量
         如果决策时间是inf则标记完成了。
         是否还有下一个tower：如果没有，则准备回仓库(检查是否有电。回去，标记占用，决策时间inf）。
               如果有： 准备去下一个点。判断电量：是否足够过去下一个点+下一个点到最远充电桩+demand
                    如果电量够：（如果在仓库，就要取消占用）直接过去，更新当前位置、电量、当前tour id++，更新决策时间++
                    如果不够：根据当前位置，找到最近的空仓库。判断是否足够电回仓库（不够电则inf，break）然后回去，标记占用，决策时间++。
                            然后暂时不去当前tower。
                    把无人机和更新的决策时间放回去。


        """
        def get_empty_depot(cur_pos, in_depot):
            # 返回空仓库序列？   输入是当前位置和当前无人机组状态。根据位置和仓库状态计算出{depot_id:distance}
            all_d= set([i for i in range(uav_num)])
            # print(f"all_d={all_d}")
            # print(f"in_depot={in_depot}")
            unavailable_d = set([i for i in in_depot.values() if i is not False])
            # print(f"unavailable_d={unavailable_d}")

            empty_depot= all_d - unavailable_d
            # print(f"empty_depot={empty_depot}")
            empty_depot={d:depots_pos[d] for d in empty_depot}
            empty_depot_dis={k:get_distance(cur_pos,dpos) for k,dpos in empty_depot.items()}
            empty_depot_dis=sorted(empty_depot_dis.items(),key=lambda x:x[1])
            # print(f"empty_depot_dis={empty_depot_dis}")
            return empty_depot_dis

        def get_farthest_depot(cur_pos):
            # 返回最远的仓库的距离？？   输入是当前位置
            # empty_depot=set(i for i in range(uav_num)) - set(in_depot)
            # print(f"empty_depot={empty_depot}")
            # empty_depot={d:depots_pos[d] }
            empty_depot_dis=[get_distance(cur_pos,depots_pos[d]) for d in range(uav_num)]
            farthest=max(empty_depot_dis)
            # sorted(empty_depot_dis,key=lambda x:x[1])
            # print(f"farthest={farthest}")
            return farthest

        total_distance = 0
        visited_plan = {cu: False for p in solution.values() for cu in p}

        if get_path:
            path = {depot_id: [depots_pos[depot_id]] for depot_id in range(uav_num)} # 初始的时候装填仓库位置

        decision_order = [(depot_id,0) for depot_id in solution.keys()]
        current_tour_p={k:0 for k in solution.keys()} # 记录每个无人机当前tour到哪个点了。 ???还是记录当前坐标呢
        # in_depot=[depot_id for depot_id in range(len(solution))]
        in_depot={k:k for k in solution.keys()} # 记录是否在仓库（初始为是），如果是，则元素是仓库id。否则是False
        remain_battery={k:BATTERY_CAPACITY for k in solution.keys() }

        while True: #先不写停止标准
            # 排序
            decision_order.sort(key=lambda x:x[1])
            # print(f"decision_order={decision_order}")
            uav_id, now = decision_order.pop(0)
            if now == np.inf: # 说明所有无人机都回到仓库了。
                # print("结束了捏")
                break # 退出while

            route = solution[uav_id]
            battery = remain_battery[uav_id]

            if in_depot[uav_id] is not False: # 如果当前在仓库。获取xy位置
                current_pos = depots_pos[in_depot[uav_id]]
            else:
                current_pos = tower_pos [route[current_tour_p[uav_id]-1]]

            if current_tour_p[uav_id] == len(route):# 如果没有下一个tower了
                # print(f"没有下一个点了, {uav_id}要回仓库了！！！")
                empty_depots=get_empty_depot(current_pos,in_depot)
                depot_id = empty_depots[0][0]  #选最近的空depot id
                return_distance = empty_depots[0][1]
                battery -= return_distance * upper_bound  # todo 已波动修改
                total_distance += return_distance
                if battery < 0:
                    raise ValueError(f"返回仓库的时候 battery {battery}<0 ")
                if get_path:
                    path[uav_id].append(depots_pos[depot_id])

                battery = BATTERY_CAPACITY  # 恢复满电
                # 更新状态
                # current_tour_p[uav_id]+=1 # 试试不变会不会有bug
                in_depot[uav_id] = depot_id
                remain_battery[uav_id]= battery
                decision_order.append((uav_id,np.inf))
            else: # 如果还有下一个tower要去
                tower_id = route[current_tour_p[uav_id]] # 下一个tower点。
                distance2c = get_distance(current_pos, tower_pos[tower_id])
                # 看看是否能去最远的仓库（不管是否空仓库）
                c2farthest = get_farthest_depot(tower_pos[tower_id])
                distance2c2depot = distance2c + c2farthest #下一个点回最远仓库

                if battery < upper_bound * distance2c2depot + demands[tower_id]:  # 如果电不够去下一个电塔后再充电桩 # todo 已波动修改
                    if in_depot[uav_id] is not False:
                        # print("明明在充电桩但是却不能出发……")
                        total_distance = np.inf
                        break

                    # 返回仓库充电
                    empty_depots = get_empty_depot(current_pos, in_depot)
                    depot_id = empty_depots[0][0]  # 选最近的depot id
                    return_distance = empty_depots[0][1]
                    battery -= return_distance * upper_bound #  todo 已波动修改
                    if battery < 0:
                        raise ValueError(f"返回仓库的时候 battery {battery}<0 ")
                    total_distance += return_distance
                    battery = BATTERY_CAPACITY # 恢复满电
                    if get_path:
                        path[uav_id].append(depots_pos[depot_id])
                    # 更新状态
                    # current_tour_p[uav_id] += 1 # 这个不变
                    in_depot[uav_id] = depot_id
                    remain_battery[uav_id] = battery
                    decision_order.append((uav_id, now+ return_distance * upper_bound))

                else: # 是电量足够直接去下一个电塔。
                    battery -= distance2c * upper_bound  # todo 已波动修改
                    if battery < 0:
                        raise ValueError(f"返回仓库的时候 battery {battery}<0 ")
                    total_distance += distance2c
                    if get_path:
                        path[uav_id].append(tower_pos[tower_id])
                    # 更新状态
                    current_tour_p[uav_id] += 1
                    in_depot[uav_id] = False
                    visited_plan[tower_id] = True
                    remain_battery[uav_id] = battery
                    decision_order.append((uav_id, now + distance2c * upper_bound))

        if not all(visited_plan.values()):
            # print("visited_plan.values=", visited_plan.values())
            total_distance = np.inf

        if get_path:
            return total_distance, path
        else:
            return total_distance
    # 2-opt局部优化
    def two_opt(route, depot_id,cost_path_fun):
        best = route
        best_cost= cost_path_fun({depot_id: best})
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j][::-1] + route[j:]  # 把ij中间倒置
                    new_cost = cost_path_fun({depot_id: new_route})
                    if new_cost < best_cost:
                        best = new_route
                        best_cost = new_cost
                        improved = True

            route = best
        return best

    def three_opt(route, depot_id,cost_path_fun):
        """
        3-opt局部优化：通过交换路径中的三段边来优化总距离
        参数：
          - route: 当前路径（客户ID列表）
          - depot_id: 所属仓库ID（用于计算距离）
        返回优化后的路径
        """
        best_route = route.copy()
        best_cost = cost_path_fun({depot_id: best_route})
        improved = True

        while improved:
            improved = False
            for i in range(1, len(route) - 4):
                for j in range(i + 2, len(route) - 2):
                    for k in range(j + 2, len(route)):
                        # 原始路径片段：A-B-C-D-E-F
                        a, b, c, d, e, f = route[i - 1], route[i], route[j - 1], route[j], route[k - 1], route[k]

                        # 计算原始片段总距离
                        original_cost = (
                                get_distance(tower_pos[a], tower_pos[b]) +
                                get_distance(tower_pos[c], tower_pos[d]) +
                                get_distance(tower_pos[e], tower_pos[f])
                        )

                        # 生成6种可能的3-opt交换方式（实际有效4种）
                        # 方式1: A-C-B-D-E-F (反转B-C)
                        new_route1 = route[:i] + route[i:j][::-1] + route[j:]
                        cost1 = cost_path_fun({depot_id:new_route1})

                        # 方式2: A-B-D-C-E-F (反转C-D)
                        new_route2 = route[:j] + route[j:k][::-1] + route[k:]
                        cost2 = cost_path_fun({depot_id:new_route2})

                        # 方式3: A-C-D-B-E-F (交换B-C和D-E)
                        new_route3 = route[:i] + route[j:k] + route[i:j] + route[k:]
                        cost3 = cost_path_fun({depot_id:new_route3})

                        # 方式4: A-D-E-B-C-F (复杂重组)
                        new_route4 = route[:i] + route[j:k] + route[i:j][::-1] + route[k:]
                        cost4 = cost_path_fun({depot_id:new_route4})

                        # 选择最优改进
                        min_cost = min(cost1, cost2, cost3, cost4)
                        if min_cost < best_cost:
                            if min_cost == cost1:
                                best_route = new_route1
                            elif min_cost == cost2:
                                best_route = new_route2
                            elif min_cost == cost3:
                                best_route = new_route3
                            else:
                                best_route = new_route4
                            best_cost = min_cost
                            improved = True
                            break  # 退出当前循环层级以应用改进
                    if improved:
                        break
                if improved:
                    break
            route = best_route  # 更新当前路径
        return best_route

    def visualize(feasible_solution, history_, cost=-1, save=False):
        plt.figure(figsize=(10, 5))

        # 收敛曲线
        plt.subplot(121)
        plt.plot(history_)
        # print("history length=", len(history_))
        plt.title(f"Convergence (best={cost})")
        plt.xlabel("Iteration")
        plt.ylabel("Total Distance")

        # # 仓库和客户分布
        # plt.subplot(132)
        colors = plt.cm.tab20.colors
        # for depot_id, (x, y) in depots_pos.items():
        #     plt.scatter(x, y, c=colors[depot_id], s=200, marker='s', label=f"Depot {depot_id}")
        # for cust_id, (x, y) in tower_pos.items():
        #     plt.scatter(x, y, c='black', s=50, alpha=0.5)
        # plt.title("Depots and Customers")
        # plt.legend()

        # 路径可视化
        plt.subplot(122)

        for depot_id, path in feasible_solution.items():
            color = colors[depot_id]
            path = feasible_solution[depot_id]
            x = [p[0] for p in path]
            y = [p[1] for p in path]
            plt.plot(x, y, marker='o', color=color, linewidth=1.7, markersize=3)

        for depot_id, (x, y) in depots_pos.items():
            plt.scatter(x, y, color=colors[depot_id], s=100, marker='*', label=f"Depot {depot_id}")
        plt.title("Optimized Routes")

        plt.tight_layout()
        if save:
            title=f"{uav_num}U {tower_num}T opt={opt} share={share_}.jpg"
            plt.savefig(title)
            print(f"picture save as {title}")
        plt.show()



    def vns():

        if share_:
            calculate_cost_path_fun = calculate_cost_path_share
        else:
            calculate_cost_path_fun = calculate_cost_path_independent

        opt_function=[]
        if 2 in opt:
            opt_function.append(two_opt)
        elif 3 in opt :
            opt_function.append(three_opt)

        current_solution = initial_solution()
        best_solution = {key: v.copy() for key, v in current_solution.items()}
        best_cost=calculate_cost_path_fun(best_solution)
        cost_history = []




        for t in range(MAX_ITER):
            progress_bar(t+1, MAX_ITER)
            # time.sleep(0.05)

            k = 1
            print(f" {t+1} iteration:",end="")
            print(f" best cost={round(best_cost, 3)}", end="")
            sys.stdout.flush()
            while k <= SHAKE_STRENGTH:
                # 扰动生成新解
                shaken_solution = shake(current_solution, k)
                # 局部优化（对每条路径应用2-opt）
                for depot_id in shaken_solution:
                    # shaken_solution[depot_id] = two_opt(shaken_solution[depot_id], depot_id) #  2 OPT
                    # shaken_solution[depot_id] = three_opt(shaken_solution[depot_id], depot_id)# 3 OPT

                    for opt_fun in opt_function: # 遍历operator
                        shaken_solution[depot_id] = opt_fun(shaken_solution[depot_id], depot_id,calculate_cost_path_fun)

                current_cost = calculate_cost_path_fun(current_solution)
                shaken_cost = calculate_cost_path_fun(shaken_solution)

                if shaken_cost < current_cost:
                    current_solution = shaken_solution
                    # print(f"{t} iteration: shaken_cost {shaken_cost} < current_cost {current_cost}")
                    k = 1  # 重置扰动强度
                    if shaken_cost < best_cost:
                        best_cost = shaken_cost
                        best_solution = {key: v.copy() for key, v in shaken_solution.items()}
                    # cost_history.append(best_cost) # 这个放外面就行了吧？
                else:
                    k += 1  # 增加扰动强度
            cost_history.append(best_cost)

        best_cost,feasible_best_solution = calculate_cost_path_fun(best_solution,get_path=True)
        print(f"最优总距离: {best_cost:.2f}")
        return best_cost, feasible_best_solution, cost_history


    distance, feasible_best_solu, history = vns()

    if draw_path:
        visualize(feasible_best_solu, history, cost=distance, save=False)

    return distance, feasible_best_solu, history


def multi_converge_visualize(datasize, uav_num,tower_num,opt,share_,history_, mean_cost=-1, save=False):
    """
    用于一次运行多个地图的时候，合并收敛历史，画出收敛图。
    """

    # 收敛曲线
    # plt.subplot(131)
    plt.plot(history_)
    plt.title(f"Convergence mean(best)={mean_cost}")
    plt.xlabel("Iteration")
    plt.ylabel("Total Distance")

    plt.tight_layout()
    if save:
        title=f"{datasize}MAPS:{uav_num}U {tower_num}T opt={opt} share={share_}.jpg"
        plt.savefig(title)
        print(f"picture save as {title}")
    plt.show()

def run_VNS_vrp(max_iteration, position_set_, tower_num, uav_num, opt, share_):
    """
    适用于一次性运行n个VNS算法VRP问题实例。是用于把RL的数据集传进来训练的接口。
    position_set: 为多个地图的集合。
    """
    print("Run VNS set:")
    print(f"max_iteration={max_iteration}")
    print(f"SHARE={share_}")
    print(f"uav num={uav_num}\ntower_num={tower_num}")
    print(f"opt function={opt}")
    cost_set = []

    if type(position_set_) is not np.ndarray:
        position_set_ = position_set_.numpy()
    position_set_ = position_set_.transpose(0, 2, 1)
    run_time = position_set_.shape[0]
    print(f"数据集大小={run_time}")
    mean_converge_history=np.zeros(max_iteration)


    for t in range(run_time):
        position = position_set_[t]  # 使用外界传进来的坐标
        print(f"地图样本{t}:",end="")
        cost, solution, converge_history = VNS_VRP_Problem(max_iteration, uav_num, tower_num, position,
                                                           opt=opt, share_=share_,
                                                           draw_path= False if run_time > 1 else True)
        cost_set.append(cost)
        mean_converge_history+=np.array(converge_history)

    if run_time>1:
        mean_converge_history/=run_time
        mean_cost= sum(cost_set) / run_time
        multi_converge_visualize(run_time,uav_num,tower_num, opt,share_,
                                 mean_converge_history,mean_cost,save=False)

    # average_reward = sum(reward_set) / run_time
    return cost_set


import sys

def progress_bar(finish_tasks_number, tasks_number):
    print("\r", end="")
    percentage = round(finish_tasks_number / tasks_number * 100)
    print("\r进度: {}%: ".format(percentage), "▓" * (percentage // 2), " "*(100//2- percentage//2), end="|")
    sys.stdout.flush()


# 主程序
if __name__ == "__main__":
    max_iter=10 # VNS 迭代次数。
    run_times=1
    tower_n=50
    uav_n=5
    share=True
    opt_function=[2,3]
    np.random.seed(123) # todo 随机种子！！方便debug。
    position_set = np.random.random(size=(run_times, 2, tower_n + uav_n))

    # 一次性运行多个地图实例。
    cost_set = run_VNS_vrp(max_iter, position_set, tower_n, uav_n, opt=opt_function, share_= share)
    print(f"Run {run_times} times. VNS Average tour length: {np.mean(cost_set)}")
