import random

import numpy as np
import matplotlib.pyplot as plt

# todo：添加注释、检查约束是否正确、调整为我的问题的约束、
# todo 加入时间波动！！考虑时间实时？？。
# todo 写share!!!！
# todo: 嵌入对比代码!!!!和greedy速速比较（注意随机种子
# todo 统计指标的对比、输出format


def VNS_VRP_Problem(MAX_ITER, uav_num, tower_num, position, opt, share_):
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

    # depots_pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(uav_num)}
    # customers_pos = {i: (random.uniform(0, 1), random.uniform(0, 1)) for i in range(tower_num)}

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

    def calculate_cost_path(solution, get_path=False):
        """
        会在这个函数里面对解进行评估。注意solution={depot_i:客户点list}，
        需要按顺序检查电量并且插入充电桩。【注意！需要检查 “去下一个点+回仓库” 是否足够】

        todo 实现波动。
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

                if battery < distance2c2depot + demands[tower_id]:  # 如果电不够去下一个电塔后再充电桩
                    # 返回仓库充电
                    return_distance = get_distance(current_pos, depots_pos[depot_id]) # 立刻回自己仓库
                    battery -= return_distance
                    if battery < 0:
                        raise ValueError(f"返回仓库的时候 battery {battery}<0 ")

                    total_distance += return_distance
                    battery = BATTERY_CAPACITY # 恢复满电
                    current_pos = depots_pos[depot_id]
                    if get_path:
                        path[depot_id].append(depots_pos[depot_id])
                    distance2c = get_distance(current_pos, tower_pos[tower_id])

                total_distance += distance2c
                battery -= (distance2c + demands[tower_id])
                if battery < 0:
                    raise ValueError(f"battery={battery} <0")
                visited_plan[tower_id] = True
                current_pos = tower_pos[tower_id]
                if get_path:
                    path[depot_id].append(tower_pos[tower_id])
            # 返回仓库
            dis2depot = get_distance(current_pos, depots_pos[depot_id])
            total_distance += dis2depot
            battery -= dis2depot
            if battery < 0:
                raise ValueError(f"返回仓库的时候 battery {battery}<0 ")
            if get_path:
                path[depot_id].append(depots_pos[depot_id])

        if not all(visited_plan.values()):
            print("visited_plan.values(=", visited_plan.values())  # todo 以后大量地图的时候看看会不会报错。
            total_distance = np.inf

        if get_path:
            return total_distance, path
        else:
            return total_distance

    # 2-opt局部优化
    def two_opt(route, depot_id):
        best = route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 1):
                for j in range(i + 1, len(route)):
                    new_route = route[:i] + route[i:j][::-1] + route[j:]  # 把ij中间倒置
                    new_cost = calculate_cost_path({depot_id: new_route})
                    if new_cost < calculate_cost_path({depot_id: route}):
                        best = new_route
                        improved = True
            route = best
        return best

    def three_opt(route, depot_id):
        """
        3-opt局部优化：通过交换路径中的三段边来优化总距离
        参数：
          - route: 当前路径（客户ID列表）
          - depot_id: 所属仓库ID（用于计算距离）
        返回优化后的路径
        """
        best_route = route.copy()
        best_cost = calculate_cost_path({depot_id: best_route}) # fixme
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
                        cost1 = calculate_cost_path({depot_id:new_route1})# fixme

                        # 方式2: A-B-D-C-E-F (反转C-D)
                        new_route2 = route[:j] + route[j:k][::-1] + route[k:]
                        cost2 = calculate_cost_path({depot_id:new_route2})# fixme

                        # 方式3: A-C-D-B-E-F (交换B-C和D-E)
                        new_route3 = route[:i] + route[j:k] + route[i:j] + route[k:]
                        cost3 = calculate_cost_path({depot_id:new_route3})# fixme

                        # 方式4: A-D-E-B-C-F (复杂重组)
                        new_route4 = route[:i] + route[j:k] + route[i:j][::-1] + route[k:]
                        cost4 = calculate_cost_path({depot_id:new_route4})# fixme

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
        plt.figure(figsize=(15, 5))

        # 收敛曲线
        plt.subplot(131)
        plt.plot(history_)
        print("history length=", len(history_))
        plt.title(f"Convergence (best={cost})")
        plt.xlabel("Iteration")
        plt.ylabel("Total Distance")

        # 仓库和客户分布
        plt.subplot(132)
        colors = plt.cm.tab20.colors
        for depot_id, (x, y) in depots_pos.items():
            plt.scatter(x, y, c=colors[depot_id], s=200, marker='s', label=f"Depot {depot_id}")
        for cust_id, (x, y) in tower_pos.items():
            plt.scatter(x, y, c='black', s=50, alpha=0.5)
        plt.title("Depots and Customers")
        plt.legend()

        # _, feasible_solution = calculate_cost_path(solution, get_path=True)
        # 路径可视化
        plt.subplot(133)
        for depot_id, path in feasible_solution.items():
            color = colors[depot_id]
            # path = [depots[depot_id]] + [customers_pos[c] for c in route] + [depots[depot_id]] # 没有画出来中间返回仓库吗！
            path = feasible_solution[depot_id]
            x = [p[0] for p in path]
            y = [p[1] for p in path]
            plt.plot(x, y, marker='o', color=color, linewidth=2, markersize=6)
        plt.title("Optimized Routes")

        plt.tight_layout()
        if save:
            title=f"{uav_num}U {tower_num}T share={share_}.jpg"
            plt.savefig(title)
            print(f"picture save as {title}")
        plt.show()



    def vns():
        current_solution = initial_solution()
        best_solution = {key: v.copy() for key, v in current_solution.items()}
        best_cost=calculate_cost_path(best_solution)
        cost_history = []

        if opt == 2:
            opt_function = two_opt
        elif opt==3:
            opt_function =three_opt
        else:
            raise ValueError("不能识别要使用的opt函数")

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
                    shaken_solution[depot_id] = opt_function(shaken_solution[depot_id], depot_id)#

                current_cost = calculate_cost_path(current_solution)
                shaken_cost = calculate_cost_path(shaken_solution)

                if shaken_cost < current_cost:
                    current_solution = shaken_solution
                    # print(f"{t} iteration: shaken_cost {shaken_cost} < current_cost {current_cost}")
                    k = 1  # 重置扰动强度
                    best_solution = {key: v.copy() for key, v in shaken_solution.items()}
                    best_cost= shaken_cost
                    # cost_history.append(best_cost) # 这个放外面就行了吧？
                else:
                    k += 1  # 增加扰动强度
            cost_history.append(best_cost)

        best_cost,feasible_best_solution = calculate_cost_path(best_solution,get_path=True)
        print(f"最优总距离: {best_cost:.2f}")
        return best_cost, feasible_best_solution, cost_history


    distance, feasible_best_solu, history = vns()

    # visualize(feasible_best_solu, history, cost=distance,save=False)

    return distance, feasible_best_solu, history

def multi_converge_visualize(datasize,uav_num,tower_num,share_,history_, mean_cost=-1, save=False):
    """
    用于一次运行多个地图的时候，合并收敛历史，画出收敛图。
    """
    # plt.figure(figsize=(10, 5))

    # 收敛曲线
    # plt.subplot(131)
    plt.plot(history_)
    plt.title(f"Convergence mean(best)={mean_cost}")
    plt.xlabel("Iteration")
    plt.ylabel("Total Distance")

    plt.tight_layout()
    if save:
        title=f"{datasize}MAPS:{uav_num}U {tower_num}T share={share_}.jpg"
        plt.savefig(title)
        print(f"picture save as {title}")
    plt.show()

def run_vns_VRP(max_iteration, position_set_, tower_num, uav_num, opt,share_):
    """
    适用于一次性运行n个VNS算法VRP问题实例。是用于把RL的数据集传进来训练的接口。
    position_set: 为多个地图的集合。
    """
    print("Run VNS set:")
    print(f"max_iteration={max_iteration}")
    print(f"uav num={uav_num}\ntower_num={tower_num}")
    cost_set = []

    if type(position_set_) is not np.ndarray:
        position_set_ = position_set_.numpy()
    position_set_ = position_set_.transpose(0, 2, 1)
    run_time = position_set_.shape[0]

    mean_converge_history=np.zeros(max_iteration)

    for t in range(run_time):
        position = position_set_[t]  # 使用外界传进来的坐标
        cost, solution, converge_history = VNS_VRP_Problem(max_iteration, uav_num, tower_num, position,opt=opt, share_=False)
        cost_set.append(cost)
        mean_converge_history+=np.array(converge_history)

    if run_time>1:
        mean_converge_history/=run_time
        mean_cost= sum(cost_set) / run_time
        multi_converge_visualize(run_time,uav_num,tower_num, share_,
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
    max_iter=50 # VNS 迭代次数。
    run_times=10
    tower_n=50
    uav_n=5
    share=False
    np.random.seed(123) # todo 随机种子！！方便debug。
    position_set = np.random.random(size=(run_times, 2, tower_n + uav_n))

    # 一次性运行多个地图实例。
    cost_set = run_vns_VRP(max_iter, position_set, tower_n, uav_n, opt=2 ,share_= share)

    print(f"Run {run_times} times. VNS Average tour length: {np.mean(cost_set)}")
