import random
import numpy as np
import matplotlib.pyplot as plt

#-----------------------

'''
    较小种群（20-50）：适合简单问题，计算速度快，但可能陷入局部最优。
    较大种群（100-500）：适合复杂问题，搜索空间更大，但计算成本较高。
    
    较小迭代次数（50-200）：适合简单问题或快速验证。
    较大迭代次数（500-5000）：适合复杂问题，确保算法充分收敛。

    较低交叉概率（0.6-0.7）：适合问题解空间较小的情况。
    较高交叉概率（0.8-0.9）：适合问题解空间较大的情况。    如果算法过早收敛，可以适当提高交叉概率。
    
    较低变异概率（0.01-0.05）：适合种群多样性较高的情况。
    较高变异概率（0.1-0.2）：适合种群多样性较低的情况。如果算法陷入局部最优，可以适当提高变异概率。
'''








# # 交叉（顺序交叉）旧版，没有保留父代相对顺序，不是经典OX
# def crossover(parent1, parent2):
#     # todo 有问题，要改掉 crossover
#     if random.random() < CROSSOVER_RATE:
#         start, end = sorted(random.sample(range(NUM_CUSTOMERS), 2))
#         child1 = parent1[start:end]
#         child2 = parent2[start:end]
#         for p in [parent2, parent1]:
#             for gene in p:
#                 if gene not in child1:
#                     child1.append(gene)
#                 if gene not in child2:
#                     child2.append(gene)
#         return child1, child2
#     else:
#         return parent1, parent2



# 变异（交换变异）

# 无注释版本
# def mutation(individual):
#     if random.random() < MUTATION_RATE:
#         idx1, idx2 = random.sample(range(NUM_CUSTOMERS), 2)
#         individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
#     return individual








# 主函数
def genetic_algorithm(GA_args, UAV_NUM, TOWER_NUM, position, upper_bound, lower_bound, share_,draw_path):
    """
        主函数：执行遗传算法流程，包括初始化种群、选择、交叉、变异和结果输出
        返回：无
        输出：打印每代最佳适应度，最终解和可视化结果
        """
    MAP_SIZE=1
    BATTERY_CAPACITY=2.8
    DEMAND=0.1
    DEPOT_NEED=0.1
    POP_SIZE = GA_args["POP_SIZE"]
    GENERATIONS = GA_args["GENERATIONS"]
    CROSSOVER_RATE = GA_args["CROSSOVER_RATE"]
    MUTATION_RATE = GA_args["MUTATION_RATE"]
    TOURNAMENT_SIZE = GA_args["TOURNAMENT_SIZE"]
    STAGNATION_LIMIT =  GA_args["STAGNATION_LIMIT"] #10  # 连续10代不改进则停止

    # 随机生成客户需求
    # customer_demands = {i: round(random.uniform(*DEMAND_RANGE), 2) for i in range(1, NUM_CUSTOMERS + 1)}
    demands = {i: DEMAND for i in range(0, TOWER_NUM)}  # todo ？？？

    # 随机生成客户和仓库坐标（地图大小为1x1）
    locations = {}
    tower_pos = []
    depots_pos = []

    # # todo 记得改数据集
    for i in range(UAV_NUM):  # 仓库数量等于车辆数量
        locations[f"Depot_{i}"] = tuple(position[i])  # 仓库坐标
        depots_pos.append(locations[f"Depot_{i}"])

    for i in range(TOWER_NUM):
        locations[f"Customer_{i}"] = tuple(position[i+UAV_NUM])  # 客户坐标
        tower_pos.append(locations[f"Customer_{i}"])


    # 计算距离矩阵
    def calculate_distance_matrix(locations):
        num_points = len(locations)
        distance_matrix = np.zeros((num_points, num_points))
        keys = list(locations.keys())  # 字符串列表
        for i in range(num_points):
            for j in range(num_points):
                x1, y1 = locations[keys[i]]
                x2, y2 = locations[keys[j]]
                distance_matrix[i][j] = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        return distance_matrix, keys
    DISTANCE_MATRIX, LOCATION_KEYS = calculate_distance_matrix(locations)


    def initialize_population():
        """
        初始化种群函数（伪代码风格注释）
        功能：生成包含POP_SIZE个随机个体（客户序列）的初始种群
        输出：population = [[个体1], [个体2], ...]
              其中每个个体 = [客户ID1, 客户ID2, ...] (客户ID范围: 1~NUM_CUSTOMERS)
        """
        population = []  # 空种群（二维列表，最终形态: [[3,1,2], [2,3,1], ...]）

        for _ in range(POP_SIZE):  # POP_SIZE=种群规模（如100）
            individual = list(range(0, TOWER_NUM))  # 基础序列[1,2,...,N]
            random.shuffle(individual)  # 随机打乱生成新个体（如[3,1,2]）
            population.append(individual)  # 加入种群

        return population  # 返回格式示例：[[3,1,2], [2,3,1], ...]

    def mutation(individual):
        """
        交换变异（Swap Mutation）实现 - 适用于路径问题（TSP/VRP）
        参数:
            individual: 待变异个体 [gene1, gene2,...] (gene=客户ID，不含仓库)
            MUTATION_RATE: 变异概率（典型值0.01~0.1）
            NUM_CUSTOMERS: 客户数量（个体长度）
        返回:
            individual: 可能变异后的个体（直接修改输入对象）
        行业标准验证:
            ✓ 符合基本交换变异标准（随机交换两个基因）
            ✓ 保持路径合法性（无重复/丢失客户）
            ✓ 低变异概率（避免破坏优良基因）
        改进建议:
            ▪ 可增加自适应变异率（根据迭代次数动态调整）
            ▪ 可考虑逆转变异（inversion）或插入变异（insertion）
        """
        if random.random() < MUTATION_RATE:  # 按概率执行变异
            # 关键行：随机选择两个不同位置（确保idx1 != idx2）
            idx1, idx2 = random.sample(range(TOWER_NUM), 2)
            # 执行交换：individual=[1,2,3,4], idx1=0, idx2=2 → [3,2,1,4]
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual  # 返回修改后的个体（Python列表为可变对象）
    # # 新版交叉OX
    def crossover(parent1, parent2):
        if random.random() < CROSSOVER_RATE:
            start, end = sorted(random.sample(range(TOWER_NUM), 2))

            def make_child(parent_main, parent_aux):
                child = parent_main[start:end]
                ptr = end % TOWER_NUM
                while len(child) < TOWER_NUM:
                    gene = parent_aux[ptr]
                    if gene not in child:
                        child.append(gene)
                    ptr = (ptr + 1) % TOWER_NUM
                return child

            return make_child(parent1, parent2), make_child(parent2, parent1)
        else:
            return parent1.copy(), parent2.copy()


    def tournament(population, tournament_size, fitness_fun):
        """
        锦标赛选择核心函数（单次锦标赛）
        参数:
            population: 当前种群 [个体1, 个体2,...]
                        个体格式 = [客户ID1, 客户ID2,...]（同初始化函数）
            tournament_size: 锦标赛规模k（典型值3-5）
        返回:
            best_individual: 优胜个体（适应度最优的候选个体）
        行业标准实现验证：
            ✓ 符合标准锦标赛流程（随机选k比1）
            ✓ 使用min()配合calculate_fitness实现最小化问题优化
            ✓ 无放回抽样避免重复选择（random.sample特性）
        """
        # todo: 没有调整单词锦标赛规模：
        #  可根据迭代次数调整TOURNAMENT_SIZE（早期大规模保持多样性，后期小规模加快收敛）
        # TOURNAMENT_SIZE = max(2, 5 - int(generation / GENERATIONS * 3))

        candidates = random.sample(population, tournament_size)  # 关键行：无放回随机采样k个候选
        best_individual = min(candidates, key=fitness_fun)  # 关键行：选择适应度最优个体
        # （假设calculate_fitness返回路径总距离，越小越好）
        return best_individual

    def tournament_selection(population, fitness_fun):
        """
        种群级锦标赛选择
        参数:
            population: 当前完整种群（列表，含POP_SIZE个个体）
        返回:
            selected: 新种群（通过锦标赛选择出的POP_SIZE个个体）
        行业实现要点：
            ✓ 保持种群大小不变（循环POP_SIZE次）
            ✓ 允许个体被重复选择（符合GA选择阶段特性）
            ✓ 通常配合精英保留策略（此未实现，可改进）
        全局变量依赖:
            POP_SIZE: 种群规模（需与初始化保持一致）
            TOURNAMENT_SIZE: 锦标赛规模（建议3-5）
        """
        selected = []  # 新种群初始化
        for _ in range(POP_SIZE):  # 维持种群规模不变
            winner = tournament(population, TOURNAMENT_SIZE, fitness_fun)  # 关键行：执行单次锦标赛
            selected.append(winner)  # 关键行：将胜者加入新种群

        # todo 精英保留 在tournament_selection末尾添加（保持最优个体不丢失）
        # best_individual = min(population, key=calculate_fitness)
        # selected[0] = best_individual  # 替换新种群中第一个个体

        return selected

    # 计算适应度（总距离）
    # def calculate_fitness(individual):
    #     total_distance = 0
    #     vehicles = [{"battery": BATTERY_CAPACITY, "location": f"Depot_{i}"} for i in range(TOWER_NUM)]  # 每辆车初始在仓库
    #
    #     for customer in individual:  # 遍历解里面的每个客户
    #
    #         min_distance = float('inf')
    #         selected_vehicle = None
    #         for vehicle in vehicles:  # 看看附近有没有最近的可用车辆
    #             distance = DISTANCE_MATRIX[LOCATION_KEYS.index(vehicle["location"])][
    #                 LOCATION_KEYS.index(f"Customer_{customer}")]
    #             if vehicle["battery"] >= distance + demands[customer] and distance < min_distance:
    #                 min_distance = distance
    #                 selected_vehicle = vehicle
    #
    #         if selected_vehicle is None:  # 如果没有可用车辆，
    #             # 选择电量最多的车辆返回最近的仓库充电
    #             selected_vehicle = max(vehicles, key=lambda x: x["battery"])
    #             nearest_depot = min(range(TOWER_NUM),
    #                                 key=lambda x: DISTANCE_MATRIX[LOCATION_KEYS.index(selected_vehicle["location"])][
    #                                     LOCATION_KEYS.index(f"Depot_{x}")])
    #             total_distance += DISTANCE_MATRIX[LOCATION_KEYS.index(selected_vehicle["location"])][
    #                 LOCATION_KEYS.index(f"Depot_{nearest_depot}")]
    #             selected_vehicle["battery"] = BATTERY_CAPACITY
    #             selected_vehicle["location"] = f"Depot_{nearest_depot}"
    #             distance = DISTANCE_MATRIX[LOCATION_KEYS.index(selected_vehicle["location"])][
    #                 LOCATION_KEYS.index(f"Customer_{customer}")]
    #
    #         # 更新车辆状态
    #         total_distance += distance
    #         selected_vehicle["battery"] -= distance + demands[customer]
    #         selected_vehicle["location"] = f"Customer_{customer}"
    #
    #     # 所有车辆返回最近的仓库
    #     for vehicle in vehicles:
    #         nearest_depot = min(range(TOWER_NUM),
    #                             key=lambda x: DISTANCE_MATRIX[LOCATION_KEYS.index(vehicle["location"])][
    #                                 LOCATION_KEYS.index(f"Depot_{x}")])
    #         total_distance += DISTANCE_MATRIX[LOCATION_KEYS.index(vehicle["location"])][
    #             LOCATION_KEYS.index(f"Depot_{nearest_depot}")]
    #
    #     return total_distance

    def VNS_calculate_cost_path_share(solution, get_path=False):
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
        # dis_matrix=
        dis_dict = {}

        # 计算距离矩阵
        def get_distance(a, b):
            id = (a[0], a[1], b[0], b[1])
            id2 = (b[0], b[1], a[0], a[1])
            if id in dis_dict:
                return dis_dict[id]
            elif id2 in dis_dict:
                return dis_dict[id2]
            else:
                dis = np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
                dis_dict[id] = dis
            # 不行这个是计算两个具体坐标的函数。
            return dis

        def get_empty_depot(cur_pos, in_depot):
            # 返回空仓库序列？   输入是当前位置和当前无人机组状态。根据位置和仓库状态计算出{depot_id:distance}
            all_d = set([i for i in range(UAV_NUM)])
            # print(f"all_d={all_d}")
            # print(f"in_depot={in_depot}")
            unavailable_d = set([i for i in in_depot.values() if i is not False])
            # print(f"unavailable_d={unavailable_d}")

            empty_depot = all_d - unavailable_d
            # print(f"empty_depot={empty_depot}")
            empty_depot = {d: depots_pos[d] for d in empty_depot}
            empty_depot_dis = {k: get_distance(cur_pos, dpos) for k, dpos in empty_depot.items()}
            empty_depot_dis = sorted(empty_depot_dis.items(), key=lambda x: x[1])
            # print(f"empty_depot_dis={empty_depot_dis}")
            return empty_depot_dis

        def get_farthest_depot(cur_pos):
            # 返回最远的仓库的距离？？   输入是当前位置
            # empty_depot=set(i for i in range(uav_num)) - set(in_depot)
            # print(f"empty_depot={empty_depot}")
            # empty_depot={d:depots_pos[d] }
            empty_depot_dis = [get_distance(cur_pos, depots_pos[d]) for d in range(UAV_NUM)]
            farthest = max(empty_depot_dis)
            # sorted(empty_depot_dis,key=lambda x:x[1])
            # print(f"farthest={farthest}")
            return farthest

        total_distance = 0
        visited_plan = {cu: False for p in solution.values() for cu in p}

        if get_path:
            path = {depot_id: [depots_pos[depot_id]] for depot_id in range(UAV_NUM)}  # 初始的时候装填仓库位置

        decision_order = [(depot_id, 0) for depot_id in solution.keys()]
        current_tour_p = {k: 0 for k in solution.keys()}  # 记录每个无人机当前tour到哪个点了。 ???还是记录当前坐标呢
        # in_depot=[depot_id for depot_id in range(len(solution))]
        in_depot = {k: k for k in solution.keys()}  # 记录是否在仓库（初始为是），如果是，则元素是仓库id。否则是False
        remain_battery = {k: BATTERY_CAPACITY for k in solution.keys()}

        while True:  # 先不写停止标准
            # 排序
            decision_order.sort(key=lambda x: x[1])
            uav_id, now = decision_order.pop(0)
            if now == np.inf:  # 说明所有无人机都回到仓库了。
                # print("结束了捏")
                break  # 退出while

            route = solution[uav_id]
            battery = remain_battery[uav_id]

            if in_depot[uav_id] is not False:  # 如果当前在仓库。获取xy位置
                current_pos = depots_pos[in_depot[uav_id]]
            else:
                current_pos = tower_pos[route[current_tour_p[uav_id] - 1]]

            if current_tour_p[uav_id] == len(route):  # 如果没有下一个tower了
                # print(f"没有下一个点了, {uav_id}要回仓库了！！！")
                empty_depots = get_empty_depot(current_pos, in_depot)
                if len(empty_depots) != 0:

                    depot_id = empty_depots[0][0]  # 选最近的空depot id
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
                    remain_battery[uav_id] = battery
                else:
                    pass
                    # print("len(empty_depots)==0")
                    # raise ValueError("len(empty_depots)==0")
                decision_order.append((uav_id, np.inf))

            else:  # 如果还有下一个tower要去
                tower_id = route[current_tour_p[uav_id]]  # 下一个tower点。
                distance2c = get_distance(current_pos, tower_pos[tower_id])
                # 看看是否能去最远的仓库（不管是否空仓库）
                c2farthest = get_farthest_depot(tower_pos[tower_id])
                distance2c2depot = distance2c + c2farthest  # 下一个点回最远仓库

                if battery < upper_bound * distance2c2depot + demands[tower_id]:  # 如果电不够去下一个电塔后再充电桩 # todo 已波动修改
                    if in_depot[uav_id] is not False:
                        # print("明明在充电桩但是却不能出发……")
                        total_distance = np.inf
                        break

                    # 返回仓库充电
                    empty_depots = get_empty_depot(current_pos, in_depot)
                    depot_id = empty_depots[0][0]  # 选最近的depot id
                    return_distance = empty_depots[0][1]
                    battery -= return_distance * upper_bound  # todo 已波动修改
                    if battery < 0:
                        raise ValueError(f"返回仓库的时候 battery {battery}<0 ")
                    total_distance += return_distance
                    battery = BATTERY_CAPACITY  # 恢复满电
                    if get_path:
                        path[uav_id].append(depots_pos[depot_id])
                    # 更新状态
                    # current_tour_p[uav_id] += 1 # 这个不变
                    in_depot[uav_id] = depot_id
                    remain_battery[uav_id] = battery
                    decision_order.append((uav_id, now + return_distance * upper_bound + DEPOT_NEED))

                else:  # 是电量足够直接去下一个电塔。
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
                    decision_order.append((uav_id, now + distance2c * upper_bound + demands[tower_id]))

        if not all(visited_plan.values()):
            # print("visited_plan.values=", visited_plan.values())
            total_distance = np.inf

        if get_path:
            return total_distance, path
        else:
            return total_distance

    def calculate_fitness_2(individual, get_path=False):

        total_distance = 0
        vehicles_pos = [f"Depot_{i}" for i in range(UAV_NUM)]
        # 先分配客户，然后调用VNS的fitness计算充电桩
        visit_designate = {i: [] for i in range(UAV_NUM)}  # 需要是{uav_id: list[tower id] }

        for customer in individual:  # 遍历解里面的每个客户
            # customer 是客户编号下标。
            min_distance = float('inf')
            selected_uav_id = None

            for uav_id in range(UAV_NUM):  # 看看附近有没有最近的可用车辆
                distance = DISTANCE_MATRIX[LOCATION_KEYS.index(vehicles_pos[uav_id])][
                    LOCATION_KEYS.index(f"Customer_{customer}")]
                if distance < min_distance:
                    min_distance = distance
                    selected_uav_id = uav_id

            vehicles_pos[selected_uav_id] = f"Customer_{customer}"  # 更新位置
            visit_designate[selected_uav_id].append(customer)  # 放入序列

        total_distance = VNS_calculate_cost_path_share(visit_designate, get_path)
        # print(f"VNS 计算的{total_distance}")

        return total_distance

    # 可视化收敛性
    def plot_convergence(fitness_history):
        plt.plot(range(1, GENERATIONS + 1), fitness_history, marker='o')
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness (Total Distance)")
        plt.title("Convergence of Genetic Algorithm")
        plt.grid()
        # plt.show()
        plt.savefig("GA_convergence.jpg")

    # 初始化种群：生成POP_SIZE个随机个体（每个个体代表一个可能的路径解）
    population = initialize_population()
    # 用于记录每一代的最佳适应度（路径总距离），用于后续收敛分析
    fitness_history = []
    fitness_fun=calculate_fitness_2
    print(f"使用fitness_fun= 2")
    best_fitness=np.inf
    # 在全局参数区域定义早停阈值（建议放在代码开头）

    # 在主循环前初始化跟踪变量
    global_best = float('inf')
    global_best_individual=None
    no_improve_counter = 0  # new

    from TW_Baseline.VNS距离矩阵版 import progress_bar
    # 主循环：迭代GENERATIONS代
    for generation in range(GENERATIONS):

        progress_bar(generation + 1, GENERATIONS)
        print(f" global best cost={round(global_best, 5)}", end="")

        # 选择阶段：从当前种群中选择较优的个体作为父母
        # population = selection(population)  # 注释掉的轮盘赌选择 todo
        population = tournament_selection(population,fitness_fun)  # 使用锦标赛选择

        # 准备新一代种群
        new_population = []

        # 每次处理两个父母，步长为2（因为每次交叉产生两个孩子）
        for i in range(0, POP_SIZE, 2):
            # 选择两个相邻的个体作为父母
            parent1, parent2 = population[i], population[i + 1]

            # 交叉阶段：通过顺序交叉(OX)产生两个后代
            child1, child2 = crossover(parent1, parent2)

            # 变异阶段：对每个后代进行交换变异并加入新种群
            new_population.append(mutation(child1))
            new_population.append(mutation(child2))

        # 更新种群为新生成的种群
        population = new_population

        best_individual = min(population, key=fitness_fun)
        best_fitness = fitness_fun(best_individual)
        # 记录并输出当前代的最佳适应度
        fitness_history.append(best_fitness)

        # ===== 新增早停检测逻辑 ===== # new
        if best_fitness < global_best:
            global_best = best_fitness
            global_best_individual=best_individual
            no_improve_counter = 0
        else:
            no_improve_counter += 1

        if no_improve_counter >= STAGNATION_LIMIT:
            print(f"\nEarly stopping at generation {generation}")
            break
        # ========================== # new

    # 算法结束后输出最终结果
    # best_individual = min(population, key=fitness_fun)
    # best_fitness, feasible_best_solution = fitness_fun(best_individual, get_path=True)
    # todo 0530改成使用整个历史最佳的个体
    if global_best_individual is not None:
        global_best_fitness, feasible_best_solution = fitness_fun(global_best_individual, get_path=True)
    else:
        print("global_best_individual is None")
        global_best_fitness, feasible_best_solution=float('inf'),None


    print("Best Fitness (Total Distance):", global_best_fitness)  # 打印最佳路径总距离
    # print("Best feasible_best_solution:", feasible_best_solution)  # 打印最佳路径解
    # 可视化部分
    # plot_convergence(fitness_history)  # 绘制适应度收敛曲线
    # plot_routes(best_individual)  # 绘制最佳路径路线图
    fitness_history= fitness_history+ [fitness_history[-1]]*(GENERATIONS-len(fitness_history)) #补全长度
    return global_best_fitness, feasible_best_solution, fitness_history

def run_GA_vrp(GA_args, position_set_, uav_num,tower_num,upper, lower,  share_, initialG=False):
    """
    适用于一次性运行n个GA算法VRP问题实例。是用于把RL的数据集传进来训练的接口。
    position_set: 为多个地图的集合。
    """

    print("Run GA set:")
    print(f"max_iteration={GA_args['GENERATIONS']}")
    print(f"uav num={uav_num}, tower_num={tower_num}")
    print(f"upper bound={upper}, lower bound={lower}")
    print(f"SHARE={share_}")

    cost_set = []

    if type(position_set_) is not np.ndarray:
        position_set_ = position_set_.numpy()
    position_set_ = position_set_.transpose(0, 2, 1)
    data_size = position_set_.shape[0]
    print(f"数据集大小={data_size}")
    mean_converge_history = np.zeros(GA_args['GENERATIONS'])

    import time
    # 开始计时
    start_time = time.time()
    for t in range(data_size):
        position = position_set_[t]  # 使用外界传进来的坐标
        print(f"地图样本{t}:", end="")
        cost, solution, converge_history = genetic_algorithm(GA_args, uav_num, tower_num, position,
                                                           upper, lower, share_=share_,
                                                           draw_path=False if data_size > 1 else True)

        cost_set.append(cost)
        mean_converge_history += np.array(converge_history)

    end_time = time.time()
    total_time = round(end_time - start_time, 4)
    avg_time = round(total_time / data_size, 2)
    print(f"GA T{tower_num} uav{uav_num} 总运行时间{total_time}s。平均一图时间：{avg_time}s")

    if data_size > 1:
        mean_converge_history /= data_size
        mean_cost = sum(cost_set) / data_size
        print(f"cost_set={cost_set}")
        from TW_Baseline.VNS距离矩阵版 import multi_converge_visualize
        multi_converge_visualize(data_size, uav_num, tower_num, 0,share_,
                                 mean_converge_history, mean_cost, save=False)

    # average_reward = sum(reward_set) / run_time
    return cost_set, avg_time


# if __name__ == "__main__":
#     # 参数设置
#
#     run_times=1
#     tower_n=200
#     uav_n=10
#     share = True
#     upper, lower= 1.1,0.9
#     opt_function=[2,3]
#     np.random.seed(123) # todo 随机种子！！方便debug。
#     position_set = np.random.random(size=(run_times, 2, tower_n + uav_n))
#
#     GA_args= {
#         "POP_SIZE" : 50,  # 种群大小
#         "GENERATIONS" :  100,  # 迭代次数
#         "CROSSOVER_RATE" : 0.7,  # 交叉概率
#         "MUTATION_RATE": 0.05,  # 变异概率  （典型值0.01~0.1）
#         "TOURNAMENT_SIZE" :  3}
#
#     # 一次性运行多个地图实例。
#     cost_set,avg_time = run_GA_vrp(GA_args, position_set, uav_n,tower_n, upper, lower,share_= share)
#     print(f"Run {run_times} times. GA Average tour length: {np.mean(cost_set)}")


