# # todo :可能demand不相等才是能拉开贪心和RL的设定吗！！
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
#
#
#
# def DRL4VRP_Problem(tower_num, uav_num, position):
#     map_size = 1  # 这是边长。100*100.能量需要2*√2 *100=280
#     max_energy = 2 *  map_size * 1.4  # 为了保证一定能够飞到最远端。 # todo 对齐1
#     tower_need=0.1 # todo 暂定是0.1。对齐1
#
#     # 创建地图.前uav_num个是给无人机仓库用的
#     # position = [[random.randint(0, map_size), random.randint(0, map_size)] for _ in range(point_num)]
#     tower_position = position[uav_num:] # 电塔坐标
#
#     visited_tower = [False] * tower_num  # 标记是否访问过。todo  以后如果demand不同，可以用visited来存放demand的值。
#     # tower_position_mask = np.array(tower_position, dtype=float) # 直接使用visited_tower进行计算 。此全局变量废除。
#     tower_demand=np.full(tower_num,tower_need)
#
#     def cal_distance_list(l1_to_l2):
#         """
#         :param l1_to_l2: N*1 np array
#         :return: N*1 np array的距离列表
#         """
#         dis = np.square(l1_to_l2)
#         dis = np.sqrt(np.sum(dis, axis=1))  # 得到距离
#         return dis
#
#
#     def L2(p1, p2):
#         p1 = np.array(p1)
#         p2 = np.array(p2)
#         return np.linalg.norm(p1 - p2)
#
#
#     '''
#     todo 首先要怎么比较：
#       1，使用srp的代码的版本是要用训练好的网络，给定一定数量（1000）的地图，给出平均路程结果
#       2，保持地图随机种子不变，使用相同的地图给贪心来实现
#       其他问题：
#
#     todo：
#         实现多无人机的时候，改成回到自己的仓库【看来要从这里开始实现啊啊……】
#
#     fixme 贪心思路：【多无人机且回到自己的仓库+轮流决策】
#         最外层循环：
#         对于这一两无人机：
#             如果目前不在仓库：
#                 对于周围所有其他非仓库的点：计算本仓库~那个仓库+那个仓库回到自己的仓库的距离
#                 如果当前能量<那个距离：则选择回到自己的仓库。补充能量
#                 如果能量足够：过去那个仓库。并且标记已访问。消耗能量
#             如果目前在自己的仓库：
#                 对于周围所有其他非仓库的点：计算最近的点，过去，标记已访问。消耗能量
#         如果所有点都访问完毕并且无人机回到自己的仓库：停止
#         还没结束：下一辆无人机
#
#     '''
#
#
#     def plot_track(points_set):
#         fig, ax = plt.subplots(figsize=(6, 6))
#         ax.set_ylim(0, map_size)
#         ax.set_xlim(0, map_size)
#
#         plt.tight_layout()
#         # ax.grid()
#         # plt.scatter(x[0], y[0], marker="*")  # 仓库
#
#         # ----------------------
#         x = [np.array([p[0] for p in track]) for track in points_set]
#         y = [np.array([p[1] for p in track]) for track in points_set]
#
#         tower_x=[x1[1:] for x1 in x]
#         tower_y=[y1[1:] for y1 in y]
#         for xt,yt in zip(tower_x,tower_y):
#             plt.scatter(xt,yt,s=10) # 画出城市的点 # s是大小
#         # 用于画出仓库点。拼接在一起
#         plt.scatter([x1[0] for x1 in x], [y1[0] for y1 in y], marker="*",s=90)  # 仓库
#
#         lines = [ax.plot([], [], lw=1)[0] for _ in range(len(points_set))]
#         plt.tight_layout()
#         # ----------------------
#
#         def init():
#             for line in lines:
#                 line.set_data([], [])
#             return lines
#
#         def animate(N):
#             for line, x1, y1 in zip(lines, x, y):
#                 line.set_data(x1[:N], y1[:N])
#             return lines
#
#         ani = animation.FuncAnimation(fig, animate, 50, init_func=init, interval=200)
#         plt.show()
#
#
#     class UAV:
#         def __init__(self, depot_pos):
#             """
#             初始化仓库位置、路径和当前飞机能量。
#             :param depot_pos:给定此无人机的仓库位置
#             """
#             self.depot_pos = depot_pos
#             self.track = [depot_pos]  # 路程轨迹。初始化仓库位置。当前所在位置=取下标-1
#             self.energy = max_energy # 初始化
#
#         def is_in_depot(self):
#             # print(f"{self.depot_pos}\n{self.track[-1]}")
#             # return self.depot_pos == self.track[-1]
#
#             return all(self.depot_pos == self.track[-1])# 变成np以后二维的比较就会逐元素比较
#
#         def print_track(self):
#             print(self.track)
#
#         def get_total_dis(self):
#             """
#             :return: 总路程长度。
#             """
#             # 开头结尾都是仓库。1231
#             start = np.array(self.track[:-1])
#             end = np.array(self.track[1:])
#             dis = cal_distance_list(start - end)
#             return np.sum(dis)  # todo 保留几位小数比较好???
#
#         def get_cost_min_tower(self):
#             """
#             由于需要考虑电塔的demand，nearest定义改为距离+demand最小， 然后再加上由函数外界重新判断是否电量足够，不够就回仓库……
#             :return: 返回一个【没访问过的】、且过去tower+回仓库最总距离最短的：tower 编号，去tower距离、tower回depot距离
#             """
#
#             mask=-np.log10(np.array(1)-visited_tower)[:, np.newaxis] # 通过visited 01计算mask
#             tower_position_mask = mask + tower_position # mask加到postion上
#
#             current_pos = np.array(self.track[-1])
#             dis_list_go = cal_distance_list(current_pos - np.array(tower_position_mask))
#             dis_list_back = cal_distance_list(self.depot_pos - np.array(tower_position_mask))
#             dis_total = dis_list_go + dis_list_back + tower_demand # fix:考虑电塔demand
#             nearest_index = np.argmin(dis_total)
#             return nearest_index, dis_list_go[nearest_index], dis_list_back[nearest_index],tower_demand[nearest_index]
#
#     # todo 当前贪心版本：
#     #   多无人机：是
#     #   是否只能回到自己仓库：是
#     #   可否连续访问仓库：是！！可以停留在仓库，如果最远的有需求tower已经去不了的话。
#
#     def run():
#         uav_set = [UAV(position[i]) for i in range(uav_num)]
#
#         uav_index = 0
#         # 开始循环每个无人机
#         while not all(visited_tower):  # 如果还有电塔没有访问
#             uav_index %= uav_num
#             uav = uav_set[uav_index]
#             # 计算最近的点的下标和距离
#             near_tower_index, dis_go, dis_to_depot,demand = uav.get_cost_min_tower() # fix:考虑电塔demand
#             if uav.is_in_depot():  # 在仓库。（这个是留给以后扩展成可以连续访问仓库用的）
#                 # if dis_go + dis_to_depot  > uav.energy:  # 电量不够
#                 if dis_go + dis_to_depot + demand > uav.energy:  # 电量不够 # fix :考虑电塔demand
#                     print(f"Note: 最近的tower是{dis_go} 回仓库是{dis_to_depot}，"
#                           f"大于满格能量：{uav.energy}。uav留在原地")
#                 else:  # 如果能量足够：过去那个tower。并且标记已访问。消耗能量
#                     uav.track.append(tower_position[near_tower_index])
#                     # uav.energy -= dis_go
#                     uav.energy -=  dis_go+demand # fix:考虑电塔demand。
#                     tower_demand[near_tower_index] = 0 # 更新demand
#                     visited_tower[near_tower_index] = True
#                     # print(f"uav{uav_index} visited tower{near_tower_index}:{uav.track[-1]}.remaining energy:{uav.energy}")
#             else:  # 不在仓库
#                 # if dis_go + dis_to_depot > uav.energy:  # 电量不够
#                 if dis_go + dis_to_depot + demand > uav.energy:  # 电量不够 回到自己的仓库 # fix :考虑电塔demand
#                     back_depot_energy=L2(uav.depot_pos, uav.track[-1])
#                     if uav.energy<back_depot_energy:
#                         raise ValueError(f"uav {uav_index} has energy {uav.energy} "
#                                          f"but go back to depot need {back_depot_energy}")
#                     uav.energy -= back_depot_energy
#                     # print(f"uav{uav_index} go back to its depot {uav.depot_pos},remaining energy:{uav.energy}.")
#                     uav.track.append(uav.depot_pos)
#                     uav.energy = max_energy
#                     # print(f"uav{uav_index} charged to max energy:{uav.energy}.")
#                 else:  # 如果能量足够：过去那个tower。并且标记已访问。消耗能量
#                     uav.track.append(tower_position[near_tower_index])
#                     # uav.energy -= dis_go
#                     uav.energy -= dis_go+demand # fix: 减去电塔demand
#                     tower_demand[near_tower_index]=0 # 更新demand
#                     visited_tower[near_tower_index] = True
#                     # print(f"uav{uav_index} visited tower{near_tower_index}:{uav.track[-1]}.remaining energy:{uav.energy}")
#             if uav.energy<0:
#                 raise ValueError("uav energy < 0")
#             # print(f"tower demand: {tower_demand}")
#             uav_index += 1
#
#         if (tower_demand>0).any():
#             raise ValueError("The uav haven't visited all the tower!!!")
#
#         # 计算总路程部分。
#         total_dis = 0
#         all_track = []
#         for uav_index in range(uav_num):
#             uav = uav_set[uav_index]
#             uav.track.append(uav.depot_pos)  # 手动添加回仓库的点。
#             # print(f"track uav{uav_index}:")
#             # uav.print_track()
#             all_track.append(uav.track)
#             dis = uav.get_total_dis()
#             total_dis += dis # 记录总距离。
#             # print(f"distance:{dis}")
#
#         # print(f"total distance:{total_dis}")
#
#         # plot_track(all_track) # 画出本地图的动态图。
#
#         return total_dis
#
#     return run() # 返回本次
#
#
#
# def run_greedy_VRP(position_set,tower_n,uav_n):
#     '''
#     run_greedy_VRP 适用于一次性运行n个贪心算法VRP问题实例。
#     position_set: 为多个地图的集合。
#     '''
#     # print("Run greedy:")
#     # print(f"uav number:{uav_n}")
#
#     reward_set = []
#     if type(position_set) is not np.ndarray:
#         position_set=position_set.numpy()
#     position_set = position_set.transpose(0, 2, 1)
#
#     run_times=position_set.shape[0]
#     for t in range(run_times):
#         position = position_set[t]  # 使用外界传进来的坐标
#
#         reward = DRL4VRP_Problem(tower_n, uav_n, position)
#         reward_set.append(reward)
#
#     average_reward = sum(reward_set) / run_times
#     # print(f"Run {run_times} times. Average tour length:  {average_reward}")
#     return reward_set
#
# def draw_path_change():
#     '''
#     飞机数量1-10，电塔是飞机数量20倍。
#     '''
#     run_times=1000
#     uav_n=5
#     y_avg_reward = []
#     for uav_n in range(1, 11):
#         tower_n = 20 * uav_n # 电塔数量是飞机数量20倍。
#         position_set = np.random.random(size=(run_times, 2, tower_n + uav_n))
#         reward_set = run_greedy_VRP(position_set, tower_n, uav_n)
#         # print(f"Run {run_times} times. Average tour length:  {np.mean(reward_set)}")
#         y_avg_reward.append(np.mean(reward_set))
#     plt.plot(list(range(1,10)),y_avg_reward)
#     plt.show()
#
# def draw_uav_change():
#     '''
#     电塔数量固定，看飞机数量的影响
#     '''
#     run_times=500
#     tower_n = 300
#     y_avg_reward = []
#     uav_n_lb=3
#     uav_n_ub=10
#     for uav_n in range(uav_n_lb, uav_n_ub):
#         position_set = np.random.random(size=(run_times, 2, tower_n + uav_n))
#         reward_set = run_greedy_VRP(position_set, tower_n, uav_n)
#         print(f"Run {run_times} times. Average tour length:  {np.mean(reward_set)}")
#         y_avg_reward.append(np.mean(reward_set))
#     plt.plot(list(range(uav_n_lb,uav_n_ub)),y_avg_reward)
#     plt.savefig(f"R(uav_num) of {tower_n} tower")
#     plt.show()
#
#
#
# if __name__ == "__main__":
#     run_times=1000
#     tower_n=200
#     uav_n=5
#     np.random.seed(111)# todo 方便debug
#     position_set = np.random.random(size=(run_times, 2, tower_n + uav_n))
#     reward_set=run_greedy_VRP(position_set,tower_n,uav_n)
#     print(f"Run {run_times} times. Average tour length:  {np.mean(reward_set)}") # 100，200、5：28.24913963952968
#
#     # draw_path_change()
#     # draw_uav_change()
#     pass
#
