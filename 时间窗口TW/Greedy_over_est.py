
import math
import os.path
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def DRL4VRP_Problem(tower_num, uav_num, position,est_upper, est_lower, _share):
    """
    """
    map_size = 1  # 这是边长。100*100.能量需要2*√2 *100=280
    max_energy = 2 *  map_size * 1.4  # 为了保证一定能够飞到最远端。 # todo 对齐1
    tower_need=0.1 # todo 暂定是0.1。对齐1

    # 创建地图.前uav_num个是给无人机仓库用的
    tower_position = position[uav_num:] # 电塔坐标

    visited_tower = [False] * tower_num  # 标记是否访问过。todo  以后如果demand不同，可以用visited来存放demand的值。
    empty_depot=[False] * uav_num # 充电桩是否为空。初始为满。离开和访问的时候更新
    tower_demand = np.full(tower_num,tower_need)

    # 新增：N2N dis：包括电塔和仓库在内的所有点之间的两点距离
    pos_self = np.expand_dims(position,1).repeat(tower_num+uav_num,axis=1) # 扩展一下维度，用于距离矩阵计算
    n2n_distance=np.sqrt(np.sum(np.square(pos_self-position),axis=2))

    # 2.7新增：动态环境估计的上下界
    est_upper=est_upper
    est_lower=est_lower


    def cal_distance_list(l1_to_l2):
        """
        :param l1_to_l2: N*1 np array
        :return: N*1 np array的距离列表
        """
        dis = np.square(l1_to_l2)
        dis = np.sqrt(np.sum(dis, axis=1))  # 得到距离
        return dis


    def L2(p1, p2):
        p1 = np.array(p1)
        p2 = np.array(p2)
        return np.linalg.norm(p1 - p2)

    def get_random_factor(upper,lower):

        # return lower + np.random.rand() * (upper - lower) #  20+200 share:28.216352640810392
        return upper # 20+200 no share:29.416347104914276 【Over estimate 实验要启用这一行】
    '''
    首先要怎么比较：
      1，使用srp的代码的版本是要用训练好的网络，给定一定数量（1000）的地图，给出平均路程结果
      2，保持地图随机种子不变，使用相同的地图给贪心来实现
    
    贪心思路：【多无人机且回到自己的仓库+轮流决策】
        最外层循环：
        对于这一两无人机：
            如果目前不在仓库：
                对于周围所有其他非仓库的点：计算本仓库~那个仓库+那个仓库回到自己的仓库的距离
                如果当前能量<那个距离：则选择回到自己的仓库。补充能量
                如果能量足够：过去那个仓库。并且标记已访问。消耗能量
            如果目前在自己的仓库：
                对于周围所有其他非仓库的点：计算最近的点，过去，标记已访问。消耗能量
        如果所有点都访问完毕并且无人机回到自己的仓库：停止
        还没结束：下一辆无人机
    
    '''


    def plot_track(points_set,total_dis,save_dir):
        '''
        用于画出轨迹的动图 todo 现在只能画一张图……【要限定固定的画图张数量吗
        '''
        plt.close('all')
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_ylim(0, map_size)
        ax.set_xlim(0, map_size)

        plt.tight_layout()
        # ax.grid()
        # plt.scatter(x[0], y[0], marker="*")  # 仓库

        # ----------------------
        x = [np.array([p[0] for p in track]) for track in points_set]
        y = [np.array([p[1] for p in track]) for track in points_set]

        tower_x=[x1[1:] for x1 in x]
        tower_y=[y1[1:] for y1 in y]
        for xt,yt in zip(tower_x,tower_y):
            plt.scatter(xt,yt,s=10) # 画出城市的点 # s是大小
        # 用于画出仓库点。拼接在一起
        plt.scatter([x1[0] for x1 in x], [y1[0] for y1 in y], marker="*",s=90)  # 仓库

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
        title=f"Greedy T{tower_num} UAV{uav_num} reward{round(total_dis,3)} share={str(_share)}"
        plt.title(title)
        ani.save(os.path.join(save_dir,f"{title}.gif"), writer='pillow',dpi=500)  # 保存
        # plt.show()
        plt.close('all')
        print(f"{title} 绘图完成")



    class UAV:
        def __init__(self, depot_pos,init_id):
            """
            初始化仓库位置、路径和当前飞机能量。
            :param depot_pos:给定此无人机的仓库位置 【共享仓库是不能用这个变量的】
            """
            #self.depot_pos = depot_pos
            self.depot_id=init_id
            self.current_node_id=init_id # 要多加一个这个。
            self.track = [depot_pos]  # 路程轨迹坐标……不是id。初始化仓库位置。当前所在位置=取[-1]
            self.energy = max_energy # 初始化

        def is_in_depot(self):
            # 【非共享版本】
            return self.depot_id==self.current_node_id# 变成np以后二维的比较就会逐元素比较
        
        def is_in_depot_share(self):
            """
            # 【共享仓库版本】
            """
            return self.current_node_id < uav_num # 当前位置是否是仓库

        def print_track(self):
            print(self.track)

        def get_total_dis(self):
            """
            :return: 总路程长度。
            """
            # 开头结尾都是仓库。1231
            start = np.array(self.track[:-1])
            end = np.array(self.track[1:])
            dis = cal_distance_list(start - end)
            return np.sum(dis)

        # def get_cost_min_tower_old(self):
        #     """
        #     【非共享仓库版本】
        #     由于需要考虑电塔的demand，nearest定义改为距离+demand最小， 然后再加上由函数外界重新判断是否电量足够，不够就回仓库……
        #     :return: 返回一个【没访问过的】、且过去tower+回仓库最总距离最短的：tower 编号，去tower距离、tower回depot距离
        #     """
        #
        #     mask=-np.log10(np.array(1)-visited_tower)[:, np.newaxis] # 通过visited 01计算mask
        #     tower_position_mask = mask + tower_position # mask加到postion上 todo ……mask 不应该加到距离上吗……怎么加到positon上了
        #
        #     current_pos = np.array(self.track[-1])
        #     # dis_list_go = cal_distance_list(current_pos - np.array(tower_position_mask))# 当前点-所有tower
        #     dis_list_go_test= n2n_distance[self.current_node_id, uav_num:] + mask.squeeze()
        #     # if not all(dis_list_go_test==dis_list_go):
        #     #     raise  ValueError("dis_list_go_test!=dis_list_go")
        #
        #     # dis_list_back = cal_distance_list(position[self.depot_id] - np.array(tower_position_mask))
        #     dis_list_back_test= n2n_distance[self.depot_id,uav_num:] + mask.squeeze()
        #     # if not all(dis_list_back==dis_list_back_test):
        #     #     raise  ValueError("dis_list_back==dis_list_back_test")
        #     dis_total = dis_list_go_test + dis_list_back_test + tower_demand # fix:考虑电塔demand
        #     nearest_index = np.argmin(dis_total)
        #     return nearest_index, dis_list_go_test[nearest_index], dis_list_back_test[nearest_index],tower_demand[nearest_index]

        def get_cost_min_tower(self):
            """
            【非共享仓库版本】20250206新版
            nearest：考虑距离最小的“可行电塔”， 然后再加上由函数外界重新判断是否电量足够，不够就回仓库。
            （可行电塔：指  过去电塔距离+电塔demand+回自己的充电桩 ＞ 当前电量）
            :return: 返回一个【没访问过的】、且过去tower距离最短的：tower 编号，去tower距离、tower回depot距离
            """

            mask = -np.log10(np.array(1) - visited_tower)[:, np.newaxis]  # 通过visited 01计算mask

            dis_list_go = n2n_distance[self.current_node_id, uav_num:] + mask.squeeze()
            dis_list_back = n2n_distance[self.depot_id, uav_num:] + mask.squeeze()

            #dis_total = dis_list_go + dis_list_back + tower_demand  # fix:考虑电塔demand

            # 20250206修改:不考虑电塔demand
            nearest_tower_index = np.argmin(dis_list_go) # todo 注意这里是电塔的index ！0是第一个电塔！！不包括充电桩
            return nearest_tower_index, dis_list_go[nearest_tower_index], dis_list_back[nearest_tower_index], tower_demand[
                nearest_tower_index]

        def get_nearest_depot_id(self):
            '''
            共享版本，用于电量不足时，找到最近的【空的】充电桩id。 【非共享版本直接回自己绑定的即可】
            返回值：仓库id，当前点到仓库的距离。
            '''
            t2depot= n2n_distance[self.current_node_id, :uav_num]
            depot_mask=-np.log10(empty_depot) # 01 转换成inf 和0
            t2depot+=depot_mask # 加上mask
            depot_id=np.argmin(t2depot)
            if not empty_depot[depot_id]: # 如果全部仓库都有无人机就报错
                raise ValueError("No depot available!!")
            return depot_id,t2depot[depot_id]


        def get_cost_min_tower_share(self):
            """
            【共享仓库版本】
            由于需要考虑电塔的demand，nearest定义改为距离+demand最小， 然后再加上由函数外界重新判断是否电量足够，不够就回仓库……
            :return: 返回一个【没访问过的】、且过去tower距离最短的：tower 编号，去tower距离、tower回depot距离
            """

            def get_farthest_depot_dis():
                '''
                 # return: D所有tower~其最远仓库的距离. 维度是(uav_num,)
                '''
                # n2n_distance 是所有点到另一个点的距离
                tower2depot_dis = n2n_distance[uav_num:, :uav_num]  # 所有tower 到所有仓库的距离
                tower2depot_max_dis = np.max(tower2depot_dis, axis=1)
                return tower2depot_max_dis

            mask= -np.log10(np.array(1)-visited_tower)[:, np.newaxis] # 通过visited 0和1计算mask,变成inf 和 0

            dis_list_go_test= n2n_distance[self.current_node_id, uav_num:] + mask.squeeze()
            dis_list_back = get_farthest_depot_dis() # D 所有tower~其最远仓库的距离（这里不考虑仓库是否为空！RL也不考虑。能去最远的肯定能去其他的。）
            # dis_total = dis_list_go_test + dis_list_back + tower_demand # D当前点~所有tower + D 所有tower~其最远仓库的距离 +demand
            nearest_index = np.argmin(dis_list_go_test) # fixme注意这是tower id 不是node id  20250206 修改为只考虑去电塔距离最近

            return nearest_index, dis_list_go_test[nearest_index], dis_list_back[nearest_index],tower_demand[nearest_index]


    def run():
        uav_set = [UAV(depot_pos=position[i],init_id=i) for i in range(uav_num)]

        # uav_index = 0
        uav_order=[(i,0) for i in range(uav_num)] # (uav_id, task_time)
        step=0
        # 开始循环每个无人机
        while not all(visited_tower) or any(empty_depot):  # 如果还有电塔没有访问、是否所有无人机都在仓库
            # uav_index %= uav_num
            uav_index=uav_order[0][0]
            uav = uav_set[uav_index]
            # 计算最近的点的下标和距离
            if _share:
                near_tower_index, dis_go, dis_to_depot, demand = uav.get_cost_min_tower_share() # fix:考虑电塔demand
                in_depot = uav.is_in_depot_share()
            else:
                near_tower_index, dis_go, dis_to_depot, demand = uav.get_cost_min_tower()
                in_depot = uav.is_in_depot()

            if in_depot: # fix√ # 如果在仓库。（这个是留给以后扩展成可以连续访问仓库用的）
                if est_upper*(dis_go + dis_to_depot + demand) > uav.energy:  # 电量不够：自己的任务已经完成了
                    #if dis_go != math.inf:
                        # print(f"Note: 最近的tower是{dis_go} 回仓库是{dis_to_depot}，"
                        #       f"需求加来回路程>满格能量：{uav.energy}。uav留在原地")
                    #else:
                        # Note：距离是inf，说明所有城市都访问过了。需要等其他无人机完成任务飞回来。
                        # print("似乎所有任务都完成了。只需要等着")
                        # pass

                    # print(f"Note:{uav.energy}。uav留在原地")
                    uav_order.pop(0)
                    uav_order.append((uav_index, math.inf))
                        # todo---------------------合并这两个情况。。。。。。。。。。。。。。。。
                else:  # 如果能量足够：过去那个tower。并且标记已访问。消耗能量
                    empty_depot[uav.current_node_id]=True #
                    uav.track.append(tower_position[near_tower_index]) #这是tower里面的id！！
                    tower_demand[near_tower_index] = 0  # 更新demand
                    visited_tower[near_tower_index] = True
                    uav.current_node_id= near_tower_index + uav_num # 切换当前位置

                    rand_factor = get_random_factor(est_upper,est_lower)

                    cost=(dis_go)*rand_factor +demand # fix:考虑电塔demand。
                    uav.energy -=  cost
                    uav_order.append((uav_index,uav_order.pop(0)[1]+cost)) # 添加任务时间排序

            else:  # 不在仓库
                if  est_upper*(dis_go + dis_to_depot + demand) > uav.energy:  # 电量不够 回到自己的仓库
                    if _share: ## 共享：找到最近的【空】仓库
                        depot_id, back_depot_energy = uav.get_nearest_depot_id()
                    else: # 非共享：直接返回自己的仓库
                        depot_id=uav.depot_id
                        back_depot_energy = n2n_distance[depot_id,uav.current_node_id]

                    if not empty_depot[depot_id]: #检查是否为空
                        raise ValueError(f"试图访问非空的充电桩{uav.current_node_id }")
                    rand_factor = get_random_factor(est_upper,est_lower)

                    cost = back_depot_energy*rand_factor
                    uav.energy -= cost
                    if uav.energy<0:
                        raise ValueError("还没回到充电桩就没电了，，")
                    uav.track.append(position[depot_id]) # 被选择的充电桩坐标
                    uav.current_node_id = depot_id  # 更新为成被选择的充电桩
                    empty_depot[depot_id] = False
                    uav.energy = max_energy
                    # rand_factor = get_random_factor(est_upper,est_lower)
                    uav_order.append((uav_index, uav_order.pop(0)[1] + cost+0.1))  # 添加任务时间排序
                else:  # 如果能量足够：过去下一个tower。并且标记已访问。消耗能量
                    uav.track.append(tower_position[near_tower_index])
                    uav.current_node_id=near_tower_index+uav_num # 转换成node id
                    tower_demand[near_tower_index]=0 # 更新demand
                    visited_tower[near_tower_index] = True
                    rand_factor = get_random_factor(est_upper,est_lower)
                    cost = (dis_go ) * rand_factor + demand # fix: 减去电塔demand
                    uav.energy -= cost
                    uav_order.append((uav_index, uav_order.pop(0)[1] + cost))  # 添加任务时间排序
            if uav.energy<0:
                raise ValueError("uav energy < 0")
            # uav_index += 1
            uav_order.sort(key=lambda x: x[1])
            step+=1
            if step> tower_num*2:
                print("greedy达到最大迭代次数，退出。")
                break

        if (tower_demand>0).any():
            raise ValueError("The uav haven't visited all the tower!!!")

        # 计算总路程部分。
        total_dis = 0
        all_track = [] # 不是兄弟你记录这个干什么
        for u in range(uav_num):
            uav = uav_set[u]
            if uav.current_node_id>= uav_num:
                raise ValueError(f"track uav{u} final position:{uav.current_node_id} is not depot")
            all_track.append(uav.track) #
            dis = uav.get_total_dis()
            total_dis += dis # 记录总距离。


        # plot_track(all_track,total_dis,"Greedy_GIF") # 画出本地图的动态图。

        return total_dis

    return run() # 返回本次



def run_greedy_VRP(position_set,tower_n,uav_n,upper,lower,share):
    '''
    适用于一次性运行n个贪心算法VRP问题实例。是用于把RL的数据集传进来训练的接口。
    position_set: 为多个地图的集合。
    '''
    # print("Run greedy:")
    # print(f"uav number:{uav_n}")

    reward_set = []
    if type(position_set) is not np.ndarray:
        position_set=position_set.numpy()
    position_set = position_set.transpose(0, 2, 1)

    run_time=position_set.shape[0]
    for t in range(run_time):
        position = position_set[t]  # 使用外界传进来的坐标
        reward = DRL4VRP_Problem(tower_n, uav_n, position,upper,lower,share)
        reward_set.append(reward)

    average_reward = sum(reward_set) / run_time
    # print(f"Run {run_times} times. Average tour length:  {average_reward}")
    return reward_set

def draw_path_change(share):
    '''
    飞机数量1-10，电塔是飞机数量20倍。
    '''
    run_times=1000
    y_avg_reward = []
    for uav_num in range(1, 11):
        tower_num = 20 * uav_num # 电塔数量是飞机数量20倍。
        position_set = np.random.random(size=(run_times, 2, tower_num + uav_num))
        reward_set = run_greedy_VRP(position_set, tower_num, uav_num, share)
        # print(f"Run {run_times} times. Average tour length:  {np.mean(reward_set)}")
        y_avg_reward.append(np.mean(reward_set))
    plt.plot(list(range(1,10)),y_avg_reward)
    plt.show()

def draw_uav_change(upper,lower,share):
    '''
    电塔数量固定，看飞机数量的影响
    '''
    run_times=500
    tower_n = 300
    y_avg_reward = []
    uav_n_lb=3
    uav_n_ub=10
    for uav_num in range(uav_n_lb, uav_n_ub):
        position_set1 = np.random.random(size=(run_times, 2, tower_n + uav_num))
        reward_set = run_greedy_VRP(position_set1, tower_n, uav_num, upper,lower,share)
        print(f"Run {run_times} times. Average tour length:  {np.mean(reward_set)}")
        y_avg_reward.append(np.mean(reward_set))
    plt.plot(list(range(uav_n_lb,uav_n_ub)),y_avg_reward)
    plt.savefig(f"R(uav_num) of {tower_n} tower")
    plt.show()


if __name__ == "__main__":
    run_times=1000
    tower_n=50
    uav_n=5
    # share_depot=False
    np.random.seed(111) #方便debug。
    position_set = np.random.random(size=(run_times, 2, tower_n + uav_n)) # 共用地图。
    share=True
    upper, lower=1.1, 0.9

    if share:
        # # # 运行共享
        reward_set_share=run_greedy_VRP(position_set,tower_n,uav_n,upper,lower,share=True) # 共享的效果也太差了……因为一直在局部搜索，去其他人仓库的约束其实还挺差的
        print(f"Run {run_times} times. 贪心 Share Average tour length:  {np.mean(reward_set_share)}") #
        # 5+50:  13.378931645824348
    else:
        # # 运行非共享
        reward_set_independent = run_greedy_VRP(position_set, tower_n, uav_n,upper,lower, share=False) # 非共享版本
        print(f"Run {run_times} times. 贪心非Share Average tour length:  {np.mean(reward_set_independent)}")
        # 5+50: 13.655793214349499

    # draw_path_change()
    # draw_uav_change(share=False)
    pass
