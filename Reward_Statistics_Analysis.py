import os.path

import  numpy as np
from jedi.inference.finder import filter_name

from Algrithm_compare import run_multi_alg_test

class Reward_Collect:
    def __init__(self):
        # self.reward_greedy=[]
        # self.reward_RL=[]

        self.test_multi_alg_reward={}


    def get_mean(self):
        # print(f"Greedy reward mean: {np.mean(self.reward_greedy)}")
        # print(f"RL reward mean: {np.mean(self.reward_RL)}")
        # --------------新版
        for k,v in self.test_multi_alg_reward.items():
            print(f"{k} reward mean:{np.mean(v)}")
        print("")


    def get_min_max(self):
        # print(f"Greedy reward range: [{min(self.reward_greedy)},{max(self.reward_greedy)}]")
        # print(f"RL reward range: [{min(self.reward_RL)},{max(self.reward_RL)}]")
        # --------------新版
        for k,v in self.test_multi_alg_reward.items():
            print(f"{k} reward range:[{min(v)},{max(v)}]")
        print("")


    def get_variance(self):
        # print(f"Greedy variance {np.var(self.reward_greedy,dtype=np.float64)}")
        # print(f"RL variance {np.var(self.reward_RL,dtype=np.float64)}")
        for k,v in self.test_multi_alg_reward.items():
            print(f"{k} variance{np.var(v,dtype=np.float64)}")
        print("")


    def independent_T_test(self):
        from scipy import stats
        # 独立2个样本t检验
        sample1 = np.asarray(self.reward_greedy)
        sample2 = np.asarray(self.reward_RL)
        r = stats.ttest_ind(sample1, sample2)
        print("独立样本t检验")
        # print("statistic:", r.__getattribute__("statistic"))
        print("pvalue:", r.__getattribute__("pvalue"))
        print("")


    def paired_T_test(self):
        '''
        成对样本t检验
        '''
        from scipy import stats
        sample1 = np.asarray(self.reward_greedy)
        sample2 = np.asarray(self.reward_RL)
        r = stats.ttest_rel(sample1, sample2)
        print("成对样本t检验")
        # print("statistic:", r.__getattribute__("statistic"))
        print("pvalue:", r.__getattribute__("pvalue"))
        print("")


    def wilcoxon_rank_sum_test(self):
        """
        适用于不成对。The Mann-Whitney U test is a non-parametric version of the t-test for independent samples.
        When the means of samples from the populations are normally distributed, consider scipy. stats. ttest_ind.
        """
        from scipy.stats import ranksums
        sample1 = self.test_multi_alg_reward["RL"] # RL必须是sample1的位置
        sample2 = self.test_multi_alg_reward["GA"]
        p_value_dict={}
        print("wilcoxon_rank_sum_test")

        u1, p_rl_GA = ranksums(sample1, sample2, alternative='less') # todo 之前没有进行单侧的假设检验：应该加上参数的 alternative='less'的……。证明RL显著小于base
        p_value_dict["GA"] = p_rl_GA
        # u1, p_rl_vns = ranksums(sample1, sample3)
        # p_value_dict["VNS"] =p_rl_vns
        # print(p_value_dict)

        # raw_pvalues = [p_rl_GA]
        # Step 2: Holm校正
        # from statsmodels.stats.multitest import multipletests
        # rejected, adj_p, _, _ = multipletests(raw_pvalues, alpha=0.05, method='holm')

        # print(f"RL vs GA: raw p={raw_pvalues[0]:.4f}, adj p={adj_p[0]:.4f}")
        # print(f"RL vs VNS: raw p={raw_pvalues[1]:.4f}, adj p={adj_p[1]:.4f}")

        # p_value_dict["GA"] = adj_p[0]
        # p_value_dict["VNS"] = adj_p[1]
        print(p_value_dict)
        print("")
        #_------------------------------
        return p_value_dict


    def wilcoxon_signed_rank_test(self):
        '''
        适用于成对。 It is a non-parametric version of the paired T-test.
        '''
        from scipy import stats
        # sample1 = np.asarray(self.reward_greedy)
        # sample2 = np.asarray(self.reward_RL)
        sample1 =self.test_multi_alg_reward["RL"]
        sample2=self.test_multi_alg_reward["Greedy"]
        sample3=self.test_multi_alg_reward["VNS"]
        p_value_dict={}
        print("wilcoxon_signed_rank_test")

        res = stats.wilcoxon(sample1, sample2)
        # print("statistic:", res.__getattribute__("statistic"))
        # print("Greedy pvalue:", res.__getattribute__("pvalue"))
        p_value_dict["Greedy"]=res.__getattribute__("pvalue")
        res = stats.wilcoxon(sample1, sample3)
        # print("VNS pvalue:", res.__getattribute__("pvalue"))
        p_value_dict["VNS"] = res.__getattribute__("pvalue")
        print(p_value_dict)
        print("")
        return p_value_dict

    def save_reward(self,path):
        txt="Greedy,RL\n"
        for greedy,RL in zip(self.reward_greedy,self.reward_RL):
            txt+=f"{greedy},{RL}\n"
        with open(path,"w") as f:
            f.write(txt)
            f.close()

    def Friedman_analysis(self):
        import scipy.stats as stats
        import scikit_posthocs as sp

        # Friedman 检验
        friedman_stat, friedman_p = stats.friedmanchisquare(*(list(self.test_multi_alg_reward.values())))
        print(f"Friedman 检验统计量: {friedman_stat}, p 值: {friedman_p}")

        # Holm post hoc 分析
        if friedman_p < 0.05:  # 如果 Friedman 检验显著
            posthoc_results = sp.posthoc_wilcoxon(list(self.test_multi_alg_reward.values()), p_adjust='holm')
            print("Holm post hoc 分析结果:")
            print(self.test_multi_alg_reward.keys())
            print(posthoc_results)


    def run_analysis(self):
        '''单组实验的统计数据获取'''
        self.get_mean()
        self.get_min_max()
        self.get_variance()
        # self.independent_T_test()
        # self.paired_T_test()
        # self.wilcoxon_rank_sum_test()
        # self.wilcoxon_signed_rank_test()# 成对检查
        # self.Friedman_analysis()
        # path="data_reward.csv"
        # self.save_reward(path)

def run_reward_statistics_analysis(upper, lower, share, run_alg_name, _save_dir):
    '''
    批量实验的统计数据获取
    '''

    import argparse
    import os

    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--task', default='vrp')
    parser.add_argument('--nodes', dest='num_city', default=-1, type=int)
    parser.add_argument('--actor_lr', default=1e-4, type=float)  # 学习率，现在在训练第4epoch，我手动改了一下
    parser.add_argument('--critic_lr', default=1e-4, type=float)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--train-size', default=-1, type=int)
    parser.add_argument('--valid-size', default=100, type=int)
    parser.add_argument('--depot_num', default=10, type=int)
    # 解析为args
    args = parser.parse_known_args()[0]  # colab环境跑使用

    args.test = True

    if share:
        args.checkpoint = os.path.join("时间窗口TW", "TW_random_train_log_share", "200", "2_14_23_17_39")
    else:
        args.checkpoint = os.path.join("时间窗口TW", "TW_random_train_log", "200", "2_14_23_16_34")

    print("比较算法：", run_alg_name)

    tower_num_list = list(range(50, 201, 10))
    depot_num_list = list(range(5, 15))
    print(f"tower_num_list={tower_num_list}")
    print(f"depot_num_list={depot_num_list}")

    # txt = "TOWER\\UAV,"
    # txt += ",".join(map(str, list(depot_num_list))) + "\n"  # 第一行

    for tower_n in tower_num_list:
        for depot_num in depot_num_list:
            args.num_city = tower_n
            args.depot_num=depot_num
            reward_dict,avg_time_dict = run_multi_alg_test(upper, lower, share, args, algorithm=run_alg_name)

            analysis = Reward_Collect()

            analysis.test_multi_alg_reward = reward_dict
            analysis.run_analysis()

            raise ValueError("测试。。。")

def read_and_analysis():
    rl_dir_name="0601 Generalization_TW_random RL"
    GA_dir_name="0529 Generalization_TW_random GA(40-200T"
    uav_list=[10,20]
    tower_list=list(range(40,201,40))
    share=True
    p_value_dict={}
    for u in uav_list:
        for t in tower_list:
            file_name=f"reward on {t}T {u} UAV (share={str(share)}).csv"
            file_path=os.path.join(GA_dir_name,file_name)

            import pandas as pd
            reward_dict={}

            data = pd.read_csv(file_path)
            algorithms_name = data.columns.tolist()
            for alg in algorithms_name:
                reward_dict[alg]=data[alg].values

            file_path = os.path.join(rl_dir_name, file_name)
            data = pd.read_csv(file_path)
            # todo 预处理一下RL 要×10

            reward_dict["RL"]= data["RL"].values

            analysis = Reward_Collect()
            analysis.test_multi_alg_reward = reward_dict
            p_value = analysis.wilcoxon_rank_sum_test()
            p_value_dict[f"{u}-{t}"]=p_value


            # raise ValueError("stop")
    print(p_value_dict)
    text="-,GA\n"
    for k,v in p_value_dict.items():
        text+=f"{k},"+f"{v['GA']},\n"

    with open(os.path.join(GA_dir_name,f"wilson correction_P_value_share={share}.csv"), "w") as f:
        f.write(text)
    f.close()

def RL_Over_analysis():
    rl_dir_name="410 Generalization_TW_random RL VS Greedy VS G_VNS"
    # GVNS_dir_name="0409 Generalization_TW_random Greedy VS G_VNS"

    uav_list=[10,20]
    tower_list=list(range(300,401,50))
    share=True
    p_value_dict={}
    for u in uav_list:
        for t in tower_list:
            file_name=f"reward on {t}T {u} UAV (share={str(share)}).csv"
            file_path=os.path.join(rl_dir_name,file_name)
            import pandas as pd
            reward_dict={}
            try:
                data = pd.read_csv(file_path)
                algorithms_name = data.columns.tolist() # ['Greedy', 'G_VNS']
                for alg in algorithms_name:
                    reward_dict[alg]=data[alg].values

                # from scipy.stats import ranksums
                sample1 = reward_dict["RL"]

                # GVNSfile_path = os.path.join(GVNS_dir_name, file_name)
                # data = pd.read_csv(GVNSfile_path)

                sample2 = reward_dict["Greedy"] # todo这个要手动输入

                # 执行配对Wilcoxon检验
                from scipy.stats import wilcoxon
                res = wilcoxon(sample1, sample2, alternative='less')  # 符号秩？检验MADRL是否显著更优（值更小）
                p=res.pvalue

                print(p)
                # _------------------------------
                p_value_dict[f"{u}-{t}"]= p
            except:
                pass
            # raise ValueError("stop")
    print(p_value_dict)
    text=f"-,RL_GVNS,显著,\n"
    for k,v in p_value_dict.items():
        text+=f"{k},"+f"{v},"+f"{v<0.05}"+"\n"

    # with open(os.path.join(f"wilson_pair_RL_GVNS_P_value_share={share}.csv"), "w") as f:
    #     f.write(text)
    # f.close()

if __name__=="__main__":
    # todo 注意检查参数啊啊啊啊
    # upper, lower=1.1,0.9
    share=True
    # run_alg_name=[
    #     "RL",
    #     "RL_Over",
    #     "Greedy",
    #     # "Greedy_Over"
    # ]
    # save_dir = "Hypothesis_TW_random " + " VS ".join(run_alg_name)
    # run_reward_statistics_analysis(upper, lower, share, run_alg_name, save_dir)

    read_and_analysis()
    # RL_Over_analysis()

