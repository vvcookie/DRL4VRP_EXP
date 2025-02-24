import  numpy as np
from Algrithm_compare import run_multi_alg_test

class Reward_Collect:
    def __init__(self):
        # self.reward_greedy=[]
        # self.reward_RL=[]

        self.test_multi_alg_reward={} # todo


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
        from scipy import stats
        sample1 = np.asarray(self.reward_greedy)
        sample2 = np.asarray(self.reward_RL)
        res = stats.mannwhitneyu(sample1, sample2)
        print("wilcoxon_rank_sum_test")
        # print("statistic:", res.__getattribute__("statistic"))
        print("pvalue:", res.__getattribute__("pvalue"))
        print("")


    def wilcoxon_signed_rank_test(self):
        '''
        适用于成对。 It is a non-parametric version of the paired T-test.
        '''
        from scipy import stats
        sample1 = np.asarray(self.reward_greedy)
        sample2 = np.asarray(self.reward_RL)
        res = stats.wilcoxon(sample1, sample2)
        print("wilcoxon_signed_rank_test")
        # print("statistic:", res.__getattribute__("statistic"))
        print("pvalue:", res.__getattribute__("pvalue"))
        print("")

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
        # self.wilcoxon_signed_rank_test()
        self.Friedman_analysis()
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
    parser.add_argument('--valid-size', default=500, type=int)
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
            reward_dict = run_multi_alg_test(upper, lower, share, args, algorithm=run_alg_name)

            analysis = Reward_Collect()

            analysis.test_multi_alg_reward = reward_dict
            analysis.run_analysis()

            raise ValueError("测试。。。")


if __name__=="__main__":
    # todo 注意检查参数啊啊啊啊
    upper, lower=1.1,0.9
    share=False
    run_alg_name=[
        "RL",
        "RL_Over",
        "Greedy",
        # "Greedy_Over"
    ]
    save_dir = "Hypothesis_TW_random " + " VS ".join(run_alg_name)
    run_reward_statistics_analysis(upper, lower, share, run_alg_name, save_dir)
