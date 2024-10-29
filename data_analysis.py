import  numpy as np

class Reward_Collect:
    def __init__(self):
        self.reward_greedy=[]
        self.reward_RL=[]


    def get_mean(self):
        print(f"Greedy reward mean: {np.mean(self.reward_greedy)}")
        print(f"RL reward mean: {np.mean(self.reward_RL)}")
        print("")


    def get_min_max(self):
        print(f"Greedy reward range: [{min(self.reward_greedy)},{max(self.reward_greedy)}]")
        print(f"RL reward range: [{min(self.reward_RL)},{max(self.reward_RL)}]")
        print("")


    def get_variance(self):
        print(f"Greedy variance {np.var(self.reward_greedy,dtype=np.float64)}")
        print(f"RL variance {np.var(self.reward_RL,dtype=np.float64)}")
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


    def run_analysis(self):
        if len(self.reward_RL)!=len(self.reward_greedy):
            raise ValueError(f"RL reward length={len(self.reward_RL)}\n"
              f"Greedy reward length={len(self.reward_greedy)} is not equal!")

        self.get_mean()
        self.get_min_max()
        self.get_variance()
        self.independent_T_test()
        self.paired_T_test()
        self.wilcoxon_rank_sum_test()
        self.wilcoxon_signed_rank_test()
        path="data_reward.csv"
        self.save_reward(path)


if __name__=="__main__":
    exp=Reward_Collect()
    import pandas as pd

    df = pd.read_csv("data_reward.csv", encoding="utf-8")
    exp.reward_RL=np.array(df['RL'])
    exp.reward_greedy=np.array(df["Greedy"])

    exp.run_analysis()