import Levenshtein
import numpy as np
import math
import re
class CompensativeParameter:
    def __init__(self, attr_type, domain, occurrence, model, df):
        self.attr_type = attr_type
        self.domain = domain
        self.occurrence = occurrence
        self.model = model
        self.df = df
        self.tf_idf = {}

    # 计算一个属性的观察值 obs 对于给定的先验值 prior 的惩罚分数
    def return_penalty(self, obs, attr, index, data_line, prior):
        p_obs_G_cand = {}
        domain_base = {}
        cooccurance_base = {}
        cooccurance_dist = {}
        total_ground = 0
        toatl_cooccur = 0
        occurrence = self.occurrence[attr]
        attr_type = self.attr_type

        if (obs == "A Null Cell" and attr_type[attr]["AllowNull"] == "N"):
            obs = ""
        for groud in prior:
            domain_base[groud] = 1 + Levenshtein.distance(obs, groud) # 计算 obs 与 groud 的距离
            total_ground += domain_base[groud]
            cooccurance_base_vec = []
            for other_attr in attr_type:
                if(other_attr == attr or other_attr in self.model.get_parents(node=attr) or other_attr in self.model.get_children(node=attr)):
                    continue
                other_val = data_line.loc[index, other_attr] # 获取 data_line 中对应的属性值 other_val
                score = occurrence[groud][other_attr][other_val] if(other_val in occurrence[groud][other_attr]) else 0 # 查找在 occurrence 中是否有关于 groud、attr 和 other_val 的共现信息
                # 如果有，将共现分数添加到 cooccurance_base_vec 中
                cooccurance_base_vec.append(score)

            cooccurance_dist[groud] = np.linalg.norm(np.array(cooccurance_base_vec), ord=2) + 1  # 计算cooccurance_base_vec中的分数向量的L2范数，以度量属性attr与其他属性之间的共现分数的复杂性，加1避免除以零错误。
            toatl_cooccur += cooccurance_dist[groud]  # 将这些值添加到 toatl_cooccur 中，以计算共现分数的总和。


        for groud in prior:
            if(not((attr_type[attr]["AllowNull"] == "Y" or
                    (attr_type[attr]["AllowNull"] == "N" and
                     groud != "A Null Cell")) and
                   (attr_type[attr]["pattern"] == None or
                    (attr_type[attr]["pattern"] != None and
                     re.search(attr_type[attr]["pattern"],groud))))  # 如果不满足这些条件
            ):
                domain_base[groud] = 0 # 都设置为0
                cooccurance_dist[groud] = 0
                p_obs_G_cand[groud] = 0
            else:
                domain_base[groud] = domain_base[groud] / total_ground
                cooccurance_dist[groud] = cooccurance_dist[groud] / toatl_cooccur
                p_obs_G_cand[groud] = cooccurance_dist[groud]

        return p_obs_G_cand
    

    def return_penalty_test(self, obs, attr, index, data_line, prior, attr_order):
        if(self.tf_idf[attr] == None):
            return {p: 1 for p in prior}
        p_obs_G_cand = {}
        obs_combine = ""
        combine_attrs, dic, dic_idf = self.tf_idf[attr]
        for at in combine_attrs:
            obs_combine += "," + data_line.loc[index, at]
        
        for ground in prior:
            obs_context = ground + obs_combine
            tf = dic.loc[obs_context] if (obs_context in dic) else 0 # 计算 TF-IDF（Term Frequency-Inverse Document Frequency） 分数。TF 表示在给定 obs_context 中属性值的频率，而 IDF 表示逆文档频率，衡量了 obs 与 obs_context 的关联性。这些分数用于衡量观察值 obs 与先验值 ground 之间的相似性或关联性。
            if (tf == 0):
                continue
            idf = math.log(self.df.shape[0] / (dic_idf.loc[obs] + 1))
            if (idf == 0):
                continue
            p_obs_G_cand[ground] = tf * idf
        return p_obs_G_cand
    
    def init_tf_idf(self, attr_order):
        for attr in self.attr_type:
            combine_attrs = []
            for at in attr_order:
                if (at == attr or not (at in self.model.get_parents(
                        node = attr) or at in self.model.get_children(node = attr))):
                    continue
                combine_attrs.append(at)
            if(len(combine_attrs) == 0):
                self.tf_idf[attr] = None
                continue
            context_attr = "_".join(combine_attrs)
            context_attr += "_TempAttribute"
            df = self.df.copy()
            for at in combine_attrs:
                if (context_attr not in df.columns):
                    df[context_attr] = df[at]
                else:
                    df[context_attr] = df[context_attr] + "," + df[at]
                    
            context_obs = attr + "_" + context_attr
            df[context_obs] = df[attr] + "," + df[context_attr]
            dic = df[context_obs].value_counts()
            dic_idf = df[attr].value_counts()
            self.tf_idf[attr] = [combine_attrs, dic, dic_idf]