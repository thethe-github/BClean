import Levenshtein
import numpy as np
import math
import re


class CPT_estimator:
	def __init__(self, estimator):
		self.estimator = estimator
	
	def get_cpts(self, model, data, choice):
		if (choice == "org"):
			model.fit(data, estimator = self.estimator)
			return model
		for sub_net in model:
			attrs_temp = list(model[sub_net].nodes)
			data_temp = data[attrs_temp]
			model[sub_net].fit(data_temp, estimator = self.estimator)
		return model


class CompensativeParameter:
	def __init__(self, attr_type, domain, occurrence, model, df):
		# 类的初始化方法，接收属性类型、域、出现频率、模型和Dataframe作为参数
		self.attr_type = attr_type
		self.domain = domain
		self.occurrence = occurrence
		self.model = model
		self.df = df
		self.tf_idf = {}
	
	# 计算给定观测值的罚分
	def return_penalty(self, obs, attr, index, data_line, prior):
		p_obs_G_cand = {} # 初始化候选罚分字典
		domain_base = {} # 域基础字典，用于存储基于域的计算结果
		cooccurance_base = {} # 共现基础字典
		cooccurance_dist = {} # 共现距离字典
		total_ground = 0 # 总基础分数
		toatl_cooccur = 0 # 总共现分数
		occurrence = self.occurrence[attr] # 当前属性的出现频率
		attr_type = self.attr_type # 属性类型
		
		if (obs == "A Null Cell" and attr_type[attr]["AllowNull"] == "N"):
			obs = "" # 如果观测值为空且当前属性不允许空值，则将观测值设置为空字符串
		for groud in prior:
			# 计算域基础值，基于Levenshtein距离
			domain_base[groud] = 1 + Levenshtein.distance(obs, groud) 
			total_ground += domain_base[groud]
			cooccurance_base_vec = []
			for other_attr in attr_type: # 遍历所有属性，计算共现基础向量
				# 排除当前属性及其贝叶斯网络中的父节点和子节点
				if (other_attr == attr or other_attr in self.model.get_parents(
						node = attr) or other_attr in self.model.get_children(node = attr)):
					continue
				other_val = data_line.loc[index, other_attr]
				# 计算共现得分
				score = occurrence[groud][other_attr][other_val] if (other_val in occurrence[groud][other_attr]) else 0
				cooccurance_base_vec.append(score)
			
			# 计算共现距离，基于L2范数和Levenshtein距离
			# cooccurance_dist[groud] = np.linalg.norm(np.array(cooccurance_base_vec), ord=2) + 1
			cooccurance_dist[groud] = np.linalg.norm(np.array(cooccurance_base_vec), ord = 2) + math.exp(
				(-1) * Levenshtein.distance(groud, obs)) + 1
			toatl_cooccur += cooccurance_dist[groud]
		
		# 再次遍历先验概率中的每个可能值，计算最终的罚分
		for groud in prior:
			# 根据属性类型和正则表达式条件过滤不符合条件的值
			if (not ((attr_type[attr]["AllowNull"] == "Y" or
			          (attr_type[attr]["AllowNull"] == "N" and
			           groud != "A Null Cell")) and
			         (attr_type[attr]["pattern"] == None or
			          (attr_type[attr]["pattern"] != None and
			           re.search(attr_type[attr]["pattern"], groud))))
			):
				# 如果不符合条件，设置相关值为0
				domain_base[groud] = 0
				cooccurance_dist[groud] = 0
				p_obs_G_cand[groud] = 0
			else:
				# 如果符合条件，计算最终罚分
				domain_base[groud] = domain_base[groud] / total_ground
				cooccurance_dist[groud] = cooccurance_dist[groud] / toatl_cooccur
				p_obs_G_cand[groud] = cooccurance_dist[groud]
		
		return p_obs_G_cand
	
	def return_penalty_test(self, obs, attr, index, data_line, prior, attr_order):
		if (self.tf_idf[attr] == None):
			return {p: 1 for p in prior}
		p_obs_G_cand = {}
		obs_combine = ""
		combine_attrs, dic, dic_idf = self.tf_idf[attr]
		for at in combine_attrs:
			obs_combine += "," + data_line.loc[index, at]
		
		for ground in prior:
			obs_context = ground + obs_combine
			tf = dic.loc[obs_context] if (obs_context in dic) else 0
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
			if (len(combine_attrs) == 0):
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