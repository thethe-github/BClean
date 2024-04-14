import os

from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators import ParameterEstimator
import pandas as pd
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch, TreeSearch, PC
from pgmpy.estimators import BDeuScore, K2Score, BicScore, StructureScore, BDsScore
from pgmpy.estimators import MmhcEstimator
import joblib
import Levenshtein, math
import re
import time
import multiprocessing
from sklearn.covariance import graphical_lasso
from sklearn import covariance, preprocessing
from scipy import sparse
from sksparse.cholmod import cholesky, analyze
import numpy as np

from concurrent import futures
from tqdm import tqdm


# from analysis import analysis

class Dataset:
	def __init__(self):
		self.tags = "A Null Cell"
		self.actual_error = {}
	
	# def get_data(self, path, attrs):
	# 	if (attrs == None):
	# 		return pd.read_csv(path, dtype = str)
	# 	df = pd.read_csv(path, dtype = "str")
	# 	attrs = [attr for attr in attrs]
	# 	df = df[attrs]
	# 	df = df.fillna(self.tags)
	# 	return df

	def get_data(self, path):
		df = pd.read_csv(path, dtype = "str")
		return df
	
	def get_real_data(self, data, attr_type):
		attrs = [attr for attr in attr_type]
		df = data.copy()
		df = df[attrs]
		df = df.fillna(self.tags)
		return df
	
	def pre_process_data(self, data, attr_type):
		'''
		:param data: Original data
		:param attrs: UC
		:return: Pre-processed data for BN training and clean
		'''
		attrs = [attr for attr in attr_type]
		df = data.copy()
		df = df[attrs]
		df2 = df.copy()
		df_train = pd.DataFrame(columns = attrs)  
		
		for line in tqdm(range(df2.shape[0])):
			data_line = df2.iloc[[line]].copy()
			add_lines = []
			attr_line = []
			for at in attrs:
				if (attr_type[at]["pattern"] != None):  
					p = attr_type[at]["pattern"]
					search = re.search(p, data_line.loc[line, at])
					if (search != None):
						val = search.group()
						if (attr_type[at]["type"] == "Numerical"):
							val = float(val)
							if (val.is_integer()):
								val = int(val)
						l = (str(val),)
					else:
						l = (data_line.loc[line, at],)
				else:
					l = (data_line.loc[line, at],)
				add_lines.append(l)
				attr_line.append(at)
			
			add_lines_num = 1
			for i in range(len(add_lines)):
				add_lines_num *= len(add_lines[i])
			
			for add_line in range(add_lines_num):
				add_line_data = data_line.copy()
				add_line_data = self.change_add_line(add_line_data, line, add_lines, attr_line, attrs)
				add_line_data.drop_duplicates()
				df_train = df_train.append(add_line_data, ignore_index = True)
		
		return df_train
	
	
	def change_add_line(self, data_line, line_index, add_lines, attr_line, attrs):
		'''
		
		:param data_line: Specific line of dataset
		:param line_index: Index of specific line
		:param add_lines: Modified version
 		:param attr_line: Modified attribute
		:param attrs: All attributes
		:return: None
		'''
		involve_attrs_num = len(add_lines) 
		key = [attr for attr in attr_line] 
		value = [[v for v in l] for l in add_lines]  
		re_data = pd.DataFrame(columns = attrs)
		temp_re_data = data_line.copy()
		re_data = self.dfs_line(re_data, temp_re_data, line_index, 0, 0, involve_attrs_num, key, value) 
		return re_data
	
	def dfs_line(self, re_line, temp_re_data, line_index, attr_index, sub_index, involve_attrs_num, key, value):
		if (attr_index == involve_attrs_num):  
			re_line = re_line.append(temp_re_data)
			return re_line
		
		change_value = value[attr_index][sub_index]
		change_attr = key[attr_index]
		temp_re_data.loc[line_index, change_attr] = change_value
		
		for sub in range(len(value[attr_index])):
			return self.dfs_line(re_line, temp_re_data, line_index, attr_index + 1, sub, involve_attrs_num, key, value)
	
	def get_error(self, df1, df2):
		self.get_actual_error(df1, df2)
		return self.actual_error
	
	def get_actual_error(self, df1, df2):
		indexs = [(i, attr) for i in range(df1.shape[0]) for attr in list(df1.columns)]
		executer = futures.ThreadPoolExecutor(max_workers = multiprocessing.cpu_count())
		fs = []
		for index in tqdm(indexs):
			a = executer.submit(self._different_and_TransToStr, [index, df1, df2])
			fs.append(a)
		flag = True
		while (True):
			for f in fs:
				flag = True and f.done()
			if (flag == True):
				executer.shutdown()
				break
		print("++++++++++++{} error cell collected++++++++++++".format(len(self.actual_error)))
	
	def _different_and_TransToStr(self, args):
		cell, df1, df2 = args
		if (df1.loc[cell[0], cell[1]] != df2.loc[cell[0], cell[1]]):
			self.actual_error[cell] = df2.loc[cell[0], cell[1]]
