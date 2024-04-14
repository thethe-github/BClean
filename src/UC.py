import pandas as pd
import json
from tdda import rexpy
import re

class UC:
	def __init__(self, data: pd.DataFrame):
		self.data = data
		self.res = {}
	
	# 获得该数据的用户约束
	def get_uc(self):
		return self.res
	
	# 计算属性的最小或最大值
	def default_setting(self, attr, type, name):
		result = None
		domain = list(self.data[attr].unique())
		if(type == "Categorical"):
			result = min(domain, key = lambda x: len(x)) if(name == "min_v") else result
			result = max(domain, key = lambda x: len(x)) if (name == "max_v") else result
		elif(type == "Numerical"):
			result = min(domain) if(name == "min_v") else result
			result = max(domain) if (name == "max_v") else result
		return result if(type == "Numerical") else len(result)
	
	# 为特定属性定义UC
	def build(self, attr, type="Categorical", min_v=None, max_v=None, null_allow="N", repairable="Y", pattern=None):
		if(attr not in self.data.columns):
			print("no attributes")
			return
		min_v = min_v if(min_v != None) else self.default_setting(attr, type, "min_v")
		min_v = min_v if (min_v != None) else self.default_setting(attr, type, "max_v")
		self.res[attr] = {"type": type, "min_length": min_v, "max_length": max_v, "AllowNull": null_allow, "repairable": repairable, "pattern": pattern}
		print("UC of attribute {} has been set".format(attr))
		
	# def build_from_json(self, jpath):
	# 	try:
	# 		js = json.load(open(jpath, "r"))
	# 		for k1 in js:
	# 			self.res[k1] = {}
	# 			for k2 in js[k1]:
	# 				if k2 == "pattern":
	# 					self.res[k1][k2] = re.compile(js[k1][k2])
	# 				else:
	# 					self.res[k1][k2] = js[k1][k2]
	# 	except:
	# 		print("No json file path '{}', generation from data".format(jpath))
	# 		for col in self.data:
	# 			self.build(attr=col)

	def build_from_json(self, jpath):
		try:
			js = json.load(open(jpath, "r"))
			for k1 in js:
				self.res[k1] = {}
				for k2 in js[k1]:
					if k2 == "pattern":
						if js[k1][k2]:
							self.res[k1][k2] = re.compile(js[k1][k2])  # 使用 re.compile 方法创建正则表达式对象，并将其存储在 self.res[k1][k2] 中
						else:
							self.res[k1][k2] = js[k1][k2]
						print(self.res)		
					else:
						self.res[k1][k2] = js[k1][k2]
		except:
			print("No json file path '{}', generation from data".format(jpath))
			for col in self.data:
				self.build(attr=col)
	
	def edit(self, df_attr, uc_attr, uc_v):
		if(df_attr not in self.res):
			print("no attributes")
			return
		if(uc_attr not in self.res[df_attr]):
			print("no uc_attributes")
			return
		
		self.res[df_attr][uc_attr] = uc_v
		print("Attribute modified")
	
	# 查看每一列的正则表达式
	def PatternDiscovery(self):
		res = {}
		for col in self.data.columns:
			res[col] = rexpy.pdextract(self.data[col])  # rexpy 是一个 Python 库，用于从数据中提取正则表达式模式。
		return res
		
	
	def remove(self, df_attr, uc_attr):
		if (df_attr not in self.res):
			print("no attributes")
			return
		if (uc_attr not in self.res[df_attr]):
			print("no uc_attributes")
			return
		
		self.res[df_attr][uc_attr] = None
		print("Attribute removed")