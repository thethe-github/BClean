from pgmpy.models import BayesianModel, BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.estimators import ParameterEstimator
import pandas as pd
from pgmpy.estimators import ExhaustiveSearch, HillClimbSearch, TreeSearch, PC
from pgmpy.estimators import BDeuScore, K2Score, BicScore, StructureScore,BDsScore
from pgmpy.estimators import MmhcEstimator
import joblib
import Levenshtein, math
from distribute.Compensative import CompensativeParameter
from src.structure import Compensative
from src.structure import BN_Structure
from src.infer import Inference
from dataset import Dataset
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

class BayesianClean:
    def __init__(self,
                 dirty_df,  
                 clean_df,
                 infer_strategy = "PIPD", 
                 tuple_prun = 1.0, 
                 maxiter = 1, 
                 num_worker = 32, 
                 chunksize = 250, 
                 model_path=None,
                 model_save_path=None,
                 attr_type=None,
                 fix_edge=None,
                 model_choice=None
                 ):
        self.start_time = time.perf_counter()
        print("+++++++++data loading++++++++")
        self.attr_type = attr_type
        self.dataLoader = Dataset()
        self.dirty_data = dirty_df
        self.clean_data = clean_df
        self.fix_edge = fix_edge
        self.model_path = model_path
        self.model_save_path = model_save_path
        self.model_choice = model_choice
        self.infer_strategy = infer_strategy
        self.tuple_prun = tuple_prun
        self.maxiter = maxiter
        self.num_worker = num_worker
        self.chunksize = chunksize
        self.data = self.dataLoader.pre_process_data(self.dirty_data, self.attr_type)
        print("+++++++++data loading complete++++++++")
        
        print("+++++++++computing error cell++++++++")
        self.actual_error = self.dataLoader.get_error(self.dirty_data, self.clean_data)
        print("error:{}".format(len(self.actual_error)))
        print("+++++++++error cell computing complete++++++++")
        
        print("+++++++++correlation computing++++++++")
        self.Compensative = Compensative(self.data, self.attr_type)
        self.Occurrence_list, self.Frequence_list, self.Occurrence_1 = self.Compensative.build()
        print("+++++++++correlation computing complete++++++++")
        
        self.structure_learing = BN_Structure(data = self.data,
                                                   model_path = self.model_path,
                                                   model_save_path = self.model_save_path,
                                                   model_choice = self.model_choice,
                                                   fix_edge = self.fix_edge)
        self.model, self.model_dict = self.structure_learing.get_bn()
        
        self.CompensativeParameter = CompensativeParameter(attr_type=self.attr_type,
                                                           domain=self.Frequence_list,
                                                           occurrence=self.Occurrence_list,
                                                           model=self.model,
                                                           df=self.data)

        self.Inference = Inference(dirty_data = self.dirty_data,
                                     data = self.data,
                                     model = self.model,
                                     model_dict = self.model_dict,
                                     attrs = self.attr_type,
                                     Frequence_list = self.Frequence_list,
                                     Occurrence_1 = self.Occurrence_1,
                                     CompensativeParameter = self.CompensativeParameter,
                                     infer_strategy = "PIPD"
                                     )
        
        self.repair_list = self.Inference.Repair(self.data,
                                                 self.clean_data,
                                                 self.model,
                                                 self.attr_type)
        
        self.end_time = time.perf_counter()


   
    def transform_data(self, data):
        data = data.fillna("A Null Cell")
        all_domain_map = {}
        for val in data:
            domain_val = data[val].unique()
            count = 0
            domain_map = {}
            for v in domain_val:
                domain_map[v] = count
                count += 1
            data[val] = data[val].apply(lambda x : x)
            all_domain_map[val] = domain_map
        return data, all_domain_map

    def produce_train(self, data, attrs):
        attrs = [attr for attr in attrs]
        data = data[attrs]
        data = data.applymap(lambda x: np.NAN if(x == "A Null Cell") else x)
        return data




    def Occurrence_score(self, pred, val, attr_main, tuple, t_id):
        Occurrence_list = self.Occurrence_list
        Frequence_list = self.Frequence_list
        P_Ai = 1
        candidate = val[len(attr_main + "_"):]
        for attr_vice in tuple:
            if (attr_main == attr_vice):
                continue
            val_vice = tuple.loc[t_id, attr_vice]
            occur = Occurrence_list[attr_main][candidate][attr_vice][val_vice] if (val_vice in Occurrence_list[attr_main][candidate][attr_vice]) else 0
            P_Ai *= occur / Frequence_list[attr_main][candidate] 
            if(P_Ai == 0.0):
                break
        return P_Ai


    def re_repair_and_update_parameter(self, repair_data, line, model, nodes_list_sort, attr_type):
        df_train = repair_data.copy()
        self.model.fit_update(df_train) 
        self.Occurrence_list, self.Frequence_list = self.occur_and_fre(df_train) 
        reprocess_line = line - 100
        while (reprocess_line < line):
            data_line = repair_data.iloc[[reprocess_line]].copy()
            self.repair_line(repair_data, data_line,  
                             reprocess_line, model, nodes_list_sort, attr_type)
            reprocess_line += 1

    
    def reconstruct_network(self, model, train_data):
        next_iter_model = BayesianNetwork()
        nodes, edges = list(model.nodes), list(model.edges)
        for node in nodes:
            next_iter_model.add_node(node)
        for edge in edges:
            next_iter_model.add_edge(edge[0], edge[1])

        next_iter_model.fit(
            train_data, estimator=BayesianEstimator, complete_samples_only=True)
        self.occur_and_fre(train_data)
        return next_iter_model

    





    