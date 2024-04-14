import time
from BClean.BClean import BayesianClean
from BClean.analysis import analysis
from BClean.dataset import Dataset
from BClean.src.UC import UC

if __name__ == '__main__':
	dirty_path = "../dataset/Inpatient/Inpatient_dirty_10.csv"
	clean_path = "../dataset/Inpatient/Inpatient_clean.csv"
	dataLoader = Dataset()
	dirty_data = dataLoader.get_data(path = dirty_path)
	clean_data = dataLoader.get_data(path = clean_path)
	model_path = None
	model_save_path = None
	fix_edge = []

	attr = UC(dirty_data)
	attr.build_from_json("../UC_json/inpatient.json")
	
	
	pat = attr.PatternDiscovery()
	print("pattern discovery:{}".format(pat))

	attr = attr.get_uc()

	dirty_data = dataLoader.get_real_data(dirty_data, attr_type = attr)
	clean_data = dataLoader.get_real_data(clean_data, attr_type = attr)

	start_time = time.perf_counter()

	model = BayesianClean(dirty_df = dirty_data, clean_df = clean_data,model_path = model_path,model_save_path = model_save_path, attr_type = attr,
	                      fix_edge = fix_edge,
	                      model_choice = "bdeu",
	                      infer_strategy = "PIPD",
	                      tuple_prun = 1.0,
	                      maxiter = 1,
	                      num_worker = 32,
	                      chunksize = 126)
	
	dirty_data, repair_data, clean_data = (model.repair_list)[0], (model.repair_list)[
		1], (model.repair_list)[2]
	
	actual_error, repair_error = model.actual_error, (model.repair_list)[3]
	
	P, R, F = analysis(actual_error, repair_error, dirty_data, clean_data)
	
	print("Repair Pre:{:.5f}, Recall:{:.5f}, F1-score:{:.5f}".format(P, R, F))
	print("++++++++++++++++++++time using:{}+++++++++++++++++++++++".format(model.end_time - model.start_time))
	print("date:{}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))