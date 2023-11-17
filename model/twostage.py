import argparse
# from dgl.data import register_data_args
import time
# from datasets.dataloader import emb_dataloader
from utils.evaluate import baseline_evaluate
import fire
import logging
from embedding.get_embedding import embedding
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
from pyod.models.pca import PCA
#from pyod.models.auto_encoder import AutoEncoder
import os

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score,precision_score,recall_score,average_precision_score,roc_auc_score,roc_curve

def main():
	parser = argparse.ArgumentParser(description='baseline')
	# register_data_args(parser)
	parser.add_argument("--mode", type=str, default='X',choices=['A','AX','X'],#A:DW X:row
			help="dropout probability")
	parser.add_argument("--seed", type=int, default=-1,
            help="random seed, -1 means dont fix seed")
	parser.add_argument("--emb-method", type=str, default='DeepWalk',
			help="embedding methods: DeepWalk, Node2Vec, LINE, SDNE, Struc2Vec")  
	parser.add_argument("--ad-method", type=str, default='OCSVM',
			help="embedding methods: PCA,OCSVM,IF")

	args = parser.parse_args()
	DATASETS_NAME = {
		 'cora': 7,
		# 'citeseer': 6,
		# 'pubmed': 3,
		# 'BlogCatalog':6,
		#  'Flickr':9,
		#'ACM': 9,

	}
	SEED=['1']
	if args.seed!=-1:
		np.random.seed(args.seed)

	for dataset_name in list(DATASETS_NAME.keys()):
		AUCs = {}
		APs = {}
		MAX_EPOCHs = {}
		args.dataset = dataset_name
		results_dir = './logs/deepwork/'
		args.outf = results_dir
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
		file2print = open('{}/results_deepwork.log'.format(results_dir), 'a+')
		file2print_detail = open('{}/results_deepwork_detail.log'.format(results_dir), 'a+')

		import datetime

		print(datetime.datetime.now())
		print(datetime.datetime.now(), file=file2print)
		print(datetime.datetime.now(), file=file2print_detail)
		print("Model\tDataset\tNormal_Label\tAUC_mean\tAUC_std\tAP_mean\tAP_std",
			  file=file2print_detail)

		print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std")
		print("Model\tDataset\tTest\tAUC_mean\tAUC_std\tAP_mean\tAP_std", file=file2print)
		file2print.flush()
		file2print_detail.flush()
		for normal_idx in range(DATASETS_NAME[dataset_name]):
			if args.dataset in ['BlogCatalog' ,'Flickr','ACM']:
				normal_idx+=1
			args.normal_class = normal_idx
			MAX_EPOCHs_seed = {}
			AUCs_seed = {}
			APs_seed = {}
			for seed in SEED:
				args.seed = seed
				torch.manual_seed(args.seed)
				torch.cuda.manual_seed_all(args.seed)
				savedir = './embeddings/deepwork/' + args.dataset + '/{}'.format(args.normal_class)
				if not os.path.exists(savedir):
					os.makedirs(savedir)




				t0 = time.time()


				dur1=time.time() - t0

					#print('AX shape',data.shape)

				if args.ad_method=='OCSVM':
					clf = OCSVM(contamination=0.1)
				if args.ad_method=='IF':
					clf = IForest(n_estimators=100,contamination=0.1,n_jobs=-1,behaviour="new")
				if args.ad_method=='PCA':
					clf = PCA(contamination=0.1)

				t1 = time.time()
				clf.fit(data[datadict['train_mask']])
				dur2=time.time() - t1

				print('traininig time:', dur1+dur2)

				t2 = time.time()
				y_pred_val=clf.predict(data[datadict['val_mask']])
				y_score_val=clf.decision_function(data[datadict['val_mask']])
				#auc,ap,f1,acc,precision,recall=baseline_evaluate(datadict,y_pred_val,y_score_val,val=True)

				dur3=time.time() - t2
				print('infer time:', dur3)

				y_pred_test=clf.predict(data[datadict['test_mask']])
				y_score_test=clf.decision_function(data[datadict['test_mask']])
				auc,ap,f1,acc,precision,recall=baseline_evaluate(datadict,y_pred_test,y_score_test,val=False)
				np.save(os.path.sep.join([savedir + '/embeddings.npy']),
						data[datadict['test_mask']].data)
				np.save(os.path.sep.join([savedir + '/label.npy']),
						datadict['labels'][datadict['test_mask']])

				AUCs_seed[seed] = auc
				APs_seed[seed] = ap
			AUCs_seed_mean = round(np.mean(list(AUCs_seed.values())), 4)
			AUCs_seed_std = round(np.std(list(AUCs_seed.values())), 4)
			APs_seed_mean = round(np.mean(list(APs_seed.values())), 4)
			APs_seed_std = round(np.std(list(APs_seed.values())), 4)
			print("Dataset: {} \t Normal Label: {} \t AUCs={}+{} \t APs={}+{} \t ".format(
				dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std, APs_seed_mean, APs_seed_std))

			print("{}\t{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
				'deepwork', dataset_name, normal_idx, AUCs_seed_mean, AUCs_seed_std,
				APs_seed_mean,
				APs_seed_std), file=file2print_detail)
			file2print_detail.flush()
			AUCs[normal_idx] = AUCs_seed_mean
			APs[normal_idx] = APs_seed_mean
		print("{}\t{}\tTest\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(
			'deepwork', dataset_name, np.mean(list(AUCs.values())),
			np.std(list(AUCs.values())),
			np.mean(list(APs.values())), np.std(list(APs.values()))))
		print("{}\t{}\tTest\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t".format(
			'deepwork', dataset_name, np.mean(list(AUCs.values())),
			np.std(list(AUCs.values())),
			np.mean(list(APs.values())), np.std(list(APs.values()))),file=file2print)
		file2print.flush()

if __name__ == '__main__':
	main()
