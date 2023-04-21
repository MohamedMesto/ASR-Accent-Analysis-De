import numpy as np
import torch
import os
import json
import math
import pdb
import time
import pdb
import argparse
from tqdm import tqdm
import pandas as pd
import csv
import glob
import textgrid
import csv
import os
import glob


main_path='/home/mmm2050/QU_DFKI_Thesis/Experimentation/ASR_Accent_Analysis_De'



def get_input_frame(current_frame):
	return (current_frame - 1)*2 + 11 - 2*5





def data_prepare(csv_path, file_info_path, data_path, rep_type, target_path):
	with open(csv_path, 'r') as f:
		ids = f.readlines()
		ids = [x.strip().split(',') for x in ids]
		#self.ids = ids
		samp_rate = 16000
		spec_stride = 0.01
		window_size = 0.02
		size = len(ids)
		rep_path = os.path.join(data_path, rep_type)
		#self.file_info_path = file_info_path
 
		
		# MMM comment
		# with open(file_info_path, 'r') as j:
		# 	file_meta = json.load(j)
		

		# create the file_meta

		############################################################


		#####################################################################################
		########### set the Data files paths on Conda Notebook on Ubuntu #################### 

		### 
		os.chdir('/home/mmm2050/QU_DFKI_Thesis/Experimentation/ASR_Accent_Analysis_De')
		main_path=os.getcwd()

		if not os.path.exists(main_path+'/Data_results'):
			os.makedirs(main_path+'/Data_results')

		if not os.path.exists(main_path+'/Figures_results'):
			os.makedirs(main_path+'/Figures_results')

		Data_path=main_path+'/Data/'
		Data_results_path=main_path+'/Data_results/'
		json_file_path=Data_path+'results.json'
		validated_tsv_path=Data_path+'validated.tsv'
		validated_tsv_path_small=Data_path+'validated_small.tsv'
		

		validated_en_tsv_path=main_path+'/DeepSpeech/data/validated.tsv'
		validated_en_tsv_path_small=main_path+'/DeepSpeech/data/validated_small.tsv'

		MCV_en_all=pd.DataFrame({})
		dataset_trans_en_test_all_duration=pd.DataFrame({})

		# df_tsv_en_data = pd.read_csv(validated_en_tsv_path, sep='\t', encoding="utf-8")

		df_tsv_en_data = pd.read_csv(validated_en_tsv_path_small, sep='\t', encoding="utf-8")



		df_tsv_en_data.drop(df_tsv_en_data[(df_tsv_en_data['accent'].isna())].index, inplace=True)
		df_tsv_en_data.drop(df_tsv_en_data[(df_tsv_en_data['sentence'].isna())].index, inplace=True)


		df_tsv_en_data['path'] = df_tsv_en_data['path'].map(lambda x: x.split('.',1)[0])
		# df_tsv_en_data['path'] = df_tsv_en_data['path'].map(lambda x: x.split('/',6)[3])
		# df_tsv_en_data['sentence'] = df_tsv_en_data['sentence'].map(lambda x: x.split('": "',2)[1])
		# df_tsv_en_data['sentence'] = df_tsv_en_data['sentence'].map(lambda x: x.split('"',2)[0])
		# df_tsv_en_data['duration'] = df_tsv_en_data['duration'].map(lambda x: x.split('": ',2)[1])
		# df_tsv_en_data['duration'] = df_tsv_en_data['duration'].map(lambda x: x.split('}',2)[0])


		dataset_trans_en_test_all_duration['audio_filepath']=df_tsv_en_data['path']
		dataset_trans_en_test_all_duration['transcript']=df_tsv_en_data['sentence']
		# dataset_trans_en_test_all_duration['duration']=df_tsv_en_data['duration']

		# print(dataset_trans_en_test_all_duration.head(20))

		# len(dataset_trans_en_test_all_duration)

		# the complete final dataframe of all Accents txt files
		# dataset_trans_en_test_all_duration.to_csv( Data_results_path+'dataset_trans_en_test_all.csv')

		transcripts = list(set(dataset_trans_en_test_all_duration['transcript'].tolist()))
		trans_dict = {x:[] for x in transcripts}
		for index, row in dataset_trans_en_test_all_duration.iterrows():
			trans_dict[row['transcript']].append(row['audio_filepath'])

		################ Creat a En dictionary for the Audio files , Accents*
		#@title Creat a EN dictionary for the Audio files , Accents#######
		###################################################################

		# initialize an empty dictionary
		file_meta = {}

		# read the TSV file
		with open(validated_en_tsv_path_small, 'r') as f:
			# skip the header row
			next(f)
			# iterate over the remaining rows
			for line in f:
				# split the line into columns
				cols = line.strip().split('\t')
				# extract the relevant columns
				filename = cols[1].split('.')[0]
				accent = cols[7]
				transcript = cols[2]
				# create a dictionary for this file
				file_dict = {'accent': accent, 'transcript': transcript}
				# add the dictionary to the file_meta dictionary
				file_meta[filename] = file_dict
				

				# Open a new CSV file in write mode
		with open(main_path+'/DeepSpeech/data/file_meta.csv', 'w', newline='') as file:
			writer = csv.writer(file)

			# Write the header row
			writer.writerow(file_meta.keys())

			# Write the values row
			writer.writerow(file_meta.values())

		# import textgrid
		# import csv
		# import os
		# import glob

		#@title  Process "align.json" files contents and import end_times,phones to pass it to ###
				################################## file_meta dict ########################################
				##########################################################################################
				

		'''
		##########################################################################################
		##### Input : the output align.json of an Audio (From Montreal Forced Aligner Output)
		##### Output:
		##########################################################################################
		'''

		# dir_path = os.path.dirname(os.path.abspath(__file__))
		############### # in this case should the textgrid ile located at the same working file
		############### AttributionAnalysis_De_NB18042023.ipynb

		Lan ='En'

		if Lan =='De':
			folder_path = main_path+'/DeepSpeech/data/textgrid/' # replace with the actual path of your directory
		elif Lan=='En':
			folder_path = main_path+'/DeepSpeech/data/textgrid_En/' # replace with the actual path of your directory

		extension = '.TextGrid'

		# Get a list of all files with the specified extension in the folder
		files = glob.glob(os.path.join(folder_path, '*' + extension))

		# Loop through the files and print their names
		for audio_file_textgrid in files:
			# print(os.path.basename(file))
			file_name=os.path.basename(audio_file_textgrid)
		# dir_path =main_path+'/DeepSpeech/data/textgrid/'
		# file_name = "common_voice_de_30676740"
			# file_path = dir_path + "/" + file_name


			# tg = textgrid.TextGrid.fromFile(dir_path + "/" + file_name + ".TextGrid")
			tg = textgrid.TextGrid.fromFile(folder_path + "/" + file_name)
			csv_input = [[], []]


			print("------- IntervalTier Example -------")
			print(os.path.basename(audio_file_textgrid),'- export Phnes and End_times Done')
			file_name_id=os.path.splitext(os.path.basename(audio_file_textgrid))[0]
			t = tg.tiers
			for i in range (len(tg.getNames())): 
				if (tg[i].name == "phones"): 
					# print(tg[i].name)
					for j in range (len(tg[i].intervals)):
						# print(tg[i][j].mark)
						csv_input[0].append(tg[i][j].mark)
						csv_input[1].append(tg[i][j].maxTime)

				################################################################################################                
				# update the file_meta dictionary with the phone transcriptions and their corresponding end times
						file_meta[file_name_id]['phones'] = csv_input[0]
						file_meta[file_name_id]['end_times'] = csv_input[1]

			with open(folder_path + "/" + file_name_id +".csv", 'w', encoding='utf-8', newline='') as file:
				writer = csv.writer(file)
				writer.writerow(csv_input)


		
			# save the updated file_meta dictionary to a CSV file
			with open(main_path+'/PhoneProbes/file_meta_end_phones_'+Lan+'.csv', 'w', newline='') as csv_file:
		
				# Define the fieldnames of the CSV file
				fieldnames = list(file_meta.keys())

				# Create a writer object
				writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

				# Write the header row
				writer.writeheader()

				# Write the data rows
				writer.writerow(file_meta)


		############################################################

		# j_npy_num=3 # please set the number of npy files

		# for j_npy_num

		for i in tqdm(range(size)):
			sample = ids[i]
			file_name, accent_label, duration = sample[0], sample[1], sample[2]

			# phones_path=main_path+'/DeepSpeech/data/Contribution/MCV/conv/'
			# aa = glob.glob1(phones_path,"*.npy")
			# aa
			# rep_type='conv'
			# file_name='common_voice_en_533247'
			# rep_path=main_path+'/DeepSpeech/data/Contribution/MCV/' +rep_type+'/'
			# aaa=os.path.join(rep_path, file_name )
			 
			
			rep_path_plus=rep_path+'/'
			
			representation_temp=glob.glob1(rep_path_plus,'{}_*_{}.npy'.format(file_name,rep_type))
			print(representation_temp)
			for i_representation_temp in representation_temp:
					
				# print(i_representation_temp)
				
				representation = np.load(rep_path_plus+i_representation_temp)
				
				representation = torch.from_numpy(representation)
				
				times = file_meta[file_name]['end_times']
				
				#print(file_meta[file_name]['phones'])
				#print(times)
				rep_list = torch.unbind(representation, dim=0)
				accent_path = os.path.join(target_path, accent_label)
				
				if not os.path.exists(accent_path):
					os.makedirs(accent_path)
				valid_phone_list = ['ao', 'ae', 'r', 'eh', 't', 'b', 'aa', 'f', 'k', 'ng', 
			's', 'g', 'ow', 'er', 'l', 'th', 'z', 'aw', 'd', 'dh', 'sh', 'hh', 'iy', 'ch', 
			'm', 'ey', 'v', 'y', 'zh', 'jh', 'p','uw', 'ah', 'w', 'n', 'oy', 'ay', 'ih', 
			'uh']
				count_dict = dict([(key, 0) for key in valid_phone_list])
				count = 0 
				
				 
				for i in range(len(rep_list)):
					
					frame_idx = i
					if(rep_type != 'spec'):
						
						frame_idx = get_input_frame(frame_idx)
					window_start = frame_idx*spec_stride
					
					window_mid = window_start + (window_size/2)
					#print(window_start, window_mid)
					alligned_phone = 'na'
					for j in range(len(times)):
						#print(window_mid, times[j])
						if (window_mid < times[j]):
							alligned_phone = file_meta[file_name]['phones'][j]
							# print(alligned_phone)
							break
					#print(alligned_phone)
					if(alligned_phone == 'na'):
						print ("Oops error in allignment for ", file_name, "frame ",frame_idx )
						
					if(alligned_phone in valid_phone_list):
						# print('No God But Allah')
						count_dict[alligned_phone] += 1
						

						path = os.path.join(accent_path, file_name+'_'+rep_type+'_'+alligned_phone+'_'+str(count_dict[alligned_phone]))
						print('No God But Allah')
						np.save(path, rep_list[i].numpy())

			

		return
parser = argparse.ArgumentParser(description='Take command line arguments')
parser.add_argument('--csv_path',type=str, metavar='DIR',
					help='path to validation csv', default=main_path+'/DeepSpeech/data/test_1750_small.csv')
					# default=main_path+'/DeepSpeech/data/test_manifest.csv')
parser.add_argument('--file_info_path',type=str, default=main_path+'/DeepSpeech/data/file_meta_end_phones.csv')
parser.add_argument('--data_path',metavar='DIR',type=str, default=main_path+'/DeepSpeech/data/Contribution/MCV/')
parser.add_argument('--rep_type',type=str,default='rnn_4')
parser.add_argument('--target_path',type=str, metavar='DIR',default=main_path+'/PhoneProbes/ConfusionMatrices/')
args = parser.parse_args()

if __name__ == '__main__':
	data_prepare(args.csv_path,
	       args.file_info_path,
		     args.data_path, 
			 args.rep_type, 
	      args.target_path)



#####    *.npy   ####



