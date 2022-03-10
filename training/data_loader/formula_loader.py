# coding: utf-8
import os
import numpy as np
import pandas as pd
from torch.utils import data
import dask.dataframe as dd

class AudioFolder(data.Dataset):
	def __init__(self, root, split, input_length=None):
		self.df = dd.read_parquet(root)
		self.id_map = self.df['music_id'].reset_index().compute().set_index('music_id')
		self.split = split
		self.input_length = input_length
		self.get_songlist()
		self.binary = np.load('./../split/4mula/binary.npy')

	def __getitem__(self, index):
		"""returns raw file

		Args:
			index (int): audio index from songlist

		Returns:
			np.ndarray: melspectrogram
		"""
		npy, tag_binary = self.get_npy(index)
		return npy.astype('float32'), tag_binary.astype('float32')

	def get_songlist(self):
		if self.split == 'TRAIN':
			self.fl = np.load('./../split/4mula/train.npy')
		elif self.split == 'VALID':
			self.fl = np.load('./../split/4mula/valid.npy')
		elif self.split == 'TEST':
			self.fl = np.load('./../split/4mula/test.npy')
		else:
			print('Split should be one of [TRAIN, VALID, TEST]')

	def get_npy(self, index):
		ix, fn = self.fl[index].split('\t')
		real_idx = self.id_map.loc[fn]['index']
		if isinstance(real_idx, pd.Series):
			real_idx = real_idx.values[0]
		print(real_idx)
		npy = self.df.loc[real_idx]['melspectrogram'].compute().values[0]
		npy = np.stack(npy).T
		random_idx = int(np.floor(np.random.random(1) * (len(npy)-self.input_length)))
		npy = np.array(npy[random_idx:random_idx+self.input_length])
		tag_binary = self.binary[int(ix)]
		return npy.T, tag_binary

	def __len__(self):
		return len(self.fl)

#class AudioFolder(data.Dataset):
#	def __init__(self, root, split, input_length=None):
#		self.root = root
#		self.split = split
#		self.input_length = input_length
#		self.get_songlist()
#		self.binary = np.load('./../split/4mula/binary.npy')
#
#	def __getitem__(self, index):
#		"""returns raw file
#
#		Args:
#			index (int): audio index from songlist
#
#		Returns:
#			np.ndarray: melspectrogram
#		"""
#		npy, tag_binary = self.get_npy(index)
#		return npy.astype('float32'), tag_binary.astype('float32')
#
#	def get_songlist(self):
#		if self.split == 'TRAIN':
#			self.fl = np.load('./../split/4mula/train.npy')
#		elif self.split == 'VALID':
#			self.fl = np.load('./../split/4mula/valid.npy')
#		elif self.split == 'TEST':
#			self.fl = np.load('./../split/4mula/test.npy')
#		else:
#			print('Split should be one of [TRAIN, VALID, TEST]')
#
#	def get_npy(self, index):
#		ix, fn = self.fl[index].split('\t')
#		npy_path = os.path.join(self.root, 'mel_npy', fn) + '.npy'
#		npy = np.load(npy_path, mmap_mode='r').T
#		
#		random_idx = int(np.floor(np.random.random(1) * (len(npy)-self.input_length)))
#		npy = np.array(npy[random_idx:random_idx+self.input_length])
#		tag_binary = self.binary[int(ix)]
#		return npy.T, tag_binary
#
#	def __len__(self):
#		return len(self.fl)


def get_audio_loader(root, batch_size, split='TRAIN', num_workers=0, input_length=None):
	data_loader = data.DataLoader(dataset=AudioFolder(root, split=split, input_length=input_length),
								  batch_size=batch_size,
								  shuffle=True,
								  drop_last=False,
								  num_workers=num_workers)
	return data_loader

