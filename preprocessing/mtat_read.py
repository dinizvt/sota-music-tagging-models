import warnings
import os
import numpy as np
import glob
import librosa
import fire
import tqdm
import threading

class Processor:
	def __init__(self):
		self.fs = 16000

	def get_paths(self, data_path):
		self.files = glob.glob(os.path.join(data_path, 'mtat', 'mp3', '*/*.mp3'))
		self.npy_path = os.path.join(data_path, 'mtat', 'npy')
		if not os.path.exists(self.npy_path):
			os.makedirs(self.npy_path)

	def get_npy(self, fn):
		x, sr = librosa.core.load(fn, sr=self.fs)
		return x

	def thread_routine (self, files, pbar):
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			for fn in files:
				pbar.update()
				npy_fn = os.path.join(self.npy_path, fn.split('/')[-1][:-3]+'npy')
				if not os.path.exists(npy_fn):
					try:
						x = self.get_npy(fn)
						np.save(open(npy_fn, 'wb'), x)
					except RuntimeError:
						# some audio files are broken
						print(fn)
						continue

	def iterate(self, data_path):
		self.get_paths(data_path)
		hw = len(self.files)//2
		pbar = tqdm.tqdm(total=len(self.files))
		left = threading.Thread(target=self.thread_routine, args = (self.files[hw:], pbar))
		right = threading.Thread(target=self.thread_routine, args = (self.files[:hw], pbar))
		left.start()
		right.start()
		left.join()
		right.join()
		
if __name__ == '__main__':
	p = Processor()
	fire.Fire({'run': p.iterate})
