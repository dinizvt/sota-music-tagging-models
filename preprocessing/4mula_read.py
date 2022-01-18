import os
import numpy as np
import glob
import librosa
import fire
import tqdm
import pandas as pd

class Processor:
    def __init__ (self):
        pass

    def get_paths (self, data_path, output_path):
        self.files = glob.glob(os.path.join(data_path, '*.parquet'))
        self.npy_path = os.path.join(output_path, 'mel_npy')
        if not os.path.exists(self.npy_path):
            os.makedirs(self.npy_path)
    
    def get_npy (self, row):
        filename = row['music_id'] + '.npy'
        if not os.path.exists(os.path.join(self.npy_path, filename)):
            np.save(os.path.join(self.npy_path, filename), np.stack(row['melspectrogram']).astype('float64'))
        return os.path.join(self.npy_path, filename)
    
    def iterate(self, data_path):
        self.get_paths(data_path)
        for fn in tqdm.tqdm(self.files):
            df = pd.read_parquet(fn)
            df.apply(self.get_npy, axis=1)
            
if __name__ == '__main__':
	p = Processor()
	fire.Fire({'run': p.iterate})
