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
    
    def make_split (self, tags_path):
        tags = pd.read_parquet(tags_path)
        tags = tags.apply(lambda x: np.concatenate([i for i in x if i is not None]), axis=1)
        tags = tags.reset_index().drop_duplicates('index').reset_index(drop=True)
        top_50 = np.load('../split/4mula/tags.npy')

        print('getting ids on dataset...')
        ids = [i.split('.')[0] for i in glob.glob(os.path.join(self.npy_path, '*.npy'))]
        tags = tags.query('index in @ids')
        print('making binary.npy file...')
        binary = tags[0].map(lambda x: [int(i in x) for i in top_50])
        binary = np.stack(binary).astype('float64')
        print('splitting dataset...')
        idx_shuff = tags.index.values.copy()
        idx_shuff = np.random.shuffle(idx_shuff)
        train,test,val = np.split(idx_shuff, [int(len(ids)*0.6), int(len(ids)*0.8)])
        print('saving files...')
        np.save('../split/4mula/binary.npy', binary)
        np.save('../split/4mula/train.npy', tags.query('index in @train').apply(lambda x: f'{x.name}\t{x.values[0]}', axis=1).values.astype('U30'))
        np.save('../split/4mula/test.npy', tags.query('index in @test').apply(lambda x: f'{x.name}\t{x.values[0]}', axis=1).values.astype('U30'))
        np.save('../split/4mula/valid.npy', tags.query('index in @val').apply(lambda x: f'{x.name}\t{x.values[0]}', axis=1).values.astype('U30'))

    def iterate(self, data_path, output_path):
        self.get_paths(data_path, output_path)
        for fn in tqdm.tqdm(self.files):
            df = pd.read_parquet(fn)
            df.apply(self.get_npy, axis=1)
            
if __name__ == '__main__':
	p = Processor()
	fire.Fire({
        'run': p.iterate,
        'split': p.make_split
    })