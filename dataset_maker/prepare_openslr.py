from scipy.io import loadmat
import os
import mne
import pandas as pd
import pickle
import librosa
from transformers import Qwen2AudioEncoder,AutoProcessor
import numpy as np
import random
from sklearn.model_selection import train_test_split

processor = AutoProcessor.from_pretrained('/home/data/data/models/Qwen2-Audio-7B-Instruct')

root = '/home/data/data/dataset/openslr/data_thchs30/data'


files = []
for file in os.listdir(root):
    if file.endswith('.wav'):
        files.append(file)

def process_audio(path):
    data,sr = librosa.load(path)
    data = librosa.resample(data.astype('float32'),orig_sr=sr,target_sr=processor.feature_extractor.sampling_rate)
    data = processor.feature_extractor(data,sampling_rate=processor.feature_extractor.sampling_rate)
    return data['input_features'][0]

if __name__ == '__main__':
    train_files,test_files = train_test_split(files,test_size=0.1)
    save_root = '/data/NeuroLM_data/dataset/openslr'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if not os.path.exists(os.path.join(save_root,'train')):
        os.makedirs(os.path.join(save_root,'train'))
    if not os.path.exists(os.path.join(save_root,'test')):
        os.makedirs(os.path.join(save_root,'test'))
    for file in train_files:
        audio = process_audio(os.path.join(root,file))
        with open(os.path.join(root,file+'.trn')) as f:
            text = f.readlines()[0].strip().replace(' ','')
        save_path = os.path.join(save_root,'train',file.replace('.','')+'.pkl')
        pickle.dump({'X':np.random.randn(19,2000).astype('float32'),'audio':audio,'y':text},open(save_path,'wb'))
    for file in test_files:
        audio = process_audio(os.path.join(root,file))
        with open(os.path.join(root,file+'.trn')) as f:
            text = f.readlines()[0].strip().replace(' ','')
        save_path = os.path.join(save_root,'test',file.replace('.','')+'.pkl')
        pickle.dump({'X':np.random.randn(19,2000).astype('float32'),'audio':audio,'y':text},open(save_path,'wb'))