from scipy.io import loadmat
import os
import mne
import pandas as pd
import pickle
import librosa
from transformers import Qwen2AudioEncoder,AutoProcessor
import numpy as np
import random

processor = AutoProcessor.from_pretrained('/home/data/data/models/Qwen2-Audio-7B-Instruct')

MODMA_chaninfo = {22:'FP1',9:'FP2',33:'F7',24:'F3',124:'F4',122:'F8',36:'C3',104:'C4',45:'T3',108:'T4',52:'P3',92:'P4',62:'PZ',11:'FZ',58:'T5',96:'T6',70:'O1',75:'OZ',83:'O2'}
ch_idx = list(MODMA_chaninfo.keys())
ch_names = [MODMA_chaninfo[i] for i in ch_idx]

def preprocessing(raw, l_freq=0.5, h_freq=75, sfreq:int=200):
    # 读取cnt
    # 滤波
    raw = raw.notch_filter(50.0)
    raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
    # 降采样
    raw = raw.resample(sfreq, n_jobs=5)
    eegData = raw.get_data(units='mV')

    return eegData

def process_audio(path):
    audio = []
    for file in os.listdir(path):
        data,sr = librosa.load(os.path.join(path,file))
        audio.append(librosa.resample(data.astype('float32'),orig_sr=sr,target_sr=processor.feature_extractor.sampling_rate))
    data = np.concat(audio)
    # data = processor.feature_extractor(data)
    return data#['input_features'][0]

if __name__ == '__main__':
    root = '/home/data/data/dataset/MODMA_EEG/EEG_RET/EEG_128channels_resting_lanzhou_2015/'
    audio_root = '/home/data/data/dataset/MODMA/audio_lanzhou_16khz/'
    info = pd.read_excel(os.path.join(root,'subjects_information_EEG_128channels_resting_lanzhou_2015.xlsx'))
    save_root = '/home/data/data/NeuroLM_data/dataset/MODMA'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if not os.path.exists(os.path.join(save_root,'train')):
        os.makedirs(os.path.join(save_root,'train'))
    if not os.path.exists(os.path.join(save_root,'val')):
        os.makedirs(os.path.join(save_root,'val'))
    if not os.path.exists(os.path.join(save_root,'test')):
        os.makedirs(os.path.join(save_root,'test'))
    l = []
    label_proj = {'HC':0,'MDD':1}
    for n,file in enumerate(os.listdir(root)):
        if file.endswith('mat'):
            data = loadmat(os.path.join(root,file))
            key = max(data.keys(),key=len)
            data = data[key]
            data = data[ch_idx]
            data = data.T
            data = (data-data.mean(0))/data.std()
            data = data.T
            subject = file[0:8]
            try:
                audio = process_audio(os.path.join(audio_root,subject))
            except:
                continue
            label = info[info.iloc[:,0]==int(file[1:8])]['type'].values[0]
            raw_info = mne.create_info(ch_names=ch_names,ch_types=['eeg']*len(ch_names),sfreq=250)
            raw = mne.io.RawArray(data,info=raw_info)
            data = preprocessing(raw)
            for i in range(30):
                if i == 0:
                    continue
                if n<4:
                    sub_path = 'val'
                elif n<9:
                    sub_path = 'test'
                else:
                    sub_path = 'train'
                save_path = os.path.join(save_root,sub_path,key+str(i)+'.pkl')
                begin_id = random.randint(0,audio.shape[0]-processor.feature_extractor.sampling_rate*30)
                audio_feature = processor.feature_extractor(audio[begin_id:begin_id+processor.feature_extractor.sampling_rate*30],sampling_rate=processor.feature_extractor.sampling_rate)
                audio_feature = audio_feature['input_features'][0]
                pickle.dump({'X':data[:,i*2000:(i+1)*2000],'audio':audio_feature,'y':label_proj[label]},open(save_path,'wb'))

