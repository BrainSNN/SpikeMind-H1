from transformers import Qwen2AudioEncoder,AutoProcessor,Qwen2AudioForConditionalGeneration
import torch
import torch.nn as nn
import librosa

device = torch.device('cuda')

class AudioEncoder(nn.Module):
    def __init__(self,model_path,n_embd, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.proj = nn.Sequential(nn.Linear(1280,1024),nn.GELU(),nn.Linear(1024,n_embd))

    def download(self):
        self.model = Qwen2AudioEncoder.from_pretrained(self.model_path).to(device)
        self.model.load_state_dict(torch.load('./checkpoints/qwen-audio-encoder.pth'))
        self.model.requires_grad_(True)

    def forward(self,x):
        x = self.model(x).last_hidden_state
        return self.proj(x)
    

# if __name__ == '__main__':
#     AudioEncoder = AudioEncoder('/data/models/Qwen2-Audio-7B-Instruct',n_embd=768)
#     print(AudioEncoder.proj.parameters())