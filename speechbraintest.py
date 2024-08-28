from speechbrain.inference.separation import SepformerSeparation as separator 
import torchaudio
import pathlib
import os 
model = separator.from_hparams(source="speechbrain/rescuespeech_sepformer", savedir='pretrained_models/rescuespeech_sepformer') 
audiopath = "/Users/aditi.joshi/Downloads/Noisy"
audiopath__save = "/Users/aditi.joshi/Downloads/Noisy/speechbrain1"
os.makedirs(audiopath__save,exist_ok=True)
# for custom file, change path 
for file in pathlib.Path(audiopath).iterdir():
    if not file.is_file():  
        continue
    audio = os.path.join(audiopath,file.name)
    audio__save = os.path.join(audiopath__save,"enhanced_"+file.name)
    print(audio)
    est_sources = model.separate_file(path=audio) 
    # print(est_sources)
    torchaudio.save(audio__save, est_sources[:, :, 0].detach().cpu(), 16000)