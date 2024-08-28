from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
import pathlib
import os
import time
import csv
import pydub
from resemble_enhance.enhancer.inference import denoise, enhance
import numpy as np


device = "cpu"
def _fn(path, solver = "Midpoint", nfe = 64, tau = 0.5, denoising = False):
    if path is None:
        return None, None

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load(path)
    dwav = dwav.mean(dim=0)

    wav1, new_sr = denoise(dwav, sr, device)
    wav2, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=lambd, tau=tau)

    wav1 = wav1.cpu().numpy()
    wav2 = wav2.cpu().numpy()
    print (wav2.shape)
    return (new_sr, wav1), (new_sr, wav2)

def get_audio_duration(audio_path):
    audio, sr = torchaudio.load(audio_path)
    return audio.shape[1] / sr

start_time = time.time()
#model creation
model = separator.from_hparams(source="speechbrain/sepformer-dns4-16k-enhancement", savedir='pretrained_models/sepformer-dns4-16k-enhancement')
model_load_time = time.time() - start_time
print(f"Model loaded in {model_load_time:.2f} seconds")

audiopath = "/Users/aditi.joshi/Downloads/Noisy"
#change for other scripts
audiopath__save = "/Users/aditi.joshi/Downloads/Noisy/resemble_enhance"
csv_path = "/Users/aditi.joshi/Downloads/resemble_enhance.csv"
total_processing_time = 0.0
total_duration = 0.0
os.makedirs(audiopath__save,exist_ok=True)
# for custom file, change path
# Prepare CSV file
with open(csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['File', 'Duration (s)', 'Processing Time (s)'])
    for file in pathlib.Path(audiopath).iterdir():
        if not file.is_file():  
            continue
        if ".DS_Store" in file.name:
            continue
        audio = os.path.join(audiopath,file.name)
    
        duration = get_audio_duration(audio)
 
        audio__save = os.path.join(audiopath__save,"enhanced_"+file.name)
        print(audio)
        print(duration)
        start_time = time.time()
        #model usage
        #est_sources = model.separate_file(path=audio)[:,:,0].detach().cpu()
        est_sources = np.ndarray([_fn(audio)[1][1]])
        processing_time = time.time() - start_time
        total_processing_time += processing_time
        total_duration += duration
        print(f"Processing time: {processing_time:.2f} seconds")
    
    # Write to CSV
        csvwriter.writerow([audio, f"{duration:.2f}", f"{processing_time:.2f}"])

        torchaudio.save(audio__save, est_sources, 16000)
    print(f"\nTotal audio duration: {total_duration:.2f} seconds")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    print(f"Average processing time per second of audio: {total_processing_time/total_duration:.2f} seconds")
 
    # Append summary to CSV
    
    csvwriter.writerow([])
    csvwriter.writerow(['Total Duration (s)', 'Total Processing Time (s)', 'Avg. Processing Time per Second of Audio (s)'])
    csvwriter.writerow([f"{total_duration:.2f}", f"{total_processing_time:.2f}", f"{total_processing_time/total_duration:.2f}"])
 
    print(f"Results saved to {csv_path}")