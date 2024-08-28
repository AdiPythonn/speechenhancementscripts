from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
import torchaudio
import pathlib
import os
import time
import csv
import pydub
def get_audio_duration(audio_path):
    audio, sr = torchaudio.load(audio_path)
    return audio.shape[1] / sr

start_time = time.time()
#model creation
model, df_state, _ = init_df()
model_load_time = time.time() - start_time
print(f"Model loaded in {model_load_time:.2f} seconds")

audiopath = "/Users/aditi.joshi/Downloads/Noisy"
#change for other scripts
audiopath__save = "/Users/aditi.joshi/Downloads/Noisy/deepfilternet"
csv_path = "/Users/aditi.joshi/Downloads/deepfilternet.csv"
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
        audio_data, _ = load_audio(audio, sr=df_state.sr())
        # Denoise the audio
        enhanced = enhance(model, df_state, audio_data) 

        processing_time = time.time() - start_time
        total_processing_time += processing_time
        total_duration += duration
        print(f"Processing time: {processing_time:.2f} seconds")
    
    # Write to CSV
        csvwriter.writerow([audio, f"{duration:.2f}", f"{processing_time:.2f}"])

        save_audio(audio__save, enhanced, df_state.sr())
    print(f"\nTotal audio duration: {total_duration:.2f} seconds")
    print(f"Total processing time: {total_processing_time:.2f} seconds")
    print(f"Average processing time per second of audio: {total_processing_time/total_duration:.2f} seconds")
 
    # Append summary to CSV
    
    csvwriter.writerow([])
    csvwriter.writerow(['Total Duration (s)', 'Total Processing Time (s)', 'Avg. Processing Time per Second of Audio (s)'])
    csvwriter.writerow([f"{total_duration:.2f}", f"{total_processing_time:.2f}", f"{total_processing_time/total_duration:.2f}"])
 
    print(f"Results saved to {csv_path}")