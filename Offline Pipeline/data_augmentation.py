import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

input_dir = r"D:\00. Codes\Python\Convert music to art\train"  
output_dir = r"D:\00. Codes\Python\Convert music to art\train_aug"  
os.makedirs(output_dir, exist_ok=True)

def augment_audio(y, sr):
    # Pitch shift
    pitch_up = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=2)
    pitch_down = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=-2)

    
    # Time stretch
    stretch_fast = librosa.effects.time_stretch(y, rate=1.1)
    stretch_slow = librosa.effects.time_stretch(y, rate=0.9)
    
    # Add noise
    noise = np.random.normal(0, 0.005, y.shape)
    y_noise = y + noise
    
    return {
        "pitch_up": pitch_up,
        "pitch_down": pitch_down,
        "stretch_fast": stretch_fast,
        "stretch_slow": stretch_slow,
        "noise": y_noise
    }

all_aug_files = []

for file in tqdm(os.listdir(input_dir)):
    if not file.lower().endswith(".wav"):
        continue
    
    path = os.path.join(input_dir, file)
    try:
        y, sr = librosa.load(path, sr=16000)
        aug_versions = augment_audio(y, sr)
        
        for aug_name, aug_data in aug_versions.items():
            new_name = f"{os.path.splitext(file)[0]}_{aug_name}.wav"
            out_path = os.path.join(output_dir, new_name)
            sf.write(out_path, aug_data, sr)
            all_aug_files.append(new_name)
    
    except Exception as e:
        print(f"Error processing {file}: {e}")

print(f"Created {len(all_aug_files)} augmented files in {output_dir}")

import pandas as pd
df_aug = pd.DataFrame({"name": all_aug_files})
aug_parquet_path = r"D:\00. Codes\Python\Convert music to art\audio_augmented_files.parquet"
df_aug.to_parquet(aug_parquet_path, index=False)
print(f"Saved augmented file list to {aug_parquet_path}")
