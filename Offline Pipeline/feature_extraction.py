import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
import librosa
from tqdm import tqdm
from laion_clap import CLAP_Module
import numpy.core.multiarray

# NOTE: This pipeline needs to be run separately for all three datasets (Train, Validation, and Test) to extract the required features.

validation = r"D:\00. Codes\Python\Convert music to art\validation"
validation_csv = r"D:\00. Codes\Python\Convert music to art\validation_metadata.csv"
save_dir = r"D:\00. Codes\Python\Convert music to art"

os.makedirs(save_dir, exist_ok=True)

TARGET_SAMPLE_RATE = 16000
FIXED_LENGTH = TARGET_SAMPLE_RATE * 10 
BATCH_SIZE = 8

audio_files = [f for f in os.listdir(validation) if f.lower().endswith(('.mp3', '.wav'))]
df = pd.read_csv(validation_csv)

df = df[df['ytid'].apply(lambda x: f"{x}.wav" in audio_files)].copy()
df['audio_path'] = df['ytid'].apply(lambda x: os.path.join(validation, f"{x}.wav"))
final_df = df[['audio_path', 'caption']].reset_index(drop=True)

torch.serialization.add_safe_globals([
    numpy.core.multiarray.scalar,
    numpy.dtype,
    numpy.dtype('float64').__class__
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLAP_Module(enable_fusion=False)
model.load_ckpt()
model.to(device)
model.eval()

def load_wav(path, target_sr=TARGET_SAMPLE_RATE):
    waveform, sr = torchaudio.load(path)
    if sr != target_sr:
        resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resample(waveform)
    return waveform[0]

def pad_or_trim(wav, max_len=FIXED_LENGTH):
    if wav.size(0) > max_len:
        return wav[:max_len]
    elif wav.size(0) < max_len:
        return F.pad(wav, (0, max_len - wav.size(0)))
    else:
        return wav

def process_and_save_embeddings(df, model, device, batch_size=BATCH_SIZE, save_dir=save_dir):
    all_embeddings = []
    all_names = []
    collected_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset (CLAP)"):
        try:
            audio_path = row['audio_path']
            name = os.path.basename(audio_path)

            wav = load_wav(audio_path)
            wav = pad_or_trim(wav)

            all_names.append(name)
            all_embeddings.append(wav)

            if len(all_embeddings) == batch_size or idx == len(df) - 1:
                with torch.no_grad():
                    inputs = torch.stack(all_embeddings)
                    inputs_np = inputs.numpy()
                    embs = model.get_audio_embedding_from_data(x=inputs_np)

                    if isinstance(embs, torch.Tensor):
                        embs = embs.detach().cpu().numpy()

                for i in range(len(embs)):
                    collected_rows.append({
                        "name": all_names[i],
                        "embedding": embs[i].tolist()
                    })

                all_embeddings = []
                all_names = []

        except Exception as e:
            print(f"⚠️ Error processing {audio_path}: {e}")

    df_output = pd.DataFrame(collected_rows)
    emb_path = os.path.join(save_dir, "audio_embeddings.parquet")
    df_output.to_parquet(emb_path, index=False)
    print(f"✅ Saved embeddings to {emb_path} with {len(df_output)} rows.")

process_and_save_embeddings(final_df, model, device)

parquet_path = os.path.join(save_dir, "audio_embeddings.parquet")
df_emb = pd.read_parquet(parquet_path)
df_caption = pd.read_csv(validation_csv)

df_emb['ytid'] = df_emb['name'].str.replace(r'\.wav$', '', regex=True).str.replace(r'\.mp3$', '', regex=True)
df_merged_captions = df_emb.merge(df_caption[['ytid', 'caption']], on='ytid', how='inner')
df_merged_captions = df_merged_captions.drop(columns=['ytid'])

embeddings_with_captions_path = os.path.join(save_dir, "embeddings_with_captions.parquet")
df_merged_captions.to_parquet(embeddings_with_captions_path, index=False)
print(f"✅ new file saved: {embeddings_with_captions_path}")

def initialize_feature_file(audio_folder, output_path):
    audio_files = [f for f in os.listdir(audio_folder) if f.lower().endswith(('.wav', '.mp3'))]
    df = pd.DataFrame({'name': audio_files})
    df.to_parquet(output_path, index=False)
    print(f"✅ Initialized empty feature file at: {output_path} with {len(df)} names.")

audio_features_path = os.path.join(save_dir, "audio_features.parquet")
initialize_feature_file(validation, audio_features_path)

def extract_and_add_feature(parquet_path, audio_folder, feature_name, feature_func, batch_size=50):
    df = pd.read_parquet(parquet_path)
    if 'name' not in df.columns:
        raise ValueError("File must contain a 'name' column.")

    new_values = []
    names = df['name'].tolist()

    for i in tqdm(range(0, len(names), batch_size), desc=f"Extracting {feature_name}"):
        batch_names = names[i:i+batch_size]
        batch_vals = []

        for name in batch_names:
            path = os.path.join(audio_folder, name)
            try:
                y, sr = librosa.load(path, sr=16000)
                val = feature_func(y, sr)
            except Exception as e:
                print(f"⚠️ Error processing {name} for {feature_name}: {e}")
                val = np.nan
            batch_vals.append(val)

        new_values.extend(batch_vals)

    df[feature_name] = new_values
    df.to_parquet(parquet_path, index=False)
    print(f"✅ Added feature '{feature_name}' to {parquet_path}")

def get_spectral_centroid(y, sr):
    return np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

def get_spectral_bandwidth(y, sr):
    return np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

def get_zero_crossing_rate(y, sr):
    return np.mean(librosa.feature.zero_crossing_rate(y))

def get_rms(y, sr):
    return np.mean(librosa.feature.rms(y=y))

def get_tempo(y, sr):
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if hasattr(tempo, '__len__'):
            tempo = np.mean(tempo)
        return float(tempo)
    except Exception as e:
        print(f"Error extracting tempo: {e}")
        return np.nan

extract_and_add_feature(audio_features_path, validation, "spectral_centroid", get_spectral_centroid)
extract_and_add_feature(audio_features_path, validation, "spectral_bandwidth", get_spectral_bandwidth)
extract_and_add_feature(audio_features_path, validation, "zero_crossing_rate", get_zero_crossing_rate)
extract_and_add_feature(audio_features_path, validation, "rms", get_rms)
extract_and_add_feature(audio_features_path, validation, "tempo", get_tempo)


output_path = os.path.join(save_dir, "complete_audio_features_validation.parquet")

df_audio = pd.read_parquet(audio_features_path)
df_embed = pd.read_parquet(embeddings_path=embeddings_with_captions_path)

df_merged_final = pd.merge(df_audio, df_embed, on='name', how='inner')
df_merged_final.to_parquet(output_path, index=False)

print("\nDONE! Two files merged perfectly into:")
print(f"{output_path}")