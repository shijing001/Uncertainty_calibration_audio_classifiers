import librosa
import argparse
import pandas as pd
import numpy as np
import pickle as pkl 
import torch
import torchaudio
import torchvision
from PIL import Image
import pdb
from sklearn.utils import shuffle as reset
parser = argparse.ArgumentParser()
parser.add_argument("--csv_file", type=str)
parser.add_argument("--data_dir", type=str)
parser.add_argument("--store_dir", type=str)
parser.add_argument("--sampling_rate", default=44100, type=int)
parser.add_argument("--seed", default=1, type=int)

def extract_spectrogram(values, clip, entries):
 for data in entries:

  num_channels = 3
  window_sizes = [25, 50, 100]
  hop_sizes = [10, 25, 50]
  centre_sec = 2.5

  specs = []
  for i in range(num_channels):
   window_length = int(round(window_sizes[i]*args.sampling_rate/1000))
   hop_length = int(round(hop_sizes[i]*args.sampling_rate/1000))

   clip = torch.Tensor(clip)
   spec = torchaudio.transforms.MelSpectrogram(sample_rate=args.sampling_rate, n_fft=4410, win_length=window_length, hop_length=hop_length, n_mels=128)(clip)
   eps = 1e-6
   spec = spec.numpy()
   spec = np.log(spec+ eps)
   spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
   specs.append(spec)
  new_entry = {}
  new_entry["audio"] = clip.numpy()
  new_entry["values"] = np.array(specs)
  new_entry["target"] = data["target"]
  values.append(new_entry)

def extract_features(audios):
 audio_names = list(audios.filename.unique())
 values = []
 for audio in audio_names:
  clip, sr = librosa.load("{}/{}".format(args.data_dir, audio), sr=args.sampling_rate)
  entries = audios.loc[audios["filename"]==audio].to_dict(orient="records")
  extract_spectrogram(values, clip, entries)
  print("Finished audio {}".format(audio))
 return values

if __name__=="__main__":
 args = parser.parse_args()
 audios = pd.read_csv(args.csv_file, skipinitialspace=True)
 #split the dataset
 training_audios = audios.sample(frac=0.8)
 validation_test = audios[~audios.index.isin(training_audios.index)]
 validation_audios = validation_test.sample(frac=0.5)
 test_audios = validation_test[~validation_test.index.isin(validation_audios.index)]

 training_audios = training_audios.reset_index(drop=True)
 validation_audios = validation_audios.reset_index(drop=True)
 test_audios = test_audios.reset_index(drop=True)

 # audios_value = extract_features(audios)

 # training_audios, validation_audios, test_audios = torch.utils.data.random_split(
    #     audios, (train_size, valid_size, test_size), generator=torch.Generator().manual_seed(args.seed))
 
 # pdb.set_trace()
 # num_folds = 5

 # for i in range(1, num_folds+1):
 #  training_audios = audios.loc[audios["fold"]!=i]
 #  validation_audios = audios.loc[audios["fold"]==i]

 training_values = extract_features(training_audios)
 with open("{}training128mel.pkl".format(args.store_dir),"wb") as handler:
  pkl.dump(training_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

 validation_values = extract_features(validation_audios)
 with open("{}validation128mel.pkl".format(args.store_dir),"wb") as handler:
  pkl.dump(validation_values, handler, protocol=pkl.HIGHEST_PROTOCOL)

 test_values = extract_features(test_audios)
 with open("{}test128mel.pkl".format(args.store_dir),"wb") as handler:
  pkl.dump(test_values, handler, protocol=pkl.HIGHEST_PROTOCOL)
