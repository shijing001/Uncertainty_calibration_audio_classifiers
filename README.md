# Uncertainty Calibration for Deep Audio Classifiers

This repository contains the PyTorch code for our paper [Uncertainty Calibration for Deep Audio Classifiers] accepted by INTERSPEECH2022. The experiments are conducted on the following two datasets which can be downloaded from the links provided:
1. [ESC-50](https://github.com/karolpiczak/ESC-50)
2. [GTZAN](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

### Preprocessing

The preprocessing is done separately to save time during the training of the models.

For ESC-50: 
```console
python preprocessing/preprocessingESC.py --csv_file /path/to/file.csv --data_dir /path/to/audio_data/ --store_dir /path/to/store_spectrograms/ --sampling_rate 44100
```

For GTZAN:
```console
python preprocessing/preprocessingGTZAN.py --data_dir /path/to/audio_data/ --store_dir /path/to/store_spectrograms/ --sampling_rate 22050
```

### Training the Models using Base and Dropout

The configurations for training the models are provided in the config folder. The sample_config.json explains the details of all the variables in the configurations. The command for training is: 
```console
python train.py --config_path /config/your_config.json
```

### Training the Models using focal loss
```console
python train_with_focal.py --config_path /config/your_config.json
```

### Training the Models using SNGP
```console
python train_with_sngp.py --config_path /config/your_config.json
```

### References
1. Our paper accepted by InerSpeech 2022, now available on ArXiv.com, https://arxiv.org/abs/2206.13071
2.https://github.com/kimjeyoung/SNGP-BERT-Pytorch
3.https://github.com/kamalesh0406/Audio-Classification
4.Rethinking CNN Models for Audio Classification. (https://arxiv.org/abs/2007.11154)
