# Udacity Natural Language Processing Nanodegree

## Speech Recognition with Neural Networks

![ASR Pipeline](./images/pipeline.png)

This project develops a recurrent neural network that functions as part of an end-to-end (A)utomatic (S)peech (R)ecognition pipeline. It converts raw audio from [LibriSpeech](https://www.openslr.org/12) data into feature representations and uses them to generate transcribed text.

### Requirements

1. Download and install [Git](https://git-scm.com)
2. Download and install [Anaconda](https://www.anaconda.com)
3. Download and install [FFmpeg](https://ffmpeg.org)

### Data Folders

- data
  - LibriSpeech
    - [dev-clean](https://www.openslr.org/resources/12/dev-clean.tar.gz)
    - [test-clean](https://www.openslr.org/resources/12/test-clean.tar.gz)

### Set-up

Clone the project repository
```
git clone https://github.com/sdonatti/nd892-project-dnn-speech-recognizer
```

Install required Python packages
```
cd nd892-project-dnn-speech-recognizer
conda env create -f environment.yaml
conda activate nd892-project-dnn-speech-recognizer
```

Define the datasets
```
python flac_to_wav.py data/LibriSpeech/dev-clean
python flac_to_wav.py data/LibriSpeech/test-clean
python create_desc_json.py data/LibriSpeech/dev-clean train_corpus.json
python create_desc_json.py data/LibriSpeech/test-clean valid_corpus.json
```

Launch the project Jupyter Notebooks
```
jupyter notebook
```

### License

This project is licensed under the [MIT License](LICENSE)