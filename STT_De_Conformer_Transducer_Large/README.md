# STT De Conformer-Transducer Large
we use the End-to-end ASR: STT De Conformer-Transducer Large} from Nvidia 
```
https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_de_conformer_transducer_large
```


This collection contains large size versions of Conformer-Transducer (around 120M parameters) trained on German NeMo ASRSet with over 2000 hours of speech. The model transcribes speech in lower case German alphabet along with spaces but without punctuation. This model was trained with the English model as initialization.

## Model Architecture

Conformer-Transducer model is an autoregressive variant of Conformer model [1] for Automatic Speech Recognition which uses Transducer loss/decoding. You may find more info on the detail of this model here: Conformer-Transducer Model.

## Training
The NeMo toolkit [3] was used for training the models. These models are fine-tuned with this example script and this base config.

The tokenizers for these models were built using the text transcripts of the train set with this script.

## Datasets

All the models in this collection are trained on a composite dataset (NeMo ASRSET) comprising of over two thousand hours of cleaned German speech:

MCV7.0 567 hours MLS 1524 hours VoxPopuli 214 hours
