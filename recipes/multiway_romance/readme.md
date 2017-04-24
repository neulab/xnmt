Steps to create train, test and dev data from TED talks and to tokenize them using sentencepiece toolkit from google.

* Install [sentencepiece](https://github.com/google/sentencepiece) library according to instructions mentioned in here (https://github.com/google/sentencepiece)
* In the file `create_exp_data.py` specify the path of the TED Talks corpus. The location of the sentence aligned corpus in TIR cluster is `/projects/tir2/users/dsachan/ted_scrapped_data/web_data_sent_aligned`
* Specify a data directory location to be used for saving the new experiment specific data files.
* Specify the **source** and **target** language list. In case of Zero Shot experiments, also specify the source and target evaluation languages.
* Set the bool values of `zero_shot` and `bilingual` flags accordingly.
* If `tokenization = True`, then `sentencepiece.sh` script is called which trains the sentencepiece model. Once can also give the option for BPE (Byte Pair Encoding)
* By default the vocab size is `32000` for tokenization, but can be changed from the parameters.
* This sentence piece model will name the encoded files using `*.tok` extension in the data directory. Also, it will save the `sentencepiece` model in the current directory which can be used to decode the sentences to original form.
* Now, one can update the `experiments-config.yaml` config file accordingly.    