import sys
import os
import subprocess
import shutil
sys.path.insert(0, "../../xnmt")  # Importing xnmt into the Pytohn Path

from input import MultilingualAlignedCorpusReader

data_path = "/home/devendra/Desktop/Neural_MT/scrapped_ted_talks_dataset/web_data_temp"
temp_dir_path = "../../ted_sample_multiway"

# Specify the training languages for Zero Shot translation.
# Currently specifying the languages in Google Multilingual Paper
mw_train_lang_dict = {'source': ['fr', 'en', 'de'], 'target': ['fr', 'en', 'de']}

# Evaluation language dict is None here
mw_eval_lang_dict = None

zero_shot = False
bilingual = True
tokenization = True
vocab_size = '32000'

# Remove the directory if it exists and create a new temp directory
if os.path.exists(temp_dir_path):
    shutil.rmtree(temp_dir_path)
os.mkdir(temp_dir_path)

# Assign the file names with splits
train_source = os.path.join(temp_dir_path, "mw_s.train")
train_target = os.path.join(temp_dir_path, "mw_t.train")
test_source = os.path.join(temp_dir_path, "mw_s.test")
test_target = os.path.join(temp_dir_path, "mw_t.test")
dev_source = os.path.join(temp_dir_path, "mw_s.dev")
dev_target = os.path.join(temp_dir_path, "mw_t.dev")

print("Reading the TED talks corpus")

obj = MultilingualAlignedCorpusReader(corpus_path=data_path, lang_dict=mw_train_lang_dict, target_token=True,
                                      eval_lang_dict=mw_eval_lang_dict, zero_shot=zero_shot, bilingual=bilingual)

# Save the source and target files from train, test and dev splits
obj.save_file(train_source, split_type='train', data_type='source')
obj.save_file(train_target, split_type='train', data_type='target')

obj.save_file(test_source, split_type='test', data_type='source')
obj.save_file(test_target, split_type='test', data_type='target')

obj.save_file(dev_source, split_type='dev', data_type='source')
obj.save_file(dev_target, split_type='dev', data_type='target')


if tokenization:
    print("Training using sentence piece model")
    # Training using sentence piece model
    custom_target_lang_list = []
    for lang in mw_train_lang_dict['target']:
        custom_target_lang_list.append("__{}__".format(lang))

    custom_target_lang_string = ",".join(custom_target_lang_list)

    print(custom_target_lang_string)
    # This sentence piece model will name the encoded files using `*.tok` extension in the data directory
    subprocess.call(['bash', 'sentencepiece.sh', temp_dir_path, train_source, train_target, test_source, test_target,
                    dev_source, dev_target, vocab_size, custom_target_lang_string], shell=False)

# Now modify the corpus path in the config.yaml file accordingly