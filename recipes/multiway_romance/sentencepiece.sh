OUTPUT_DIR=$1
train_source=$2
train_target=$3
test_source=$4
test_target=$5
dev_source=$6
dev_target=$7
vocab_size=$8
custom_target_lang_string=$9

# @IgnoreInspection BashAddShebang
# OUTPUT_DIR="/home/devendra/Desktop/Neural_MT/src/xnmt/ted_sample"

# Can use the Moses Tokenizer if needed

# Clone Moses
# if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
#  echo "Cloning moses for data processing"
#  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
# fi

# Using the default unigram version of sentence piece model
# By default, it does tokenization using whitespace
echo "--- tokenize using Sentence Piece"
spm_train --input=${train_source},${train_target},${dev_source},${dev_target} \
--vocab_size=${vocab_size} --model_prefix="m_sp" --user_defined_symbols=${custom_target_lang_string} --add_dummy_prefix=false # --control_symbols=__en__,__es__


# Encoding all the data files using sentence piece encoder
for f in ${OUTPUT_DIR}/*.*   #*-????.??
do
    echo ${f}
    spm_encode --model="m_sp.model" <  ${f} > ${f}".tok"
done
