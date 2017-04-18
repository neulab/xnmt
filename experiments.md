# Running experiments with `xnmt-run-experiments`

Configuration files are in YAML dictionary format (see `test/experiments-config.yaml`
for an example).

Top-level entries in the file correspond to individual experiments to run. Each
such entry must have four subsections: `experiment`, `train`, `decode`,
and `evaluate`. Options for each subsection are listed below.

There can be a special top-level entry named `defaults`; if it is
present, parameters defined in it will act as defaults for other experiments
in the configuration file.

The stdout and stderr outputs of an experiment will be written to `<experiment-name>.log`
and `<experiment-name>.err.log` in the current directory.

## Option tables

### experiment

| Name | Description | Type | Default value |
|------|-------------|------|---------------|
| **model_file** | Location to write the model file | str |  |
| **hyp_file** | Temporary location to write decoded output for evaluation | str |  |
| eval_metrics | Comma-separated list of evaluation metrics | str | bleu |
| **run_for_epochs** | How many epochs to run each test for | int |  |
| decode_every | Evaluation period in epochs. If set to 0, will never evaluate. | int | 0 |


### decode

| Name | Description | Type | Default value |
|------|-------------|------|---------------|
| **source_file** | path of input source file to be translated | str |  |
| input_format | format of input data: text/contvec | str | text |

### evaluate

| Name | Description | Type | Default value |
|------|-------------|------|---------------|
| **ref_file** | path of the reference file | str |  |

### train

| Name | Description | Type | Default value |
|------|-------------|------|---------------|
| eval_every |  | int | 1000 |
| batch_size |  | int | 32 |
| batch_strategy |  | str | src |
| **train_source** |  | str |  |
| **train_target** |  | str |  |
| **dev_source** |  | str |  |
| **dev_target** |  | str |  |
| input_format | format of input data: text/contvec | str | text |
| input_word_embed_dim |  | int | 67 |
| output_word_embed_dim |  | int | 67 |
| output_state_dim |  | int | 67 |
| attender_hidden_dim |  | int | 67 |
| output_mlp_hidden_dim |  | int | 67 |
| encoder_hidden_dim |  | int | 64 |
| trainer |  | str | sgd |
| eval_metrics |  | str | bleu |
| encoder_layers |  | int | 2 |
| decoder_layers |  | int | 2 |
| encoder_type |  | str | BiLSTM |
| decoder_type |  | str | LSTM |
