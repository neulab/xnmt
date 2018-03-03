Experiment configuration file format
------------------------------------

Configuration files are in `YAML dictionary format <https://docs.ansible.com/ansible/YAMLSyntax.html>`_.

At the top-level, a config file consists of a dictionary where keys are experiment
names and values are the experiment specifications. By default, all experiments
are run in lexicographical ordering, but xnmt_run_experiments can also be told
to run only a selection of the specified experiments. An example template with
2 experiments looks like this::

    exp1: !Experiment
      exp_global: ...
      preproc: ...
      model: ...
      train: ...
      evaluate: ...
    exp2: !Experiment
      exp_global: ...
      preproc: ...
      model: ...
      train: ...
      evaluate: ...

``!Experiment`` is YAML syntax specifying a Python object of the same name, and
its parameters will be passed on to the Python constructor.
There can be a special top-level entry named ``defaults``; this experiment will
never be run, but can be used as a template where components are partially shared
using YAML anchors or the !Ref mechanism (more on this later).

The usage of ``exp_global``, ``preproc``, ``model``, ``train``, ``evaluate``
are explained below.
Not all of them need to be specified, depending on the use case.

exp_global
==========
This specifies settings that are global to this experiment. An example::

  exp_global: !ExpGlobal
    model_file: '{EXP_DIR}/models/{EXP}.mod'
    log_file: '{EXP_DIR}/logs/{EXP}.log'
    default_layer_dim: 512
    dropout: 0.3

Not that for any strings used here or anywhere in the config file ``{EXP}`` will
be over-written by the name of the experiment, ``{EXP_DIR}`` will be overwritten
by the directory the config file lies in, and ``{PID}`` by the process id.

To obtain a full list of allowed parameters, please check the constructor of
``ExpGlobal``, specified under xnmt/exp_global.py. Behind the scenes, this class
also manages the DyNet parameters, it is therefore referenced by all components
that use DyNet parameters.

preproc
======= 
``xnmt`` supports a variety of data preprocessing features. Please refer to
``preprocessing.rst`` for details.

model
=====
This specifies the model architecture. An typical example looks like this::

  model: !DefaultTranslator
    src_reader: !PlainTextReader
      vocab: !Vocab {vocab_file: examples/data/head.ja.vocab}
    trg_reader: !PlainTextReader
      vocab: !Vocab {vocab_file: examples/data/head.en.vocab}
    encoder: !BiLSTMSeqTransducer
      layers: 1
    attender: !MlpAttender
      hidden_dim: 512
      state_dim: 512
      input_dim: 512
    trg_embedder: !SimpleWordEmbedder
      emb_dim: 512
    decoder: !MlpSoftmaxDecoder
      layers: 1
      mlp_hidden_dim: 512
      bridge: !CopyBridge {}

The top level entry is typically DefaultTranslator, which implements a standard
attentional sequence-to-sequence model. It allows flexible specification of
encoder, attender, source / target embedder, and other settings. Again, to obtain
the full list of supported options, please refer to the corresponding class
initializer methods.

Note that some of this Python objects are passed to their parent object's
initializer method, which requires that the children are initialized first.
``xnmt`` therefore uses a bottom-up initialization strategy, where siblings
are initialized in the order they appear in the constructor. Among others,
this causes ``exp_global`` (the first child of the top-level experiment) to be
initialized before any model component is initialized, so that model components
are free to use exp_global's global default settings, DyNet parameters, etc.
It also guarantees that preprocessing is carried out before the model training.

train
=====
A typical example looks like this::

  train: !SimpleTrainingRegimen
    trainer: !AdamTrainer
      alpha: 0.001
    run_for_epochs: 2
    src_file: examples/data/head.ja
    trg_file: examples/data/head.en
    dev_tasks:
      - !LossEvalTask
        src_file: examples/data/head.ja
        ref_file: examples/data/head.en

The expected object here is a subclass of TrainingRegimen. Besides
``SimpleTrainingRegimen``, multi-task style training regimens are supported.
For multi task training, each training regimen uses their own model, so in this
case models must be specified as sub-components of the training regimen. Please
refer to examples/08_multitask.yaml for more details on this.

evaluate
========
If specified, the model is tested after training finished.
