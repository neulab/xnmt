This trains a neural machine translation model on the [KFTT](http://www.phontron.com/kftt/) en-ja data.

How to run

    # Download data
    ./download.sh
    # Train model
    xnmt --dynet-gpu ./config.kftt.en-ja.yaml

On a GTX 1080 Ti this trains at ~2700-2800 words/sec and achieves a BLEU score of 23.14 after 12 epochs of training.

Reference output after the final evaluation, for random seed `3301906583`:

    Experiment                    | Final Scores
    -----------------------------------------------------------------------
    kftt.en-ja                    | BLEU4: 0.20526004248457844, 0.574099/0.306461/0.179848/0.114350 (BP = 0.836909, ratio=0.85, hyp_len=22787, ref_len=26844)
                                  | BLEU4: 0.23145703986424254, 0.593274/0.333834/0.205808/0.136245 (BP = 0.847868, ratio=0.86, hyp_len=24444, ref_len=28478)
