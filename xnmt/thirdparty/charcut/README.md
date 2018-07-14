# CharCut in XNMT

This package contains a modified version of CharCut: https://github.com/alardill/CharCut

# CharCut
Character-based MT evaluation and difference highlighting

CharCut compares outputs of MT systems with reference translations. It produces HTML outputs showing character-based differences along with scores that are directly inferred from the lengths of those differences, thus making the link between evaluation and visualisation straightforward.

The matching algorithm is based on an iterative search for longest common substrings, combined with a length-based threshold that limits short and noisy character matches. As a similarity metric this is not new, but to the best of our knowledge it was never applied to highlighting and scoring of MT outputs. It has the neat effect of keeping character-based differences readable by humans.

Accidentally, the scores inferred from those differences correlate very well with human judgements, similarly to other great character-based metrics like [chrF(++)](https://github.com/m-popovic/chrF) or [CharacTER](https://github.com/rwth-i6/CharacTER). It has been evaluated here:
> Adrien Lardilleux and Yves Lepage: "CharCut: Human-Targeted Character-Based MT Evaluation with Loose Differences". In [Proceedings of IWSLT 2017](http://workshop2017.iwslt.org/64.php).

It is intended to be lightweight and easy to use, so the HTML outputs are, and will be kept, slick on purpose.

## Usage

CharCut is written in Python 2.

Basic usage:
```
python charcut.py -c cand.txt -r ref.txt
```
where `cand.txt` and `ref.txt` contain corresponding candidate (MT) and reference (human) segments, 1 per line.
By default, only a document-level score is displayed on standard output. To produce a HTML output file, use the `-o` option:
```
python charcut.py -c cand.txt -r ref.txt -o mydiff.html
```

A few more options are available; call
```
python charcut.py -h
```
to list them.

Consider lowering the `-m` option value (minimum match size) for non-alphabetical writing systems such as Chinese or Japanese. The default value (3 characters) should be acceptable for most European languages, but depending on the language and data, larger values might produce better looking results.
