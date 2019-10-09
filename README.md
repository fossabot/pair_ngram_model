# Pair n-gram modeling toolkit

## Requirements

* [Python 3.6 or better](https://www.python.org/)
* [OpenFst 1.7.3](http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.7.3.tar.gz)
* [Pynini 2.0.8](http://www.opengrm.org/twiki/pub/GRM/PyniniDownload/pynini-2.0.8.tar.gz)
* [Baumwelch 0.2.9](http://www.openfst.org/twiki/pub/Contrib/FstContrib/baumwelch-0.2.9.tar.gz)
* [OpenGrm-NGram 1.3.7](http://www.opengrm.org/twiki/pub/GRM/NGramDownload/ngram-1.3.7.tar.gz)

## Suggested workflow

    readonly SEED="${RANDOM}"
    echo "Using seed: ${SEED}"
    ./split.py \
        --seed="${SEED}" \
        --input_path=lexicon.tsv \
        --train_path=train.tsv \
        --dev_path=dev.tsv \
        --test_path=test.tsv
    ./train.py \
        --seed="${SEED}" \
        --input_path=train.tsv \
        --model_path=model.fst
    # TODO: hyperparameter optimization using dev.tsv
    cut -f1 test.tsv > test.g
    cut -f2 test.tsv > gold.p
    ./rewrite.py \
        --word_path=test.g \
        --fst_path=model.fst > hypo.p
    paste gold.p hypo.p > eval.tsv
    ./evaluate.py eval.tsv
    rm -f test.g gold.p hypo.p
    rm -f train.tsv dev.tsv test.tsv eval.tsv
