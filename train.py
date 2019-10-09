#!/usr/bin/env python
"""Builds a end pair n-gram model language model."""

import argparse
import functools
import logging
import os
import subprocess
import tempfile

from typing import Set

import pynini
import pywrapfst


class PairNGramTrainer:
    """ Build a end-to-end g2p pair language model"""

    def __init__(self):
        self.g_far_path = tempfile.mkstemp(prefix="g.", suffix=".far")[1]
        self.p_far_path = tempfile.mkstemp(prefix="p.", suffix=".far")[1]
        self.covering_path = tempfile.mkstemp(
            prefix="covering.", suffix=".fst"
        )[1]
        self.aligner_path = tempfile.mkstemp(prefix="aligner.", suffix=".fat")[
            1
        ]
        self.far_path = tempfile.mkstemp(prefix="far.", suffix=".far")[1]
        self.fsa_path = tempfile.mkstemp(prefix="fsa.", suffix=".far")[1]
        self.count_path = tempfile.mkstemp(prefix="count.", suffix=".fst")[1]
        self.lm_path = tempfile.mkstemp(prefix="lm.", suffix=".fst")[1]
        self.shrunk_lm_path = tempfile.mkstemp(
            prefix="shrunk.", suffix=".fst"
        )[1]

    def _label_union(self, labels: Set[int], epsilon: bool) -> pynini.Fst:
        """Creates FSA over a union of the labels."""
        if epsilon:
            labels.add(0)
        side = pynini.Fst()
        src = side.add_state()
        side.set_start(src)
        dst = side.add_state()
        for label in labels:
            side.add_arc(src, pynini.Arc(label, label, None, dst))
        side.set_final(dst)
        assert side.verify(), "FST is ill-formed"
        return side

    def _narcs(self, f: pynini.Fst) -> int:
        """Computes the number of arcs in an FST."""
        return sum(f.num_arcs(state) for state in f.states())

    def _lexicon_covering(
        self,
        token_type: str,
        input_path: str,
        input_epsilon: bool,
        output_epsilon: bool,
    ) -> None:
        # Sets of labels for the covering grammar.
        g_labels: Set[int] = set()
        p_labels: Set[int] = set()
        # Curries compiler and compactor functions for the FARs.
        compiler = functools.partial(
            pynini.acceptor, token_type=token_type, attach_symbols=False
        )
        compactor = functools.partial(
            pywrapfst.convert, fst_type="compact_string"
        )
        logging.info("Constructing grapheme and phoneme FARs")
        g_writer = pywrapfst.FarWriter.create(self.g_far_path)
        p_writer = pywrapfst.FarWriter.create(self.p_far_path)
        with open(input_path, "r") as source:
            for (linenum, line) in enumerate(source, 1):
                key = f"{linenum:08x}"
                (g, p) = line.rstrip().split("\t", 1)
                # For both G and P, we compile a FSA, store the labels, and
                # then write the compact version to the FAR.
                g_fst = compiler(g)
                g_labels.update(g_fst.paths().ilabels())
                g_writer[key] = compactor(g_fst)
                p_fst = compiler(p)
                p_labels.update(p_fst.paths().ilabels())
                p_writer[key] = compactor(p_fst)
        logging.info("Processed %d examples", linenum)
        logging.info("Constructing covering grammar")
        logging.info("%d unique graphemes", len(g_labels))
        g_side = self._label_union(g_labels, input_epsilon)
        logging.info("%d unique phonemes", len(p_labels))
        p_side = self._label_union(p_labels, output_epsilon)
        # The covering grammar is given by (G x P)^*, a zeroth order Markov
        # model.
        covering = pynini.transducer(g_side, p_side).closure().optimize()
        assert covering.num_states() == 1, "Covering grammar FST is ill-formed"
        logging.info("Covering grammar has %d arcs", self._narcs(covering))
        covering.write(self.covering_path)

    def _alignment(
        self, max_iters: int, random_starts: int, seed: int
    ) -> None:
        # This method has to be changed after parallel aligner traning
        # becomes available.
        logging.info("Baum-Welch training")
        cmd = [
            "baumwelchtrain",
            f"--max_iters={max_iters}",
            f"--random_starts={random_starts}",
            f"--seed={seed}",
            self.g_far_path,
            self.p_far_path,
            self.covering_path,
            self.aligner_path,
        ]
        subprocess.check_call(cmd)
        os.remove(self.covering_path)
        logging.info("Baum-Welch decoding")
        cmd = [
            "baumwelchdecode",
            self.g_far_path,
            self.p_far_path,
            self.aligner_path,
            self.far_path,
        ]
        subprocess.check_call(cmd)
        os.remove(self.g_far_path)
        os.remove(self.p_far_path)
        os.remove(self.aligner_path)

    def _model(
        self,
        order: int,
        target_number_of_ngrams: int,
        smoothing_method: str,
        shrinking_method: bool,
        model_path: str,
    ):
        with pynini.Far(self.far_path, mode="r") as far_reader:
            encoder = pynini.EncodeMapper(
                far_reader.arc_type(), encode_labels=True
            )
            # Alignment encoding.
            with pynini.Far(
                self.fsa_path,
                mode="w",
                arc_type=far_reader.arc_type(),
                far_type="default",
            ) as far_writer:
                while not far_reader.done():
                    fst = far_reader.get_fst()
                    fst.encode(encoder)
                    far_writer.add(far_reader.get_key(), fst)
                    far_reader.next()
        logging.info("Building LM")
        # LM counting.
        cmd = [
            "ngramcount",
            "--require_symbols=false",
            f"--order={order}",
            self.fsa_path,
            self.count_path,
        ]
        subprocess.check_call(cmd)
        os.remove(self.fsa_path)
        # LM smoothing.
        cmd = [
            "ngrammake",
            f"--method={smoothing_method}",
            self.count_path,
            self.lm_path,
        ]
        subprocess.check_call(cmd)
        os.remove(self.count_path)
        # LM shrinking.
        if shrinking_method:
            cmd = [
                "ngramshrink",
                "--method=relative_entropy",
                f"--target_number_of_ngrams={target_number_of_ngrams}",
                self.lm_path,
                self.shrunk_lm_path,
            ]
            subprocess.check_call(cmd)
            self.lm_path = self.shrunk_lm_path
        # LM decoding.
        model = pynini.Fst.read(self.lm_path)
        os.remove(self.lm_path)
        model.decode(encoder)
        model.write(model_path)

    def train(
        self,
        input_path: str,
        token_type: str,
        input_epsilon: bool,
        output_epsilon: bool,
        max_iters: int,
        random_starts: int,
        seed: int,
        order: int,
        target_number_of_ngrams: int,
        smoothing_method: str,
        shrinking_method: str,
        model_path: str,
    ):
        self._lexicon_covering(
            token_type, input_path, input_epsilon, output_epsilon
        )
        self._alignment(max_iters, random_starts, seed)
        self._model(
            order,
            target_number_of_ngrams,
            smoothing_method,
            shrinking_method,
            model_path,
        )


def main(args: argparse.Namespace) -> None:
    trainer = PairNGramTrainer()
    trainer.train(
        # Lexicon options.
        args.input_path,
        args.token_type,
        args.input_epsilon,
        args.output_epsilon,
        # Aligner options.
        args.max_iters,
        args.random_starts,
        args.seed,
        # Language model options.
        args.order,
        args.target_number_of_ngrams,
        args.smoothing_method,
        args.shrinking_method,
        args.model_path,
    )


if __name__ == "__main__":
    logging.basicConfig(format="%(levelname)s: %(message)s", level="INFO")
    parser = argparse.ArgumentParser(description=__doc__)
    # Lexicon options.
    parser.add_argument(
        "--input_path", required=True, help="input TSV file path"
    )
    parser.add_argument(
        "--token_type",
        default="utf8",
        help="token type for acceptors. (default: %(default)s)",
    )
    parser.add_argument(
        "--input_epsilon",
        default=True,
        help="allows input graphemes to have a null alignment "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--output_epsilon",
        default=True,
        help="allows input phonemes to have a null alignment "
        "(default: %(default)s)",
    )
    # Aligner options.
    parser.add_argument(
        "--max_iters",
        default=50,
        help="maximum number of Baum-Welch iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--random_starts",
        default=10,
        help="number of random starts for Baum-Welch training "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--seed", required=True, help="random seed for Baum-Welch training"
    )
    parser.add_argument(
        "--order", default="6", help="n-gram order (default: %(default)s)"
    )
    parser.add_argument(
        "--smoothing_method",
        default="kneser_ney",
        help="smoothing method (default: %(default)s)",
    )
    parser.add_argument(
        "--shrinking_method",
        default="relative_entropy",
        help="shrinking method (default: %(default)s)",
    )
    parser.add_argument(
        "--target_number_of_ngrams",
        default=100000,
        help="target number of n-grams for shrinking (default: %(default)s)",
    )
    parser.add_argument(
        "--model_path", required=True, help="input result FST path"
    )
    main(parser.parse_args())
