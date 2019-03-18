"""
Home assignment 19-03-2019.
Krzysztof Kowalczyk kk385830
"""
import Bio
from Bio.Seq import Seq
from Bio.pairwise2 import align
from Bio.SubsMat.MatrixInfo import blosum30

import math
from itertools import product
from typing import Iterable, Callable, List, NamedTuple
from collections import defaultdict


class Alignment(NamedTuple):
    align1: str
    align2: str
    score: float
    begin: int
    end: int


def _translated_offsets(seq: Bio.Seq.Seq, max_range=3) -> Iterable[Bio.Seq.Seq]:
    for offset in range(max_range):
        seq_end = 3 * ((len(seq)-offset)//3)
        yield seq[offset:offset+seq_end].translate()

def default_align_strategy(seq1, seq2) -> Alignment:
    return Alignment(*align.localxx(seq1, seq2)[0])

def blosum_align_strategy(seq1, seq2) -> Alignment:
    blosum_extended = defaultdict(lambda: -math.inf)
    blosum_extended.update(blosum30)
    return Alignment(*align.localds(seq1, seq2, blosum_extended, -1, -0.5)[0])


def optimal_alignment(
        dna1: Bio.Seq.Seq, 
        dna2: Bio.Seq.Seq, 
        align_strategy: Callable[[Bio.Seq.Seq, Bio.Seq.Seq], Alignment]=default_align_strategy
) -> Alignment:
    """
    Calculates optimal alignment of 2 translated sequences, checking all 9 possible offsets in
    the translated DNA with the supplied alignment strategy.
    """
    translated_pairs = product(_translated_offsets(dna1), _translated_offsets(dna2))
    return max(
        (align_strategy(seq1, seq2) for seq1, seq2 in translated_pairs), 
        key = lambda alignment: alignment.score
    )
