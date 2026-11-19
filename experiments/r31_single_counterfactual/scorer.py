"""Pair a forced trace with a no-force trace at any registered onset."""
from contextlib import contextmanager

from experiments.r21_full_protection import runner_v2 as raw
from experiments.r22_census import check


@contextmanager
def arbitrary_onset(index):
    """Temporarily request exactly one no-force prefix from the stock recorder."""
    index=int(index)
    if not 0<=index<6000:raise ValueError('onset outside fixed trace')
    previous=raw.ONSET_INDICES
    raw.ONSET_INDICES=(index,)
    try:yield
    finally:raw.ONSET_INDICES=previous


def calm_record(detector,case,onset_index):
    if case.get('collision') is not None:raise ValueError('calm case must have collision=null')
    with arbitrary_onset(onset_index):return raw.episode_record(detector,case)


def pair(detector,calm,forced):
    collision=forced.get('collision')
    if collision is None:raise ValueError('forced case needs collision')
    onset=int(round(float(collision[0])*1000))
    normal=calm_record(detector,calm,onset)
    event=raw.episode_record(detector,forced)
    score=check.score(forced,event,normal)
    return normal,event,score
