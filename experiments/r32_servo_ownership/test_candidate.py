from experiments.r32_servo_ownership.candidate import OwnedSource,SETTLE_SAMPLES,REPORT_SAMPLES

def begin(x):
    for _ in range(3):x.feed(True,True,False,False)
def release(x):
    assert not x.feed(False,True,False,False);assert not x.feed(False,True,False,False);assert x.feed(False,True,False,False)
def test_reacquire_blocked_until_two_settled_windows():
    x=OwnedSource();begin(x);release(x)
    assert sum(x.feed(True,True,False,False) for _ in range(100))==REPORT_SAMPLES-1
    assert x.awaiting_settle
    for _ in range(SETTLE_SAMPLES-1):assert not x.feed(False,True,True,False)
    assert x.awaiting_settle;assert not x.feed(False,True,True,False);assert not x.awaiting_settle
def test_incumbent_owns_source_and_separate_source_rearms():
    x=OwnedSource();begin(x);x.feed(True,True,False,True)
    assert not any(x.feed(False,True,False,False) for _ in range(3));assert x.awaiting_settle
    for _ in range(SETTLE_SAMPLES):x.feed(False,True,True,False)
    begin(x);release(x);assert x.pulse==REPORT_SAMPLES-1
