from types import SimpleNamespace

from argus_operator_ui.synchronizer import FrameSynchronizer, stamp_to_nanoseconds


def test_stamp_to_nanoseconds():
    assert stamp_to_nanoseconds(SimpleNamespace(sec=3, nanosec=42)) == 3_000_000_042


def test_exact_timestamp_match_in_either_order():
    sync = FrameSynchronizer()
    assert sync.insert_image(10, "image") is None
    match = sync.insert_result(10, "result")
    assert (match.image, match.result, match.stamp_ns) == ("image", "result", 10)
    assert sync.insert_result(11, "result2") is None
    match = sync.insert_image(11, "image2")
    assert (match.image, match.result) == ("image2", "result2")


def test_different_timestamps_do_not_match():
    sync = FrameSynchronizer()
    sync.insert_image(10, "image")
    assert sync.insert_result(11, "result") is None
    assert sync.cache_sizes == (1, 1)


def test_capacity_discards_oldest():
    sync = FrameSynchronizer(capacity=2)
    for stamp in (1, 2, 3):
        sync.insert_image(stamp, stamp)
    assert sync.cache_sizes == (2, 0)
    assert sync.insert_result(1, "old") is None


def test_expired_entries_are_cleaned():
    now = [0.0]
    sync = FrameSynchronizer(timeout_sec=1.0, clock=lambda: now[0])
    sync.insert_image(1, "image")
    now[0] = 1.01
    sync.cleanup()
    assert sync.cache_sizes == (0, 0)
