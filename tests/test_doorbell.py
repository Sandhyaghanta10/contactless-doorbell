import cv2
import numpy as np
import pytest

from doorbell.benchmark import evaluate, score_open_set, threshold_sweep
from doorbell.dataset import load_gallery, split_gallery
from doorbell.detector import HaarFaceDetector
from doorbell.notifier import EmailNotifier, build_message
from doorbell.pipeline import Doorbell
from doorbell.preprocess import adjust_gamma, normalize_face
from doorbell.recognizer import FaceRecognizer, Match


# --- preprocessing ---------------------------------------------------------
def test_normalize_face_resizes_and_equalizes():
    dim = np.full((50, 80), 40, dtype=np.uint8)
    dim[10:30, 10:30] = 60
    out = normalize_face(dim, size=32, equalize=True)
    assert out.shape == (32, 32) and out.max() == 255  # equalization stretches contrast
    assert normalize_face(dim, size=32, equalize=False).max() < 255


def test_gamma_darkens_brightens_and_identity():
    img = np.full((4, 4), 128, dtype=np.uint8)
    assert adjust_gamma(img, 2.5).mean() < 128 < adjust_gamma(img, 0.4).mean()
    assert (adjust_gamma(img, 1.0) == img).all()


# --- recognizer ------------------------------------------------------------
def test_recognizer_identifies_registered_visitors(make_gallery):
    recognizer = FaceRecognizer(threshold=float("inf"))
    recognizer.train(make_gallery(range(4), photos=6))
    probes = make_gallery(range(4), photos=3, seed_offset=500)
    hits = sum(recognizer.identify(f).name == n for n, faces in probes.items() for f in faces)
    assert hits == 12


def test_threshold_turns_strangers_into_unknown(make_gallery):
    train = make_gallery(range(4), photos=6)
    stranger = make_gallery([50], photos=3)["person50"]
    known_probe = make_gallery([2], photos=1, seed_offset=900)["person2"][0]
    recognizer = FaceRecognizer(threshold=float("inf"))
    recognizer.train(train)
    d_known = recognizer.identify(known_probe).distance
    d_stranger = min(recognizer.identify(f).distance for f in stranger)
    assert d_stranger > d_known  # strangers look less similar than a registered visitor
    recognizer.threshold = (d_known + d_stranger) / 2
    assert recognizer.identify(known_probe).name == "person2"
    assert recognizer.identify(stranger[0]).name is None


def test_recognizer_guards():
    with pytest.raises(ValueError):
        FaceRecognizer().train({})
    with pytest.raises(RuntimeError):
        FaceRecognizer().identify(np.zeros((10, 10), np.uint8))


# --- pipeline --------------------------------------------------------------
class WholeImageDetector:
    def detect(self, gray):
        return [(0, 0, gray.shape[1], gray.shape[0])]


class FakeRecognizer:
    def __init__(self, name):
        self.name = name

    def identify(self, face):
        return Match(self.name, 12.0)


class FakeNotifier:
    def __init__(self):
        self.sent = []

    def notify(self, subject, body):
        self.sent.append((subject, body))


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def make_bell(name, cooldown=30.0):
    notifier, clock = FakeNotifier(), Clock()
    return Doorbell(WholeImageDetector(), FakeRecognizer(name), notifier, cooldown, clock), notifier, clock


FRAME = np.full((40, 40, 3), 100, dtype=np.uint8)  # BGR frame -> also covers colour conversion


def test_cooldown_suppresses_repeat_alerts_then_allows_again():
    bell, notifier, clock = make_bell("Asha", cooldown=30)
    bell.process(FRAME)
    clock.now = 29
    bell.process(FRAME)
    assert len(notifier.sent) == 1
    clock.now = 30
    bell.process(FRAME)
    assert len(notifier.sent) == 2


def test_known_and_unknown_visitors_are_throttled_separately():
    bell, notifier, _ = make_bell("Asha")
    bell.process(FRAME)
    bell._recognizer = FakeRecognizer(None)  # a stranger shows up right after
    bell.process(FRAME)
    assert [s for s, _ in notifier.sent] == ["Doorbell: Asha", "Doorbell: unknown visitor"]


def test_no_face_no_alert():
    class NoFaces:
        def detect(self, gray):
            return []
    notifier = FakeNotifier()
    assert Doorbell(NoFaces(), FakeRecognizer("x"), notifier).process(FRAME) == []
    assert notifier.sent == []


# --- notifier --------------------------------------------------------------
def test_build_message_headers():
    msg = build_message("a@x.com", "b@y.com", "Hi", "body")
    assert (msg["From"], msg["To"], msg["Subject"]) == ("a@x.com", "b@y.com", "Hi")


def test_email_notifier_sends_via_smtp_with_tls_and_login():
    calls = []

    class FakeSMTP:
        def __init__(self, host, port):
            calls.append(("connect", host, port))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self):
            calls.append("tls")

        def login(self, user, password):
            calls.append(("login", user))

        def send_message(self, msg):
            calls.append(("send", msg["To"]))

    EmailNotifier("smtp.test", 587, "u", "p", "to@x.com", smtp_factory=FakeSMTP).notify("s", "b")
    assert calls == [("connect", "smtp.test", 587), "tls", ("login", "u"), ("send", "to@x.com")]


def test_from_env_requires_settings(monkeypatch):
    monkeypatch.delenv("DOORBELL_SMTP_HOST", raising=False)
    with pytest.raises(SystemExit, match="DOORBELL_SMTP_HOST"):
        EmailNotifier.from_env()


# --- detector --------------------------------------------------------------
def test_haar_detector_loads_and_finds_nothing_in_blank_image():
    assert HaarFaceDetector().detect(np.full((200, 200), 128, np.uint8)) == []


# --- dataset & benchmark ---------------------------------------------------
def test_load_and_split_gallery(tmp_path, make_gallery):
    for name, faces in make_gallery(range(3), photos=6).items():
        (tmp_path / name).mkdir()
        for i, face in enumerate(faces):
            cv2.imwrite(str(tmp_path / name / f"{i}.png"), face)
    gallery = load_gallery(tmp_path)
    assert sorted(gallery) == ["person0", "person1", "person2"]
    train, test = split_gallery(gallery, train_per_person=4, seed=1)
    assert all(len(v) == 4 for v in train.values()) and all(len(v) == 2 for v in test.values())
    assert split_gallery(gallery, 4, seed=1)[0]["person0"][0].tobytes() == train["person0"][0].tobytes()
    with pytest.raises(ValueError):
        split_gallery(gallery, train_per_person=6)


def test_load_gallery_empty_dir_fails(tmp_path):
    with pytest.raises(ValueError):
        load_gallery(tmp_path)


def test_evaluate_reports_accuracy_and_latency(make_gallery):
    train = make_gallery(range(4), photos=6)
    test = make_gallery(range(4), photos=3, seed_offset=700)
    result = evaluate(train, test, size=64, equalize=True, gamma=1.0)
    assert result.accuracy == 1.0 and 0 < result.p50_ms <= result.p95_ms


def test_threshold_sweep_is_monotonic(make_gallery):
    train = make_gallery(range(4), photos=6)
    test = make_gallery(range(4), photos=3, seed_offset=700)
    strangers = make_gallery(range(50, 53), photos=3)
    known, stranger_d = score_open_set(train, test, strangers, 100, True)
    rows = threshold_sweep(known, stranger_d, [10, 50, 100, 1000])
    tars, fars = [r[1] for r in rows], [r[2] for r in rows]
    assert tars == sorted(tars) and fars == sorted(fars)  # looser threshold never accepts fewer
    assert rows[-1][1] == 1.0 and rows[-1][2] == 1.0       # infinite threshold accepts everyone
