import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tinker_extend.extend_client import (
    ExtendAPIError, ExtendClient, ExtendRunFailed, encode_multipart,
)


def test_encode_multipart_roundtrip():
    body, ctype = encode_multipart("file", "doc.pdf", b"%PDF-1.4 fake")
    assert ctype.startswith("multipart/form-data; boundary=")
    assert b'name="file"; filename="doc.pdf"' in body
    assert b"%PDF-1.4 fake" in body
    assert body.endswith(b"--\r\n")


def test_check_run_failed():
    client = ExtendClient(api_key="dummy")
    try:
        client._check_run({"status": "FAILED", "failureReason": "OCR_ERROR",
                           "failureMessage": "bad scan"})
    except ExtendRunFailed as e:
        assert e.failure_reason == "OCR_ERROR"
    else:
        raise AssertionError("expected ExtendRunFailed")


def test_check_run_processed_passthrough():
    client = ExtendClient(api_key="dummy")
    run = {"status": "PROCESSED", "output": {"chunks": []}}
    assert client._check_run(run) is run


def test_client_requires_key():
    try:
        ExtendClient(api_key="")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
