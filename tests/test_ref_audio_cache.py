import hashlib
import pathlib
import tempfile
import time

import pytest

from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_local_file_cache_key_invalidation():
    print("Testing local file cache key invalidation...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        test_file = tmp_path / "test_audio.wav"

        # 1. Create a dummy file and get its cache key
        test_file.write_text("dummy audio data 1")
        file_uri = f"file://{test_file}"

        key1 = OmniOpenAIServingSpeech._get_ref_audio_cache_key(file_uri)
        print(f"  [1] Key for original file: {key1}")

        # Wait slightly to ensure the mtime changes on the filesystem
        time.sleep(0.01)

        # 2. Modify the file
        test_file.write_text("dummy audio data 2 modified")

        # 3. Get the new cache key
        key2 = OmniOpenAIServingSpeech._get_ref_audio_cache_key(file_uri)
        print(f"  [2] Key for modified file: {key2}")

        assert key1 != key2, "Cache key should change when file is modified!"
        print("  ✓ Passed: Keys are correctly invalidated based on file metadata.")


def test_remote_url_cache_key():
    print("\nTesting remote URL cache key stability...")
    url = "https://example.com/audio.wav"
    key1 = OmniOpenAIServingSpeech._get_ref_audio_cache_key(url)
    key2 = OmniOpenAIServingSpeech._get_ref_audio_cache_key(url)

    assert key1 == key2, "Cache key for URLs should remain constant!"
    print("  ✓ Passed: URLs correctly maintain a constant cache key.")


def test_missing_file_cache_key(caplog):
    """A missing file must fall back to the URI-string hash AND emit a warning.

    The reviewer's core complaint was 'silent failure should be fixed' — so we
    must assert that the warning is actually logged, not just that the fallback
    key value is correct.
    """
    import logging

    file_uri = "file:///path/to/nonexistent/file.wav"
    with caplog.at_level(logging.WARNING):
        key = OmniOpenAIServingSpeech._get_ref_audio_cache_key(file_uri)

    expected_key = hashlib.sha1(file_uri.encode("utf-8")).hexdigest()
    assert key == expected_key, "Missing files should fallback to the string hash."
    assert any("stale cache" in r.message for r in caplog.records), (
        "A warning about stale cache must be emitted when os.stat fails"
    )


def test_percent_encoded_file_uri():
    """Percent-encoded file:// URIs (e.g. %20 for spaces) must be decoded
    so that os.stat hits the real file, not a wrong/nonexistent path."""
    print("\nTesting percent-encoded file URI...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        # Create a file whose name contains a space
        test_file = tmp_path / "my audio.wav"
        test_file.write_text("dummy audio data")
        # Build a URI with the space percent-encoded
        encoded_uri = f"file://{str(test_file).replace(' ', '%20')}"

        key = OmniOpenAIServingSpeech._get_ref_audio_cache_key(encoded_uri)
        # The key must incorporate mtime/size (not fall back to string-only),
        # so it should differ from a plain SHA-1 of the URI string.
        string_only_key = hashlib.sha1(encoded_uri.encode("utf-8")).hexdigest()
        assert key != string_only_key, (
            "Percent-encoded URI should be decoded and stat'd, "
            "not fall back to string-only cache key!"
        )
        print("  ✓ Passed: Percent-encoded file URIs are decoded correctly.")


if __name__ == "__main__":
    test_local_file_cache_key_invalidation()
    test_remote_url_cache_key()
    # test_missing_file_cache_key requires the pytest caplog fixture; run via pytest
    test_percent_encoded_file_uri()
    print("\nAll unit tests passed successfully!")
