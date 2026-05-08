import json
from unittest.mock import MagicMock, patch
from urllib.error import URLError

from drift.notifications.slack import FailedPromptSummary, notify_drift

_BASE_KWARGS = dict(
    eval_run_id=7,
    baseline_run_id=1,
    provider="openai",
    model="gpt-4o-mini",
    threshold=0.95,
    failed_count=2,
    total_compared=3,
    failed_prompts=[
        FailedPromptSummary("greet", 0.82, "Hello!", "Howdy partner!"),
        FailedPromptSummary("math", 0.71, "611", "It's 611."),
    ],
    report_path="reports/run-7.md",
)


def test_no_op_when_webhook_not_set(monkeypatch):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    with patch("drift.notifications.slack.urllib.request.urlopen") as urlopen:
        notify_drift(**_BASE_KWARGS)
    urlopen.assert_not_called()


def test_no_op_when_webhook_is_empty_string(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "")
    with patch("drift.notifications.slack.urllib.request.urlopen") as urlopen:
        notify_drift(**_BASE_KWARGS)
    urlopen.assert_not_called()


def test_posts_when_webhook_set(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/x/y/z")
    mock_resp = MagicMock()
    mock_resp.status = 200
    with patch("drift.notifications.slack.urllib.request.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = mock_resp
        notify_drift(**_BASE_KWARGS)

    urlopen.assert_called_once()
    request = urlopen.call_args[0][0]
    assert request.full_url == "https://hooks.slack.com/services/x/y/z"
    assert request.get_method() == "POST"

    body = json.loads(request.data.decode("utf-8"))
    assert "Drift detected on run 7" in body["text"]
    assert "2/3" in body["text"]

    blocks_text = json.dumps(body["blocks"])
    assert "greet" in blocks_text
    assert "0.820" in blocks_text  # similarity formatted to 3 dp
    assert "Howdy partner!" in blocks_text
    assert "openai" in blocks_text
    assert "run 1" in blocks_text  # baseline


def test_truncates_to_three_failed_prompts(monkeypatch):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/x/y/z")
    mock_resp = MagicMock()
    mock_resp.status = 200
    five_failures = [
        FailedPromptSummary(f"p{i}", 0.5, f"baseline {i}", f"now {i}") for i in range(5)
    ]
    kwargs = {**_BASE_KWARGS, "failed_count": 5, "failed_prompts": five_failures}
    with patch("drift.notifications.slack.urllib.request.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = mock_resp
        notify_drift(**kwargs)

    body = json.loads(urlopen.call_args[0][0].data.decode("utf-8"))
    blocks_text = json.dumps(body["blocks"])
    # First three present, last two omitted
    for i in range(3):
        assert f"p{i}" in blocks_text
    for i in range(3, 5):
        assert f"p{i}" not in blocks_text
    assert "and 2 more" in blocks_text


def test_failure_prints_warning_and_returns(monkeypatch, capsys):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/x/y/z")
    with patch("drift.notifications.slack.urllib.request.urlopen", side_effect=URLError("boom")):
        notify_drift(**_BASE_KWARGS)  # must not raise

    out = capsys.readouterr().out
    assert "Slack notification failed" in out
    assert "boom" in out


def test_non_200_status_prints_warning(monkeypatch, capsys):
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/x/y/z")
    mock_resp = MagicMock()
    mock_resp.status = 500
    with patch("drift.notifications.slack.urllib.request.urlopen") as urlopen:
        urlopen.return_value.__enter__.return_value = mock_resp
        notify_drift(**_BASE_KWARGS)

    out = capsys.readouterr().out
    assert "Slack returned status 500" in out
