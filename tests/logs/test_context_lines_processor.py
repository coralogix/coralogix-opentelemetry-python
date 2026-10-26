import inspect
import json
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from opentelemetry.sdk._logs import Logger, LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
    SimpleLogRecordProcessor,
)

from coralogix_opentelemetry.logs import (
    ATTR_EXCEPTION_CONTEXT_LINES,
    ContextLinesProcessor,
)
from coralogix_opentelemetry.logs import context_lines_processor as module

SOURCE = "\n".join(f"line_{n} = {n}" for n in range(1, 31)) + "\n"


def frame_line(path: Any, lineno: int, function: str = "handler") -> str:
    return f'  File "{path}", line {lineno}, in {function}\n    some_source()\n'


def traceback_text(*frames: str) -> str:
    return (
        "Traceback (most recent call last):\n" + "".join(frames) + "ValueError: boom\n"
    )


@pytest.fixture
def source_file(tmp_path: Path) -> Path:
    path = tmp_path / "app.py"
    path.write_text(SOURCE)
    return path


def export(stacktrace: Optional[str], **attributes: Any) -> Dict[str, Any]:
    exporter = InMemoryLogExporter()  # type: ignore[no-untyped-call]
    provider = LoggerProvider()
    provider.add_log_record_processor(ContextLinesProcessor())
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    if stacktrace is not None:
        attributes["exception.stacktrace"] = stacktrace
    provider.get_logger("test").emit(**emit_kwargs(attributes))
    (finished,) = exporter.get_finished_logs()
    return dict(finished.log_record.attributes or {})


def emit_kwargs(attributes: Dict[str, Any]) -> Dict[str, Any]:
    # Logger.emit only takes a LogRecord up to SDK 1.38; later versions add keyword fields
    if "body" not in inspect.signature(Logger.emit).parameters:
        from opentelemetry.sdk._logs import LogRecord

        return {"record": LogRecord(body="err", attributes=attributes)}
    return {"body": "err", "attributes": attributes}


def frames_of(attributes: Dict[str, Any]) -> List[Dict[str, Any]]:
    decoded: List[Dict[str, Any]] = json.loads(attributes[ATTR_EXCEPTION_CONTEXT_LINES])
    return decoded


def test_adds_context_for_in_app_frame(source_file: Path) -> None:
    (frame,) = frames_of(export(traceback_text(frame_line(source_file, 10))))

    assert list(frame) == [
        "index",
        "filename",
        "function",
        "lineno",
        "in_app",
        "pre_context",
        "context_line",
        "post_context",
    ]
    assert frame["index"] == 0
    assert frame["filename"] == str(source_file)
    assert frame["function"] == "handler"
    assert frame["lineno"] == 10
    assert frame["in_app"] is True
    assert frame["context_line"] == "line_10 = 10"
    assert frame["pre_context"] == [f"line_{n} = {n}" for n in range(3, 10)]
    assert frame["post_context"] == [f"line_{n} = {n}" for n in range(11, 18)]


def test_context_is_clipped_at_file_edges(source_file: Path) -> None:
    (frame,) = frames_of(export(traceback_text(frame_line(source_file, 2))))

    assert frame["pre_context"] == ["line_1 = 1"]
    (last,) = frames_of(export(traceback_text(frame_line(source_file, 30))))
    assert last["post_context"] == []


def test_index_counts_vendor_frames_like_the_rust_classifier(source_file: Path) -> None:
    stack = traceback_text(
        frame_line("/venv/lib/python3.11/site-packages/flask/app.py", 5, "dispatch"),
        frame_line(source_file, 10, "handler"),
    )

    (frame,) = frames_of(export(stack))

    assert frame["index"] == 1


def test_index_matches_every_line_the_rust_frame_regex_matches(
    source_file: Path,
) -> None:
    # FRAME_LINE_RE also matches `at ...` and `.go:N` lines; the message line below
    # takes slot 0 so the Python frame is the second match
    stack = (
        "  at something odd\n"
        + frame_line(source_file, 10)
        + "ValueError: see cgo.go:12\n"
    )

    (frame,) = frames_of(export(stack))

    assert frame["index"] == 1


def test_index_spans_chained_exceptions(source_file: Path) -> None:
    stack = (
        traceback_text(frame_line(source_file, 3, "first"))
        + "\nThe above exception was the direct cause of the following exception:\n\n"
        + traceback_text(frame_line(source_file, 4, "second"))
    )

    frames = frames_of(export(stack))

    assert [(f["index"], f["function"]) for f in frames] == [
        (0, "first"),
        (1, "second"),
    ]


@pytest.mark.parametrize(
    "path",
    [
        "relative/app.py",
        "<frozen importlib._bootstrap>",
        "<string>",
        "/venv/lib/python3.11/site-packages/pkg/mod.py",
        "/usr/lib/python3/dist-packages/pkg/mod.py",
    ],
)
def test_skips_frames_that_are_not_in_app(path: str) -> None:
    assert ATTR_EXCEPTION_CONTEXT_LINES not in export(
        traceback_text(frame_line(path, 1))
    )


def test_skips_missing_unreadable_and_out_of_range_frames(
    tmp_path: Path, source_file: Path
) -> None:
    stack = traceback_text(
        frame_line(tmp_path / "gone.py", 1),
        frame_line(source_file, 999),
        frame_line(source_file, 5, "kept"),
    )

    (frame,) = frames_of(export(stack))

    assert (frame["index"], frame["function"]) == (2, "kept")


def test_skips_files_over_one_mib(tmp_path: Path) -> None:
    big = tmp_path / "big.py"
    big.write_text("x = 1\n" * (module.MAX_FILE_BYTES // 6 + 10))

    assert ATTR_EXCEPTION_CONTEXT_LINES not in export(
        traceback_text(frame_line(big, 1))
    )


def test_sets_nothing_without_a_string_stacktrace() -> None:
    assert ATTR_EXCEPTION_CONTEXT_LINES not in export(None, other="value")
    assert ATTR_EXCEPTION_CONTEXT_LINES not in export(
        None, **{"exception.stacktrace": 5}
    )


def test_long_lines_are_snipped_from_the_start(tmp_path: Path) -> None:
    path = tmp_path / "long.py"
    path.write_text("short\n" + "y" * 400 + "\nshort\n")

    (frame,) = frames_of(export(traceback_text(frame_line(path, 2))))

    assert frame["context_line"] == "y" * 140 + " {snip}"
    assert frame["pre_context"] == ["short"]


def test_line_at_the_limit_is_not_snipped(tmp_path: Path) -> None:
    path = tmp_path / "edge.py"
    path.write_text("z" * 150 + "\n")

    (frame,) = frames_of(export(traceback_text(frame_line(path, 1))))

    assert frame["context_line"] == "z" * 150


def test_keeps_at_most_fifty_innermost_frames(source_file: Path) -> None:
    stack = traceback_text(*(frame_line(source_file, 5, f"f{n}") for n in range(60)))

    frames = frames_of(export(stack))

    assert [f["index"] for f in frames] == list(range(10, 60))


def test_drops_outermost_frames_until_it_fits_in_32_kib(tmp_path: Path) -> None:
    path = tmp_path / "wide.py"
    path.write_text("\n".join("w" * 140 for _ in range(30)) + "\n")
    stack = traceback_text(*(frame_line(path, 15, f"f{n}") for n in range(40)))

    attributes = export(stack)

    assert (
        len(attributes[ATTR_EXCEPTION_CONTEXT_LINES].encode())
        <= module.MAX_ATTRIBUTE_BYTES
    )
    frames = frames_of(attributes)
    assert 0 < len(frames) < 40
    assert frames[-1]["index"] == 39


def test_output_is_compact_json(source_file: Path) -> None:
    encoded = export(traceback_text(frame_line(source_file, 3)))[
        ATTR_EXCEPTION_CONTEXT_LINES
    ]

    assert ", " not in encoded.split('"context_line"')[0]
    assert ": " not in encoded.split('"context_line"')[0]


def test_real_traceback_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / "raiser.py"
    path.write_text("def boom():\n    raise ValueError('real')\n")
    namespace: Dict[str, Any] = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    try:
        namespace["boom"]()
    except ValueError:
        stack = traceback.format_exc()

    frames = frames_of(export(stack))

    assert frames[-1]["function"] == "boom"
    assert frames[-1]["context_line"] == "    raise ValueError('real')"


def test_never_raises(monkeypatch: pytest.MonkeyPatch, source_file: Path) -> None:
    def explode(_: str) -> None:
        raise RuntimeError("nope")

    monkeypatch.setattr(module, "_context_lines_for", explode)

    assert ATTR_EXCEPTION_CONTEXT_LINES not in export(
        traceback_text(frame_line(source_file, 3))
    )


def test_tolerates_foreign_record_shapes() -> None:
    processor = ContextLinesProcessor()

    processor.on_emit(object())
    processor.emit(None)
    assert processor.force_flush() is True
    processor.shutdown()
