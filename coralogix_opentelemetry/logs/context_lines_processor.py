import json
import linecache
import os
import re
from typing import Any, Dict, List, NamedTuple, Optional

from opentelemetry.sdk._logs import LogRecordProcessor

ATTR_EXCEPTION_STACKTRACE = "exception.stacktrace"
ATTR_EXCEPTION_CONTEXT_LINES = "cx.exception.context_lines"

LINES_OF_CONTEXT = 7
MAX_FRAMES = 50
MAX_FILE_BYTES = 1024 * 1024
MAX_LINE_LENGTH = 150
MAX_ATTRIBUTE_BYTES = 32 * 1024

# Same pattern the error-template-service uses to number frames (FRAME_LINE_RE in
# fingerprinter/default.rs). `index` must count exactly the lines it matches, including
# non-Python alternatives, or context lines attach to the wrong frame.
_FRAME_LINE = re.compile(r'^\s*at\s|^\s*File\s+"[^"]*",\s*line\s+\d+|\.go:\d+')
_PYTHON_FRAME = re.compile(
    r'^\s*File\s+"([^"]*)",\s*line\s+(\d+)(?:,\s*in\s+(.+?))?\s*$'
)
_THIRD_PARTY_DIRS = ("site-packages", "dist-packages")


class _StackFrame(NamedTuple):
    frame_index: int
    filename: str
    function: str
    lineno: int


def _frame_lines(stack: str) -> List[str]:
    # str.splitlines would also break on \x0b, \x0c, \x85 and  , which Rust's
    # str::lines does not
    lines = (line[:-1] if line.endswith("\r") else line for line in stack.split("\n"))
    return [line for line in lines if _FRAME_LINE.search(line)]


def _parse_python_frames(stack: str) -> List[_StackFrame]:
    frames = []
    for index, line in enumerate(_frame_lines(stack)):
        match = _PYTHON_FRAME.match(line)
        if match:
            filename, lineno, function = match.groups()
            frames.append(_StackFrame(index, filename, function or "?", int(lineno)))
    return frames


def _is_in_app_file(filename: str) -> bool:
    if not os.path.isabs(filename):
        return False
    parts = filename.replace("\\", "/").split("/")
    return not any(part in _THIRD_PARTY_DIRS for part in parts)


def _source_lines(filename: str) -> Optional[List[str]]:
    try:
        if os.stat(filename).st_size > MAX_FILE_BYTES:
            return None
        linecache.checkcache(filename)
        return [line.rstrip("\r\n") for line in linecache.getlines(filename)]
    except OSError:
        return None


def _snip_line(line: str) -> str:
    if len(line) <= MAX_LINE_LENGTH:
        return line
    window = MAX_LINE_LENGTH - 10
    return line[:window] + " {snip}"


def _to_frame_context(frame: _StackFrame) -> Optional[Dict[str, Any]]:
    lines = _source_lines(frame.filename)
    line_index = frame.lineno - 1
    if lines is None or not 0 <= line_index < len(lines):
        return None
    return {
        "index": frame.frame_index,
        "filename": frame.filename,
        "function": frame.function,
        "lineno": frame.lineno,
        "in_app": True,
        "pre_context": [
            _snip_line(line)
            for line in lines[max(0, line_index - LINES_OF_CONTEXT) : line_index]
        ],
        "context_line": _snip_line(lines[line_index]),
        "post_context": [
            _snip_line(line)
            for line in lines[line_index + 1 : line_index + 1 + LINES_OF_CONTEXT]
        ],
    }


def _serialize_within_budget(frames: List[Dict[str, Any]]) -> Optional[str]:
    # Python prints the innermost frame last, so shed the outermost ones first
    for first in range(len(frames)):
        encoded = json.dumps(frames[first:], separators=(",", ":"), ensure_ascii=False)
        if len(encoded.encode("utf-8")) <= MAX_ATTRIBUTE_BYTES:
            return encoded
    return None


def _context_lines_for(stack: str) -> Optional[str]:
    in_app = [
        frame
        for frame in _parse_python_frames(stack)
        if _is_in_app_file(frame.filename)
    ]
    contexts = [
        context
        for context in map(_to_frame_context, in_app[-MAX_FRAMES:])
        if context is not None
    ]
    return _serialize_within_budget(contexts) if contexts else None


class ContextLinesProcessor(LogRecordProcessor):
    """Adds `cx.exception.context_lines` to log records carrying a Python `exception.stacktrace`.

    Register it before the exporting processor. Processors run in registration order:
    SimpleLogRecordProcessor exports inside its own call, so a later write is never
    exported, and BatchLogRecordProcessor queues the record for a worker thread that may
    export it before a later write lands.
    """

    # opentelemetry-sdk >= 1.39 calls on_emit(ReadWriteLogRecord); earlier versions
    # call emit(LogData). Both expose the mutable record as `.log_record`.
    def on_emit(self, log_record: Any) -> None:
        self._add_context_lines(log_record)

    def emit(self, log_data: Any) -> None:
        self._add_context_lines(log_data)

    def _add_context_lines(self, record: Any) -> None:
        # A processor exception escapes into the caller of Logger.emit
        try:
            attributes = record.log_record.attributes
            stack = attributes.get(ATTR_EXCEPTION_STACKTRACE)
            if not isinstance(stack, str):
                return
            encoded = _context_lines_for(stack)
            if encoded:
                attributes[ATTR_EXCEPTION_CONTEXT_LINES] = encoded
        except Exception:
            return

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True

    def shutdown(self) -> None:
        return None
