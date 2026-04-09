#!/usr/bin/env python3
"""Lightweight OTLP HTTP/JSON receiver that extracts Claude Code token usage.

Listens for OpenTelemetry metrics and log signals, extracts token counts,
and writes them to a usage log file.

Usage:
    pixi run python otel_collector.py --tag mar25

Requires these env vars when launching Claude Code:
    CLAUDE_CODE_ENABLE_TELEMETRY=1
    OTEL_METRICS_EXPORTER=otlp
    OTEL_LOGS_EXPORTER=otlp
    OTEL_EXPORTER_OTLP_PROTOCOL=http/json
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
"""

import argparse
import atexit
import json
import signal
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

TOKEN_USAGE_METRIC = "claude_code.token.usage"
API_REQUEST_EVENT = "claude_code.api_request"

TOKEN_ATTR_KEYS = {
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
}


def get_attr_value(attr: dict):
    """Extract a scalar value from an OTLP JSON attribute value."""
    value = attr.get("value", {})
    for field in ("intValue", "stringValue", "doubleValue", "boolValue"):
        if field in value:
            return value[field]
    return None


def attrs_to_dict(attributes: list) -> dict:
    """Convert OTLP attribute list to a plain dict."""
    return {a["key"]: get_attr_value(a) for a in attributes}


class UsageCollector:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
        }
        self.api_calls = 0
        with open(self.log_path, "w") as f:
            f.write(f"# OTEL Token Usage Log — started {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"# Format: timestamp\tevent\tmodel\tinput\toutput\tcache_read\tcache_create\n\n")

    def record_api_request(self, attrs: dict):
        """Record token usage from a claude_code.api_request log event."""
        model = attrs.get("model", "unknown")
        input_t = _int(attrs.get("input_tokens", 0))
        output_t = _int(attrs.get("output_tokens", 0))
        cache_read = _int(attrs.get("cache_read_tokens", 0))
        cache_create = _int(attrs.get("cache_creation_tokens", 0))

        self.totals["input_tokens"] += input_t
        self.totals["output_tokens"] += output_t
        self.totals["cache_read_tokens"] += cache_read
        self.totals["cache_creation_tokens"] += cache_create
        self.api_calls += 1

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(self.log_path, "a") as f:
            f.write(f"{ts}\tapi_request\t{model}\t{input_t}\t{output_t}\t{cache_read}\t{cache_create}\n")

    def record_metric(self, metric_name: str, value: int, attrs: dict):
        """Record token usage from a claude_code.token.usage metric data point."""
        token_type = attrs.get("type", "unknown")
        model = attrs.get("model", "unknown")

        key_map = {
            "input": "input_tokens",
            "output": "output_tokens",
            "cacheRead": "cache_read_tokens",
            "cacheCreation": "cache_creation_tokens",
        }
        key = key_map.get(token_type)
        if key:
            self.totals[key] += value

        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(self.log_path, "a") as f:
            f.write(f"{ts}\tmetric\t{model}\t{token_type}={value}\n")

    def write_summary(self):
        with open(self.log_path, "a") as f:
            f.write(f"\n# ===== Session Summary =====\n")
            f.write(f"# Completed: {datetime.now(timezone.utc).isoformat()}\n")
            f.write(f"# API calls: {self.api_calls}\n")
            for k, v in self.totals.items():
                f.write(f"# {k}: {v:,}\n")
            total = sum(self.totals.values())
            f.write(f"# total_tokens: {total:,}\n")
        print(f"\nSession summary written to {self.log_path}")
        for k, v in self.totals.items():
            print(f"  {k}: {v:,}")
        print(f"  total: {sum(self.totals.values()):,}")


def _int(v) -> int:
    try:
        return int(v)
    except (ValueError, TypeError):
        return 0


def process_metrics(data: dict, collector: UsageCollector):
    """Process an OTLP JSON metrics payload."""
    for rm in data.get("resourceMetrics", []):
        for sm in rm.get("scopeMetrics", []):
            for metric in sm.get("metrics", []):
                name = metric.get("name", "")
                if name != TOKEN_USAGE_METRIC:
                    continue
                # Handle sum, gauge, or histogram
                for container_key in ("sum", "gauge"):
                    container = metric.get(container_key, {})
                    for dp in container.get("dataPoints", []):
                        value = _int(dp.get("asInt", dp.get("asDouble", 0)))
                        attrs = attrs_to_dict(dp.get("attributes", []))
                        collector.record_metric(name, value, attrs)


def process_logs(data: dict, collector: UsageCollector):
    """Process an OTLP JSON logs payload."""
    for rl in data.get("resourceLogs", []):
        for sl in rl.get("scopeLogs", []):
            for record in sl.get("logRecords", []):
                body = record.get("body", {})
                event_name = body.get("stringValue", "")
                # Also check attributes for event.name
                attrs = attrs_to_dict(record.get("attributes", []))
                if event_name == API_REQUEST_EVENT or attrs.get("event.name") == API_REQUEST_EVENT:
                    collector.record_api_request(attrs)


def process_traces(data: dict, collector: UsageCollector):
    """Process an OTLP JSON traces payload — extract any token attributes from spans."""
    for rs in data.get("resourceSpans", []):
        for ss in rs.get("scopeSpans", []):
            for span in ss.get("spans", []):
                attrs = attrs_to_dict(span.get("attributes", []))
                token_attrs = {k: _int(v) for k, v in attrs.items() if k in TOKEN_ATTR_KEYS}
                if token_attrs:
                    model = attrs.get("model", attrs.get("gen_ai.response.model", "unknown"))
                    token_attrs["model"] = model
                    collector.record_api_request(token_attrs)


class OTLPHandler(BaseHTTPRequestHandler):
    collector: UsageCollector = None

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            data = json.loads(body)
            path = self.path.rstrip("/")
            if path == "/v1/metrics":
                process_metrics(data, self.collector)
            elif path == "/v1/logs":
                process_logs(data, self.collector)
            elif path == "/v1/traces":
                process_traces(data, self.collector)
        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: failed to process {self.path}: {e}", file=sys.stderr)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, format, *args):
        pass  # Suppress default request logging


def main():
    parser = argparse.ArgumentParser(description="OTLP HTTP/JSON receiver for Claude Code token usage")
    parser.add_argument("--tag", required=True, help="Experiment tag (used in log filename)")
    parser.add_argument("--port", type=int, default=4318, help="Port to listen on (default: 4318)")
    args = parser.parse_args()

    log_path = Path("results") / f"usage-{args.tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    collector = UsageCollector(log_path)
    OTLPHandler.collector = collector

    # Write summary on exit
    atexit.register(collector.write_summary)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    server = HTTPServer(("localhost", args.port), OTLPHandler)
    print(f"OTLP collector listening on http://localhost:{args.port}")
    print(f"Writing token usage to {log_path}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
