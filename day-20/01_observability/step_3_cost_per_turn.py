from typing import Any, cast
from opentelemetry import trace
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from agent_framework.observability import configure_otel_providers

SPANS = InMemorySpanExporter()
configure_otel_providers(exporters=[SPANS])