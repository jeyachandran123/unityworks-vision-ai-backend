"""Adapters — the ONLY place external technology may appear.

Everything on the platform side of a port is deterministic and stdlib-only.
Everything vendor-, codec-, device- or framework-specific lives here, behind a
port, and is verified by that port's conformance kit before it may be activated
(invariant V3).

Flow 1 ships reference adapters that are dependency-free and therefore usable in
CI without a network, a codec library, or a camera:

    memory/         P7  HostMemoryPool
    acquisition/    P1  InMemoryRawSource       P2  PassthroughDecoder
                    P3  NoMaskPolicy · StaticZoneMask · FailingMask
                    P4  ArrivalTime · WallclockHint · Pts · Unknown ClockSync
                        JsonFileEpochStore (M2 module-private persistence)
    scheduling/     P5  CadenceAdmissionPolicy · AdmitAll · ResolutionLadder
                    P6  NullChangeDetector · SampledDigestChangeDetector
    configuration/  P23 InMemoryConfigSource · JsonFileConfigSource
                    P24 InMemorySecretProvider · EnvironmentSecretProvider
    observability/  P29 NullEventTransport · RecordingEventTransport
                    P30 InMemoryMetricsExporter · OpenMetricsTextExporter

RTSP/WebRTC sources, NVDEC/QSV/VAAPI decoders, Kafka/NATS transports and
Prometheus scrape endpoints are siblings behind these same ports. Adding them
changes no platform module.
"""

from __future__ import annotations
