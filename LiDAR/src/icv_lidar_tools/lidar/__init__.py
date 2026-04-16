from .pcap_parser_protocol import ProtocolPcapParser, MsopDecoder, DecodedMsopPacket

# Backward-compatible alias: current implementation is protocol-based.

__all__ = [
    "ProtocolPcapParser",
    "MsopDecoder",
    "DecodedMsopPacket",
]
