"""Dependency-free delivered-file hash, V2 header, source SHA and CRC checks."""
from pathlib import Path
import hashlib
import json
import struct
import zlib

root = Path(__file__).resolve().parent
manifest = json.loads((root/'manifest.json').read_text())
for row in manifest['files']:
    data = (root/row['path']).read_bytes()
    assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'], row['path']
for row in manifest['models']:
    data = (root/row['upload']).read_bytes()
    header = struct.unpack('<8s6I5I32s44s',data[:128])
    assert header[:5]==(b'MDPOL001',2,61,14,4), row['id']
    assert header[5]*4==len(data)-128 and header[6]==zlib.crc32(data[128:]), row['id']
    assert header[-2]==hashlib.sha256((root/row['onnx']).read_bytes()).digest(), row['id']
    runtime, scale, slew, contract = struct.unpack('<Iff32s',header[-1])
    assert runtime==0x834335F5 and scale==1 and slew==0 and any(contract), row['id']
print(f"PASS: {len(manifest['files'])} files; {len(manifest['models'])} V2 rise models")
