"""Regression checks for V2 headers and rejection before output replacement."""
from pathlib import Path
import hashlib
import importlib.util
import struct
import sys
import tempfile
import zlib

import onnx


def main():
    """Export both frozen candidates and reject changed metadata and unknown bytes."""
    root = Path(__file__).resolve().parent
    tools = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else root / 'tools'
    sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location('converter_under_test', tools / 'convert_policy.py')
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    from v2_contract import info
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / 'policy.mdp'
        for name, runtime in [('rise-4000', 0x834335F5), ('walk-4600', 0x11D720E4)]:
            source = root / name / 'policy.onnx'
            converter.export(source, target)
            binary = target.read_bytes()
            assert binary == (root / name / (name + '.mdp')).read_bytes()
            assert info(binary)['policy_runtime_id'] == runtime
            assert struct.unpack_from('<I', binary, 8)[0] == 2
            assert struct.unpack_from('<I', binary, 28)[0] == zlib.crc32(binary[128:])
            assert binary[52:84] == hashlib.sha256(source.read_bytes()).digest()
            for change in ('metadata', 'unknown_candidate'):
                model = onnx.load(source)
                if change == 'metadata':
                    next(p for p in model.metadata_props if p.key == 'action_scale').value = '0.5'
                else:
                    model.doc_string += 'unregistered candidate'
                previous_json = target.with_suffix('.json').read_bytes()
                try:
                    converter.export(source, target, source_bytes=model.SerializeToString())
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'{change} was not rejected')
                assert target.read_bytes() == binary
                assert target.with_suffix('.json').read_bytes() == previous_json
            print(name, 'V2 export exact; invalid candidates rejected without overwriting outputs')


if __name__ == '__main__':
    main()
