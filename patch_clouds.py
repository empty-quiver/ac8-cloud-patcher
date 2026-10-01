#!/usr/bin/env python3
"""Patch the observed AC8 SkyTraceCS half-subnormal sentinel conversion.

Python 3.9+, SPIRV-Tools (spirv-dis, spirv-val), LLVM llvm-bcanalyzer.
Dry run by default. Original dumps are never modified. No fixed shader hashes,
instruction IDs, byte offsets, or expected number of variants are embedded.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

SENTINEL = 0xb7000000  # Exact float32 bits for -1 / 131072.


class Refused(Exception):
    pass


def require(condition, message):
    if not condition:
        raise Refused(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def tool(name, supplied=None):
    choices = [supplied] if supplied else [name]
    if name == 'llvm-bcanalyzer' and not supplied:
        choices += ['llvm-bcanalyzer-' + str(n) for n in range(24, 13, -1)]
    for choice in choices:
        found = shutil.which(choice)
        if found:
            return str(Path(found).resolve())
    raise Refused('Missing tool: ' + (supplied or name))


def run(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Refused(str(exc)) from exc
    require(result.returncode == 0,
            Path(args[0]).name + ' failed: ' + (result.stderr or result.stdout).strip()[:1500])
    return result.stdout


def instructions(data):
    require(len(data) >= 20 and len(data) % 4 == 0, 'Invalid SPIR-V size')
    words = struct.unpack('<%dI' % (len(data) // 4), data)
    require(words[0] == 0x07230203, 'Not little-endian SPIR-V')
    result = []
    offset = 5
    while offset < len(words):
        count, opcode = words[offset] >> 16, words[offset] & 0xffff
        require(count > 0 and offset + count <= len(words), 'Malformed SPIR-V instruction')
        result.append((offset * 4, opcode, words[offset + 1:offset + count]))
        offset += count
    return result


def candidates(data):
    ins = instructions(data)
    floats = {a[0] for _, op, a in ins if op == 22 and len(a) == 2 and a[1] == 32}
    constants = {a[1]: a[0] for _, op, a in ins
                 if op == 43 and len(a) == 3 and a[0] in floats and a[2] == SENTINEL}
    return [(off, op, a) for off, op, a in ins
            if op in (116, 83) and len(a) == 3
            and a[2] in constants and a[0] == constants[a[2]]]


def shader_identity(dxil, filename_hash, llvm, temp):
    require(dxil.is_file(), 'Matching .dxil file is missing')
    b = dxil.read_bytes()
    # vkd3d-proton hashes input bytecode with 64-bit FNV-1, not FNV-1a.
    h = 0xcbf29ce484222325
    for byte in b:
        h = ((h * 0x100000001b3) & 0xffffffffffffffff) ^ byte
    require('%016x' % h == filename_hash, 'DXIL contents do not match the dump filename hash')
    require(len(b) >= 32 and b[:4] == b'DXBC', 'Invalid DXIL container')
    total, count = struct.unpack_from('<II', b, 24)
    require(total == len(b) and 32 + count * 4 <= len(b), 'Invalid DXIL container bounds')
    debug = []
    for offset in struct.unpack_from('<%dI' % count, b, 32):
        require(offset + 8 <= len(b), 'Invalid DXIL chunk offset')
        tag, size = struct.unpack_from('<4sI', b, offset)
        require(offset + 8 + size <= len(b), 'Invalid DXIL chunk size')
        if tag == b'ILDB':
            debug.append(b[offset + 8:offset + 8 + size])
    require(len(debug) == 1, 'Expected one ILDB debug chunk; stripped shaders are unsupported')
    chunk = debug[0]
    require(len(chunk) >= 24 and chunk[8:12] == b'DXIL', 'Unsupported ILDB program header')
    offset, size = struct.unpack_from('<II', chunk, 16)
    start = 8 + offset
    require(start + size <= len(chunk) and chunk[start:start + 4] == b'BC\xc0\xde',
            'Invalid debug bitcode range')
    bc = temp / 'debug.bc'
    bc.write_bytes(chunk[start:start + size])
    dump = run([llvm, '-dump', str(bc)])
    strings = [bytes(int(n) for n in re.findall(r'\bop\d+=(\d+)', record))
               .decode('utf-8', errors='replace')
               for record in re.findall(r'<STRING_OLD\b([^>]*)/>', dump)]
    require('SkyTraceCS' in strings, 'Debug metadata does not identify SkyTraceCS')
    sources = [s for s in strings if 'TraceRangeSetup' in s and 'cbuffer' in s and len(s) > 1000]
    require(len(sources) == 1, 'Cannot identify a unique embedded HLSL source')
    source = re.sub(r'\s+', '', sources[0])
    for snippet in ('Pack(ret1.packing_cirrus_range,float2(-1,-1));',
                    'Pack(ret1.packing_tracable_cloud_range,float2(-1,-1));',
                    'Pack_16_16(first*(1.0f/131072.f))'):
        require(snippet in source, 'Expected source sentinel/packing pattern is missing')
    return sha256(b)


def verify_use_chain(assembly, args):
    """Use disassembler ID tokens, never guess whether binary operands are IDs."""
    definitions, refs = {}, {}
    entrypoints = []
    for line in assembly.splitlines():
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        match = re.fullmatch(r'(%\d+)\s*=\s*(Op\w+)\s*(.*)', line)
        if match:
            ident, op, tail = match.groups()
            definitions[ident] = (op, tail.split())
        else:
            op, _, tail = line.partition(' ')
            ident = None
        if op == 'OpEntryPoint':
            entrypoints.append(tail.split()[0])
        # Annotations do not change the arithmetic data flow.
        if op in ('OpName', 'OpDecorate'):
            continue
        for used in set(re.findall(r'%\d+', tail)):
            refs.setdefault(used, []).append((ident, op, tail.split()))
    require(entrypoints == ['GLCompute'], 'Expected a single compute entrypoint')
    typ, result, constant = ('%' + str(x) for x in args)
    require(definitions.get(typ) == ('OpTypeFloat', ['32']), 'Expected float32')
    consumers = refs.get(result, [])
    require(len(consumers) == 1, 'Sentinel conversion has unexpected additional uses')
    vector, op, a = consumers[0]
    require(op == 'OpCompositeConstruct' and len(a) == 3 and a[1] == result,
            'Expected sentinel in the first lane of a two-component vector')
    require(definitions.get(a[0]) == ('OpTypeVector', [typ, '2']), 'Expected float2')
    zero = definitions.get(a[2])
    require(zero in (('OpConstant', [typ, '0']), ('OpConstant', [typ, '-0'])),
            'Expected zero in the second packing lane')
    consumers = refs.get(vector, [])
    require(len(consumers) == 1, 'Packing vector has unexpected additional uses')
    _, op, a = consumers[0]
    require(op == 'OpExtInst' and len(a) == 4 and a[2:] == ['PackHalf2x16', vector],
            'Expected GLSL PackHalf2x16 to consume the sentinel vector')
    require(definitions.get(a[1]) == ('OpExtInstImport', ['"GLSL.std.450"']),
            'Unexpected extended instruction set')


def inspect(path, found, tools, target_env, temp):
    require(re.fullmatch(r'[0-9a-f]{16}', path.stem) is not None,
            'Expected an unmodified vkd3d 16-digit hash filename')
    require(len(found) == 1, 'Multiple matching sentinel conversions; refusing ambiguous shader')
    off, op, operands = found[0]
    dxil_sha = shader_identity(path.with_suffix('.dxil'), path.stem, tools['llvm'], temp)
    run([tools['val'], '--target-env', target_env, str(path)])
    assembly = run([tools['dis'], '--raw-id', str(path)])
    verify_use_chain(assembly, operands)
    original = path.read_bytes()
    row = dict(shader_hash=path.stem, entrypoint='SkyTraceCS', bytes=len(original),
               original_sha256=sha256(original), dxil_sha256=dxil_sha,
               byte_offset=off, result_id=operands[1], constant_id=operands[2])
    if op == 83:
        row['status'] = 'already-patched-pattern'
        return row, None
    patched = bytearray(original)
    require(patched[off:off+4] == bytes.fromhex('74 00 04 00'), 'Unexpected instruction encoding')
    patched[off] = 0x53
    check = temp / 'patched.spv'
    check.write_bytes(patched)
    run([tools['val'], '--target-env', target_env, str(check)])
    row.update(status='patchable', patched_sha256=sha256(patched), changed_bytes=1,
               original_and_patched_validation='pass')
    return row, bytes(patched)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dump', type=Path, help='Directory of locally dumped .spv/.dxil pairs')
    parser.add_argument('--apply', action='store_true', help='Write validated patches (default: dry run)')
    parser.add_argument('--output', type=Path, help='New output directory; must not already exist')
    parser.add_argument('--report', type=Path, help='Optional new JSON report file')
    parser.add_argument('--target-env', default='vulkan1.3', help='spirv-val target (default: vulkan1.3)')
    parser.add_argument('--spirv-dis')
    parser.add_argument('--spirv-val')
    parser.add_argument('--llvm-bcanalyzer')
    args = parser.parse_args(argv)
    try:
        require(args.dump.is_dir(), 'Dump directory does not exist')
        require(not args.apply or args.output is not None, '--apply requires --output')
        require(args.apply or args.output is None, '--output requires --apply')
        require(args.output is None or not args.output.exists(), 'Output directory already exists; use a new one')
        require(args.report is None or not args.report.exists(), 'Report file already exists; use a new one')
        tools = {'dis': tool('spirv-dis', args.spirv_dis), 'val': tool('spirv-val', args.spirv_val),
                 'llvm': tool('llvm-bcanalyzer', args.llvm_bcanalyzer)}
        report = dict(format_version=1, mode='apply' if args.apply else 'dry-run',
                      target_env=args.target_env, dump_directory=str(args.dump.resolve()),
                      tools=tools, tool_versions={k: run([v, '--version']).splitlines()[0] for k, v in tools.items()},
                      scanned=0, ignored=0, shaders=[])
        pending = []
        with tempfile.TemporaryDirectory(prefix='ac8-cloud-patcher-') as temp:
            temp = Path(temp)
            for path in sorted(args.dump.glob('*.spv')):
                report['scanned'] += 1
                try:
                    b = path.read_bytes()
                    if len(b) < 20 or len(b) % 4 or b[:4] != b'\x03\x02\x23\x07':
                        row = dict(shader_hash=path.stem, status='skipped-invalid-header',
                                   reason='Not a complete little-endian SPIR-V module; left untouched')
                        report['shaders'].append(row)
                        print(path.name + ': ' + row['reason'], flush=True)
                        continue
                    # Fast filter; avoids disassembling thousands of unrelated shaders.
                    if struct.pack('<I', SENTINEL) not in b:
                        report['ignored'] += 1
                        continue
                    found = candidates(b)
                    if not found:
                        report['ignored'] += 1
                        continue
                    row, patched = inspect(path, found, tools, args.target_env, temp)
                    if patched is not None:
                        pending.append((path.name, patched))
                except (Refused, OSError, ValueError, struct.error) as exc:
                    row = dict(shader_hash=path.stem, status='refused', reason=str(exc))
                report['shaders'].append(row)
                print(path.stem + ': ' + row['status'] + (': ' + row['reason'] if 'reason' in row else ''), flush=True)
        counts = Counter(r['status'] for r in report['shaders'])
        report['counts'] = dict(counts)
        report['written'] = 0
        if args.apply and not counts['refused'] and pending:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='.ac8-cloud-patches-', dir=args.output.parent) as staging:
                stage = Path(staging) / 'complete'
                stage.mkdir()
                for name, data in pending:
                    (stage / name).write_bytes(data)
                report['written'] = len(pending)
                report['output_directory'] = str(args.output.resolve())
                (stage / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
                # Claim the destination exclusively; never replace an existing directory.
                args.output.mkdir()
                for file in stage.iterdir():
                    os.replace(file, args.output / file.name)
        if args.report:
            with args.report.open('x') as f:
                json.dump(report, f, indent=2)
                f.write('\n')
        print('Scanned {scanned}; ignored {ignored}; {counts}; wrote {written}.'.format(**report))
        if counts['refused']:
            print('Refused candidates: no patch directory was written.', file=sys.stderr)
            return 2
        if not counts['patchable'] and not counts['already-patched-pattern']:
            print('No supported sentinel pattern found. This does not prove the game is fixed.', file=sys.stderr)
            return 1
        return 0
    except (Refused, OSError, ValueError) as exc:
        print('Error: ' + str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
