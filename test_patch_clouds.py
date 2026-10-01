#!/usr/bin/env python3
"""Integration tests using a user's own, unmodified AC8 shader dump.

Usage: python3 test_patch_clouds.py /path/to/dump
Dependencies: tools used by patch_clouds.py, plus spirv-as on PATH.
No game shader binaries are included in this package.
"""
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import patch_clouds as patch

DUMP = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else None
SCRIPT = Path(__file__).with_name('patch_clouds.py')


class PatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if DUMP is None or not DUMP.is_dir():
            raise RuntimeError('Pass an original AC8 shader dump directory')
        cls.fixture = next((p for p in sorted(DUMP.glob('*.spv'))
                            if p.with_suffix('.dxil').exists()
                            and patch.candidates(p.read_bytes())), None)
        if cls.fixture is None:
            raise RuntimeError('No candidate fixture found')
        cls.llvm = patch.tool('llvm-bcanalyzer')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ac8-patcher-test-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.dump = self.base / 'dump'
        self.dump.mkdir()
        for ext in ('.spv', '.dxil'):
            shutil.copy2(self.fixture.with_suffix(ext), self.dump / (self.fixture.stem + ext))
        self.shader = self.dump / self.fixture.name

    def cli(self, *extra, expected=0):
        p = subprocess.run([sys.executable, str(SCRIPT), str(self.dump),
                            '--llvm-bcanalyzer', self.llvm, *map(str, extra)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, expected, p.stdout + p.stderr)
        return p.stdout + p.stderr

    def test_dry_run_preserves_files(self):
        before = {p.name: patch.sha256(p.read_bytes()) for p in self.dump.iterdir()}
        text = self.cli()
        self.assertIn('patchable', text)
        self.assertEqual(before, {p.name: patch.sha256(p.read_bytes()) for p in self.dump.iterdir()})
        self.assertEqual(sorted(p.name for p in self.base.iterdir()), ['dump'])

    def test_exact_one_byte_and_refuse_existing_output(self):
        out = self.base / 'patched'
        original = self.shader.read_bytes()
        self.cli('--apply', '--output', out)
        replacement = (out / self.shader.name).read_bytes()
        differences = [i for i, (a, b) in enumerate(zip(original, replacement)) if a != b]
        self.assertEqual(differences, [patch.candidates(original)[0][0]])
        self.assertEqual((original[differences[0]], replacement[differences[0]]), (0x74, 0x53))
        self.assertEqual(self.shader.read_bytes(), original)
        self.cli('--apply', '--output', out, expected=2)
        self.assertEqual((out / self.shader.name).read_bytes(), replacement)

    def test_already_patched_is_not_rewritten(self):
        b = bytearray(self.shader.read_bytes())
        b[patch.candidates(b)[0][0]] = 83
        self.shader.write_bytes(b)
        out = self.base / 'patched'
        self.assertIn('already-patched-pattern', self.cli('--apply', '--output', out))
        self.assertFalse(out.exists())

    def test_missing_dxil_refused(self):
        self.shader.with_suffix('.dxil').unlink()
        out = self.base / 'patched'
        self.assertIn('missing', self.cli('--apply', '--output', out, expected=2))
        self.assertFalse(out.exists())

    def test_mismatched_dxil_hash_refused(self):
        p = self.shader.with_suffix('.dxil')
        b = bytearray(p.read_bytes())
        b[4] ^= 1
        p.write_bytes(b)
        self.assertIn('filename hash', self.cli(expected=2))

    def disassemble(self):
        return patch.run([patch.tool('spirv-dis'), '--raw-id', str(self.shader)])

    def assemble(self, text):
        source = self.base / 'test.spvasm'
        source.write_text(text)
        # Preserve the module version; targeting Vulkan alone can upgrade it
        # and change entrypoint-interface requirements unrelated to this test.
        version = struct.unpack_from('<I', self.shader.read_bytes(), 4)[0]
        target = 'spv%d.%d' % ((version >> 16) & 255, (version >> 8) & 255)
        patch.run([patch.tool('spirv-as'), '--target-env', target,
                   '--preserve-numeric-ids', str(source), '-o', str(self.shader)])
        patch.run([patch.tool('spirv-val'), '--target-env', 'vulkan1.3', str(self.shader)])

    def test_instruction_ids_and_offsets_can_change(self):
        old_offset, _, args = patch.candidates(self.shader.read_bytes())[0]
        old_result = args[1]
        text = self.disassemble()
        fresh = max(map(int, re.findall(r'%(\d+)', text))) + 1
        text = re.sub(r'%' + str(old_result) + r'\b', '%' + str(fresh), text)
        text = text.replace('%' + str(fresh) + ' = OpQuantizeToF16',
                            'OpNop\n%' + str(fresh) + ' = OpQuantizeToF16')
        self.assemble(text)
        off, _, new_args = patch.candidates(self.shader.read_bytes())[0]
        self.assertNotEqual(off, old_offset)
        self.assertEqual(new_args[1], fresh)
        out = self.base / 'patched'
        self.cli('--apply', '--output', out)
        row = json.loads((out / 'manifest.json').read_text())['shaders'][0]
        self.assertEqual(row['result_id'], fresh)
        self.assertEqual(row['byte_offset'], off)

    def test_extra_data_flow_use_refused(self):
        _, _, (typ, result, _) = patch.candidates(self.shader.read_bytes())[0]
        text = self.disassemble()
        fresh = max(map(int, re.findall(r'%(\d+)', text))) + 1
        line = next(l for l in text.splitlines() if re.search(r'%' + str(result) + r'\s*= OpQuantizeToF16', l))
        text = text.replace(line, line + '\n%%%d = OpCopyObject %%%d %%%d' % (fresh, typ, result))
        self.assemble(text)
        self.assertIn('unexpected additional uses', self.cli(expected=2))

    def test_duplicate_conversion_refused(self):
        _, _, (typ, result, constant) = patch.candidates(self.shader.read_bytes())[0]
        text = self.disassemble()
        fresh = max(map(int, re.findall(r'%(\d+)', text))) + 1
        line = next(l for l in text.splitlines() if re.search(r'%' + str(result) + r'\s*= OpQuantizeToF16', l))
        text = text.replace(line, line + '\n%%%d = OpQuantizeToF16 %%%d %%%d' % (fresh, typ, constant))
        self.assemble(text)
        self.assertIn('ambiguous', self.cli(expected=2))

    def test_unknown_hash_is_not_allowlisted(self):
        # Synthetic identity test: change the container checksum field only.
        # This is not claimed to be a runnable, recompiled game shader.
        d = bytearray(self.shader.with_suffix('.dxil').read_bytes())
        d[4] ^= 1
        h = 0xcbf29ce484222325
        for byte in d:
            h = ((h * 0x100000001b3) & 0xffffffffffffffff) ^ byte
        new = '%016x' % h
        self.shader.with_suffix('.dxil').unlink()
        (self.dump / (new + '.dxil')).write_bytes(d)
        self.shader.rename(self.dump / (new + '.spv'))
        self.assertIn(new + ': patchable', self.cli())

    def test_no_pattern_has_nonzero_exit(self):
        b = self.shader.read_bytes().replace(struct.pack('<I', patch.SENTINEL), struct.pack('<I', 0xb8000000))
        self.shader.write_bytes(b)
        self.assertIn('No supported sentinel pattern', self.cli(expected=1))

    def test_bad_instruction_length_refused(self):
        b = bytearray(self.shader.read_bytes())
        off = patch.candidates(b)[0][0]
        struct.pack_into('<I', b, off, 116)  # Zero word count.
        self.shader.write_bytes(b)
        self.assertIn('Malformed', self.cli(expected=2))

    def test_non_spirv_dump_skipped_and_reported(self):
        junk = self.dump / '0000000000000000.spv'
        junk.write_bytes(b'\0' * 80)
        report = self.base / 'report.json'
        self.cli('--report', report)
        data = json.loads(report.read_text())
        self.assertEqual(data['counts']['skipped-invalid-header'], 1)
        self.assertEqual(data['counts']['patchable'], 1)
        self.assertEqual(junk.read_bytes(), b'\0' * 80)


if __name__ == '__main__':
    unittest.main(verbosity=2)
