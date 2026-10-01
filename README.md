# AC8 cloud sentinel patcher

Using NVIDIA Nsight, with help from GPT-6-Astra in Codex, I traced this to a cloud ray-march stall in `SkyTraceCS` and have a targeted workaround. I've put the patcher here so someone else can apply the same workaround to shaders dumped on their own system.

<img width="2560" height="850" alt="Cloud rendering comparison before and after the sentinel workaround" src="https://github.com/user-attachments/assets/0b7cfc90-4916-43b9-a89d-136ba30cba61" />

The shader initializes invalid cloud/cirrus ranges to `(-1, -1)`, then packs them into half precision after dividing by 131072. The generated SPIR-V applies `OpQuantizeToF16` to `-1/131072`, a half-precision subnormal, which becomes negative zero. The invalid cirrus range consequently becomes `(-0, -0)`: a ray at distance zero passes the inclusive range check, gets a zero-length step, and exhausts its 128-iteration budget without rendering clouds.

<details>
<summary>Relevant HLSL source excerpts</summary>

These excerpts come from the embedded HLSL debug source. Unrelated code is omitted, formatting is adjusted, and comments are mine.

```hlsl
// Invalid cirrus-range sentinel:
Pack(ret1.packing_cirrus_range, float2(-1, -1));

// Range packing:
void Pack(inout Packing_float2_TO_uint_Remap1e5 pack, float2 first)
{
    pack.first_packed = Pack_16_16(first * (1.0f / 131072.f));
}

// Inclusive range test:
bool IsInsideRange(float2 range, float t)
{
    return (range.x <= t) && (t <= range.y);
}

// Cirrus membership uses that test:
Pack_Bitmask(
    trace_state.bitmask,
    IsInsideRange(Unpack(ret1.packing_cirrus_range), trace_state.t),
    2
);

// Later, the cirrus range limits the step:
if (IsInsideCirrus(trace_state))
{
    const float cirrus_trace_range_length =
        Unpack_Y(ret1.packing_cirrus_range) -
        Unpack_X(ret1.packing_cirrus_range);

    step_length = min(step_length, cirrus_trace_range_length / (3 + 1));
}
float next_stop = trace_state.t + step_length;
```

With the intended `(-1, -1)` sentinel, `t = 0` is outside the range. After conversion to `(-0, -0)`, it is inside, and the range-length clamp forces the step to zero.

</details>

For shader `e2ace7e3e8dbf87b`, my patch is:

```diff
-%498 = OpQuantizeToF16 %float %float_n7_62939453en06
+%498 = OpCopyObject   %float %float_n7_62939453en06
```

The patch keeps the half-precision packing. It removes a separate `OpQuantizeToF16` immediately before it, which turns the small negative sentinel into negative zero. With `OpCopyObject`, the original value reaches the packing instruction unchanged; in the Nvidia Nsight replay, it survives packing and unpacks back to `-1`.
The original HLSL packing helper already expresses a direct `f32tof16(v)` conversion. This workaround changes the generated SPIR-V; I haven’t established which part of the shader compilation path should receive the upstream fix.

The patch changes one byte in the 249,084-byte SPIR-V file, at offset `24096` (`0x5E20`):

```text
74 00 04 00 → 53 00 04 00
```

Original SHA-256 of the shader:
`6b6eda17573f21c74c2fe6466cb784d64c1c9b6227e7bd39fe5c64ddbce5c469`

I apply it using `VKD3D_SHADER_OVERRIDE`, placing the replacement at `<override directory>/e2ace7e3e8dbf87b.spv`. vkd3d-proton substitutes the hash-matched shader when creating the pipeline. No modified Proton build or game file edits are required.

On an RTX 4090 with Proton Experimental, this restored clouds below the horizon in both the same Nsight replay and live gameplay. Instrumented replay confirmed that the sentinel remained `-1` and the ray advanced normally. Depth tests, iteration limits, density evaluation, and compositing remain intact; the workaround does not disable ray tracing.

After another mission showed the symptom, I found **16 `SkyTraceCS` variants** with identical cloud-range setup source and extended the same one-byte correction to each. All original and patched modules pass `spirv-val --target-env vulkan1.3`. I verified the original variant in replay and gameplay; the additional variants still need gameplay verification.

<details>
<summary>The 16 shader hashes found so far</summary>

These are the **16 `SkyTraceCS` shader hashes** found with the faulty sentinel conversion:

```text
22fdcb556e490841
38f591d7629df2b6
43b1f2ca42fb23c0
59582be9428f0c60
713ad0f307870d6c
76d3b0fd2325587f
80f225e283cc4d72
87b42e91ceafed10
88ad084f83cf8ef9
aaf93c80eebd46d5
aff49440c4c76f41
c652fabba323d136
e2ace7e3e8dbf87b
e6afff4a21f36ae7
f00552bae8b97a19
fcef06642b47c3cd
```

Instruction IDs and byte offsets for the patch differ between variants.
This covers the variants of this shader I've found. The script below does not use this list as an allowlist.

</details>

This establishes the failure mechanism in the captured shader, and a workaround, but not a real upstream fix. I am at least able to play the game with the clouds working correctly though! I haven't tested every mission yet. These are the variants I've found and patched so far; coverage beyond them is still unverified.

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 |
| NVIDIA driver | 580.159.03 |
| Proton | Experimental 11.0-20260924 (`experimental-11.0-20260924-x86_64`) |
| Game | ACE COMBAT 8, Steam build 25201480, AppID `2288340` |
| Vulkan API supported by NVIDIA driver | 1.4.312 |
| Host Vulkan loader | 1.3.204 |
| vkd3d-proton | 3.1.0, build `7f0c30ad3c8f28d` |

## Using the patcher

The script looks for the shader's source identity and compiled instruction pattern,
so it does not depend on my exact shader hashes, instruction IDs, byte offsets,
or number of variants. It works on **shaders dumped on your own system**.

It defaults to a dry run. It never changes the source dump, game files, Proton
files, Steam settings, or graphics settings. It writes replacement shaders only
when given `--apply`, into a new directory you select. It makes no network requests
and does not require Nsight or RenderDoc. No game shaders are included here.

I tested the workaround on the setup above. That does not establish rendering
correctness on every driver or GPU. The checks are deliberately conservative;
the script refuses unsupported patterns rather than guessing.

```sh
git clone https://github.com/empty-quiver/ac8-cloud-patcher.git
cd ac8-cloud-patcher
```

## Requirements

- Python 3.9 or newer; no Python packages are required.
- Recent SPIRV-Tools: `spirv-dis` and `spirv-val` on `PATH`.
  Version **2025.3 was tested**. Ubuntu 22.04's 2022.1 package is too old for the
  NVIDIA capabilities in these dumps. Newer versions may also work.
- LLVM's `llvm-bcanalyzer`, used to identify the shader from embedded DXIL debug
  metadata. LLVM 14 and 20 were tested. Version-suffixed executable names such as
  `llvm-bcanalyzer-14` are detected automatically.

Check availability:

```sh
python3 --version
spirv-dis --version
spirv-val --version
llvm-bcanalyzer --version
```

Use your distribution's packages if sufficiently recent. For example,
`sudo apt install python3 llvm spirv-tools` installs these on Debian/Ubuntu,
but check the SPIRV-Tools version afterward. No root access is needed to run the
patcher.

If your SPIRV-Tools package is too old, a tested source-build route is:

```sh
# Requires git, Python, CMake, Ninja, and a C++ compiler.
git clone --depth 1 --branch v2025.3 https://github.com/KhronosGroup/SPIRV-Tools.git
cd SPIRV-Tools
python3 utils/git-sync-deps
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSPIRV_SKIP_TESTS=ON
cmake --build build --target spirv-dis spirv-val -j 4
export PATH="$PWD/build/tools:$PATH"
```

You can instead pass tool paths explicitly:
`--spirv-dis /path/to/spirv-dis --spirv-val /path/to/spirv-val
--llvm-bcanalyzer /path/to/llvm-bcanalyzer`.

## 1. Dump shaders from your current setup

Close the game. Create a **fresh, empty** dump directory:

```sh
mkdir -p "$HOME/ac8-cloud-fix/dump"
```

In Steam → ACE COMBAT 8 → Properties → Launch Options, temporarily use the
following, replacing `YOURUSER` with your Linux username. Use absolute paths:

```text
VKD3D_SHADER_DUMP_PATH="/home/YOURUSER/ac8-cloud-fix/dump" VKD3D_SHADER_CACHE_PATH=0 %command%
```

Disable any existing shader override first, including overrides in Proton's
`user_settings.py`. Otherwise you may dump an incomplete set or use an older
replacement instead of compiling the original shader. Preserve your old launch
options so you can restore them later.

Launch the game, let its shader warmup finish, and load the affected mission.
Then exit the game. The directory should contain matching `<hash>.spv` and
`<hash>.dxil` files. A full dump can occupy several GB; the tested dump was 6.7 GB.
Remove the temporary dump options after collecting it.

`VKD3D_SHADER_CACHE_PATH=0` temporarily disables vkd3d's internal cache so cached
translations do not hide shaders from the dump. Application-managed pipeline
caches may still affect collection. If the dump is empty or incomplete, inspect
the log/settings rather than assuming the shader has been fixed. Do not mix
dumps from different Proton versions or GPU configurations in one directory.

## 2. Inspect without writing patches

From the directory containing `patch_clouds.py`:

```sh
python3 patch_clouds.py "$HOME/ac8-cloud-fix/dump"
```

For each candidate the script:

1. Checks the DXIL bytes against the filename's vkd3d FNV-1 hash.
2. Confirms `SkyTraceCS` and the expected sentinel/packing source in the embedded
   HLSL debug metadata.
3. Finds the float32 constant `-1 / 131072` and follows its specific conversion,
   vector construction, and `GLSL.std.450 PackHalf2x16` use chain.
4. Rejects ambiguous conversions or unexpected additional uses.
5. Validates both original and proposed replacement with `spirv-val`.

It does not search for a hardcoded byte offset or change every half conversion.
Stripped DXIL metadata and different instruction patterns are unsupported; the
script will refuse them rather than guess. The source checks are deliberately
conservative, so an equivalent but differently written shader may be refused.

## 3. Generate replacement files

```sh
python3 patch_clouds.py "$HOME/ac8-cloud-fix/dump" \
  --apply --output "$HOME/ac8-cloud-fix/patched"
```

The output directory **must not exist**. On success it contains `<hash>.spv`
replacements and `manifest.json`, recording input/output SHA-256 hashes, offsets,
IDs, tool versions, and validation results. Unrelated shaders are not copied.
If any identified candidate is refused, no patch directory is written.

Status meanings:

- `patchable`: recognized and validated; written only with `--apply`.
- `already-patched-pattern`: the recognized chain already uses `OpCopyObject`;
  left untouched and not copied. This is not proof of correct rendering.
- `refused`: a potential match failed checks; inspect the reason.
- `skipped-invalid-header`: the file is not a complete little-endian SPIR-V
  module. It is reported and left untouched; it has not been repaired or verified.

Exit codes: `0` means supported candidates were handled; `1` means no supported
pattern was found; `2` means an error or refused candidate. Always check the
summary's `wrote` count. Exit 0 with already-patched inputs can write zero files.
To save a dry-run report, add `--report /path/to/new-report.json`.

## 4. Load and verify

Use these Steam launch options for a verification run, again replacing `YOURUSER`:

```text
VKD3D_SHADER_OVERRIDE="/home/YOURUSER/ac8-cloud-fix/patched" VKD3D_SHADER_DEBUG=info VKD3D_LOG_FILE="Z:/home/YOURUSER/ac8-cloud-fix/verify.log" %command%
```

Start a fresh game process and load the affected mission. In `verify.log`, look
for `Overriding shader hash ... with alternative SPIR-V module from ...`:

```sh
grep 'Overriding shader hash' "$HOME/ac8-cloud-fix/verify.log"
```

This confirms a replacement loaded; separately check that clouds render
correctly against terrain. A shader-loading message does not establish correct
rendering or coverage of all missions.

If there are no replacement messages, first check paths, Steam launch options,
and any `user_settings.py` overrides. For one diagnostic launch, temporarily add
`VKD3D_SHADER_CACHE_PATH=0` to bypass vkd3d's internal cache. Do not delete all
Steam/NVIDIA caches. Remove that diagnostic setting afterward.

After verification, the normal launch option can be just:

```text
VKD3D_SHADER_OVERRIDE="/home/YOURUSER/ac8-cloud-fix/patched" %command%
```

No ray-tracing or graphics-quality changes are required by this workaround.

## Undo or update

Remove `VKD3D_SHADER_OVERRIDE` from the launch configuration and restart the game.
If you configured it in `user_settings.py`, remove it there too. The script never
edits those settings itself. Keep the original dump for comparison.

After changing Proton, GPU, relevant rendering configuration, or game build,
generate a fresh dump and a new output directory. Even if the original shader
hash stays the same, the translated SPIR-V and resource bindings can differ.
Do not rename somebody else's replacement to force a hash match.

## Tests

The package contains no game shaders. With your own original dump and `spirv-as`
also on `PATH`, run:

```sh
python3 test_patch_clouds.py "$HOME/ac8-cloud-fix/dump"
```

See `TESTING.md` for the actual validation performed for this release. The tests
include dry-run preservation, byte-exact changes, output collision refusal,
already-patched inputs, damaged dumps, identity mismatch, ambiguous matches,
extra uses, and changed instruction IDs/offsets. A synthetic hash-change test
checks that there is no hash allowlist; it is not a cross-driver rendering test.

## References

- [vkd3d-proton dump/override documentation](https://github.com/HansKristian-Work/vkd3d-proton#advanced-shader-debugging)
- [vkd3d-proton shader-cache documentation](https://github.com/HansKristian-Work/vkd3d-proton#shader-cache)
- [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools)
- [My original comment on the AC8 Proton issue](https://github.com/ValveSoftware/Proton/issues/10198#issuecomment-5924274607)

## AI use disclosure

I used GPT-6-Astra in Codex to debug this with Nvidia Nsight frame captures. Getting Nvidia Nsight to capture a frame without incompatibility issues and then have it actually replayable on the GPU required a lot of iterating. I also do computer things for a living :3
