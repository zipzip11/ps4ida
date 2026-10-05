# ps4ida — IDA Pro loader for PlayStation 4 modules

IDA Pro 9.4+ loader for PlayStation 4 modules, written against the modern
`ida_*` IDAPython API (no `idc`, no `ctypes`).

## Credits

Based on [ps4_module_loader](https://github.com/SocraticBliss/ps4_module_loader)
by SocraticBliss. Thanks to everyone credited in its README.

## Install

Copy into your user directory (`~/.idapro` on Linux/macOS, `%APPDATA%\Hex-Rays\IDA Pro` on Windows):

| file | destination |
|---|---|
| `ps4ida.py` | `loaders/` |
| `ps4ida_avx.py` (optional, AVX lifter) | `plugins/` |
| `aerolib.csv` (NID → name database) | `loaders/` (next to the loader) |
| `ps4_errno_700.til` (optional, SCE error-code enum) | `til/` |

Open `eboot.bin` / `*.prx` / `*.sprx` / `*.elf` and choose
**PlayStation 4 … (ps4ida.py)**. Tick **Manual load** to get the options
form (image base, shader handling, optional passes); otherwise defaults are used.

## What it does

* **Input**: decrypted ELF, or *fake-signed* SELF with plaintext segments
  (encrypted/compressed SELFs are rejected with a clear message). Only modules
  with SCE e_types or SCE dynamic data are claimed, so FreeBSD/Linux ELFs and
  the PS4 kernel are left to other loaders.
* **Segments** from program headers, then split using unwind/PLT data the way a
  section-based ELF load would look: `.text`, `.plt`, `.rodata`, `.eh_frame`,
  `.eh_frame_hdr`, `.sce_process_param`, `.data`, `.got.plt`, `.bss`,
  `.data.rel.ro`. Read-only data in the R-X mapping gets an R-only segment so IDA
  never makes code out of it. Non-loaded metadata (PT_DYNAMIC, PT_INTERP, …) is
  no longer mapped at fake addresses.
* **Base address**: position-independent modules (ET_SCE_DYNEXEC, PRX) are
  loaded at `0x400000` by default; fixed-address executables at their link
  address.
* **Relocations** (`R_X86_64_RELATIVE/64/GLOB_DAT/JUMP_SLOT`, TLS ones annotated)
  are applied **with fixups**, so *Edit → Segments → Rebase program* works.
  Relocated slots outside code become `dq offset …` (vtables, function tables).
* **Imports**: every undefined symbol gets an item in an `extern` segment and is
  registered in the *Imports* view per SCE library. GOT slots point at the
  extern items. PLT stubs are created as thunks and carry the plain name
  (`memcpy`), while the extern has `__imp_memcpy`, so pseudocode reads naturally.
  Known prototypes from `gnulnx_x64.til` are applied, and well-known no-return
  functions (`__stack_chk_fail`, `__cxa_throw`, `abort`, …) are marked.
* **NIDs** are resolved through `aerolib.csv`, and library/module ids are decoded
  correctly (multi-digit base64, 32-bit name offsets). Each import/export has a
  comment with its NID, library and module. Unknown NIDs become `nid_<NID>`.
* **Exports**, `_start`, `_init`, `_fini`, `.init_array`/`.fini_array` targets.
* **Process/module param** is typed (`SceProcParam`/`SceModuleParam`, sized from
  the module's own `size` field) and the SDK version is shown.
* **Functions are seeded from `.eh_frame_hdr`** (exact starts of every
  compiled function), which replaces the old byte-pattern prologue hunting.
  After auto-analysis, any FDE IDA could not turn into a function is created
  with its unwind bounds.
* **GCN shaders**: embedded shader binaries are found from their header
  (`s_mov_b32 vcc_hi, imm` → `OrbShdr` footer) and become named, opaque byte
  arrays (`gcn_shader_<hash>`) with a typed `ShaderBinaryInfo`, so auto-analysis
  never disassembles them as x86. Reflection data (uniform/semantic names) is
  folded into the blob, using the wrapping container's size when one is present
  (as in Bloodborne), otherwise up to the next shader. Because the blobs are
  defined items, their strings stay out of the Strings view (unless its
  "Ignore instructions/data definitions" option is ticked). Untick *Skip
  embedded GCN shaders* to leave them as plain bytes (still named and commented).
* **Syscall wrappers** (`mov rax, N; mov r10, rcx; syscall`) are commented.
* **SCE error codes**: once the first auto-analysis finishes, `0x80xxxxxx`
  immediates matching `PS4_ERROR_CODES` are converted to enum members.
* **SCE dynlib data** (symbol/relocation/string/hash tables, `_DYNAMIC`) is
  mapped into a trailing `.sce_dynlibdata` segment and typed with ELF structures
  for browsing. This can be switched off.
* A summary (module, SDK, fingerprint, needed modules, libraries, linked library
  versions) is written as the database's header comment.

## Differences from ps4_module_loader

| old | new |
|---|---|
| claimed *every* x86-64 ELF (`/bin/true` loaded as a "PS4 module") | only SCE modules |
| symbol lookup by sorted string offsets (O(n²), wrong when names aren't in order) | symbol table indexed properly |
| `R_X86_64_64` added the addend to the **symbol index** | S + A |
| module/library name offsets masked to 12 bits, versions swapped | correct `DT_SCE_*` layouts |
| library id `AB` decoded as `A + B` | base64 big-endian (`A*64 + B`) |
| DYNAMIC/DYNLIBDATA mapped at `file offset + 0x1000000` | parsed from the file, optionally mapped after the image |
| ~40 byte-pattern scans for prologues | `.eh_frame_hdr` FDE table |
| `ctypes` into `libida` for imports | `ida_loader.import_module` |
| enum/struct APIs removed in IDA 9 (error-code pass silently did nothing) | `tinfo_t` based |
| error codes via regex text search over the listing before analysis | byte scan at load, applied after analysis |
| blanket "every unknown RELRO qword is data" | relocations define the data; data coagulation (`AF_FINAL`) disabled |
| yes/no/cancel base-address prompt on every load | options form only on *Manual load* |
| syscall table had `no` stripped from names (`mkd`, `nasleep`, `kmq_tify`) | fixed |

## AVX lifter plugin (`ps4ida_avx.py`)

Optional, independent of the loader (works on any x64 database): copy to
`plugins/`. The x64 decompiler leaves VEX-encoded instructions as `__asm`
blocks; the plugin lifts them in a microcode filter:

* 128-bit VEX instructions are rewritten into their SSE twin and handed to the
  decompiler's own SSE lifter (`vaddps d,s1,s2` -> `movaps d,s1; addps d,s2`).
* Lane extracts (`vpshufd x,y,k`, `vmovhlps`, `vinsertps`, ...) become lane
  moves, so code reads `v.m128_f32[3]` instead of shuffle intrinsics.
* `roundss/roundsd` -> `floorf/ceilf/truncf/rintf`; scalar `sqrt/min/max` with
  awkward operand orders -> `sqrtf/fminf/fmaxf`.
* `vpshufd`/`vmovdqa` feeding float code are typed as float (`_mm_shuffle_ps`,
  `_mm_load_ps`) instead of `__m128i` with casts.
* 256-bit `ymm` code and remaining forms become `_mm256_*`/`_mm_*` intrinsic
  calls (`_mm256_add_ps`, `_mm256_set_m128`, `_mm256_extractf128_ps`, ...).
  The decompiler keeps `xmmN`/`ymmN` apart; a per-function reaching-definitions
  pass decides where the halves must be re-joined, including VEX upper-lane
  zeroing (`_mm256_zextps128_ps256`).

On Bloodborne, all 26,479 functions containing AVX decompile with no `__asm`
left and no decompilation failures; decompilation is ~25% slower. Toggle per database with
*Edit -> Plugins -> ps4ida AVX lifter*; re-decompile (F5) to refresh cached
pseudocode.

## Limitations / not handled

* Encrypted SELF/SPRX: decrypt first.
* PS5 (prospero) modules use different dynamic tags and are not claimed.
* No SCE SDK prototypes are bundled; only libc/libstdc++ prototypes from IDA's
  own type libraries are applied.
