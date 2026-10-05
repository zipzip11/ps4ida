"""
ps4ida.py -- IDA Pro 9.4+ loader for PlayStation 4 user-mode modules.

Loads eboot.bin / *.prx / *.sprx / *.elf (decrypted ELF, or fake-signed SELF
with plaintext segments) for x86-64 analysis:

  * segments from program headers, split into .text/.plt/.rodata/.eh_frame/
    .eh_frame_hdr/.data/.got.plt/.bss using .eh_frame and PLT information
  * SCE dynamic info (DT_SCE_*): modules, libraries, symbols, NIDs
  * relocations applied with fixups recorded (Edit > Segments > Rebase works)
  * imports in an `extern` segment + Imports view, PLT stubs named
  * exports, entry point, init/fini, process/module param
  * function starts seeded from .eh_frame FDEs
  * embedded GCN shader binaries detected and kept out of x86 analysis
  * optional: syscall wrapper comments, SCE error-code enum after analysis

Install: copy this file and aerolib.csv to <IDAUSR>/loaders, and optionally
ps4_errno_700.til to <IDAUSR>/til.  Tick "Manual load" in the load dialog to
get the options form (image base, shader handling, ...).
"""

from __future__ import annotations

import bisect
import os
import re
import struct
import time
from dataclasses import dataclass

import ida_auto
import ida_bytes
import ida_diskio
import ida_entry
import ida_fixup
import ida_funcs
import ida_ida
import ida_idaapi
import ida_idp
import ida_kernwin
import ida_lines
import ida_loader
import ida_name
import ida_nalt
import ida_netnode
import ida_offset
import ida_segment
import ida_typeinf
import ida_ua

BADADDR = ida_idaapi.BADADDR
MASK64 = (1 << 64) - 1

# ---------------------------------------------------------------------------
# Format constants
# ---------------------------------------------------------------------------

ELF_MAGIC = b"\x7fELF"
SELF_MAGIC = b"\x4f\x15\x3d\x1d"
EM_X86_64 = 62

ET_EXEC = 2
ET_DYN = 3
ET_SCE_EXEC = 0xFE00
ET_SCE_REPLAY_EXEC = 0xFE01
ET_SCE_RELEXEC = 0xFE04
ET_SCE_STUBLIB = 0xFE0C
ET_SCE_DYNEXEC = 0xFE10
ET_SCE_DYNAMIC = 0xFE18

ET_NAMES = {
    ET_EXEC: "executable",
    ET_DYN: "shared object",
    ET_SCE_EXEC: "executable",
    ET_SCE_REPLAY_EXEC: "replay executable",
    ET_SCE_RELEXEC: "relocatable executable",
    ET_SCE_STUBLIB: "stub library",
    ET_SCE_DYNEXEC: "executable (ASLR)",
    ET_SCE_DYNAMIC: "PRX",
}

PT_LOAD = 1
PT_DYNAMIC = 2
PT_INTERP = 3
PT_TLS = 7
PT_SCE_DYNLIBDATA = 0x61000000
PT_SCE_PROCPARAM = 0x61000001
PT_SCE_MODULE_PARAM = 0x61000002
PT_SCE_RELRO = 0x61000010
PT_GNU_EH_FRAME = 0x6474E550
PT_SCE_COMMENT = 0x6FFFFF00
PT_SCE_LIBVERSION = 0x6FFFFF01

PF_X = 1
PF_W = 2
PF_R = 4

STB_LOCAL = 0
STB_GLOBAL = 1
STB_WEAK = 2
STT_NOTYPE = 0
STT_OBJECT = 1
STT_FUNC = 2
STT_TLS = 6

DT_NULL = 0x00
DT_NEEDED = 0x01
DT_INIT = 0x0C
DT_FINI = 0x0D
DT_SONAME = 0x0E
DT_DEBUG = 0x15
DT_TEXTREL = 0x16
DT_INIT_ARRAY = 0x19
DT_FINI_ARRAY = 0x1A
DT_INIT_ARRAYSZ = 0x1B
DT_FINI_ARRAYSZ = 0x1C
DT_FLAGS = 0x1E
DT_PREINIT_ARRAY = 0x20
DT_PREINIT_ARRAYSZ = 0x21
DT_SCE_IDTABENTSZ = 0x61000005
DT_SCE_FINGERPRINT = 0x61000007
DT_SCE_ORIGINAL_FILENAME = 0x61000009
DT_SCE_MODULE_INFO = 0x6100000D
DT_SCE_NEEDED_MODULE = 0x6100000F
DT_SCE_MODULE_ATTR = 0x61000011
DT_SCE_EXPORT_LIB = 0x61000013
DT_SCE_IMPORT_LIB = 0x61000015
DT_SCE_EXPORT_LIB_ATTR = 0x61000017
DT_SCE_IMPORT_LIB_ATTR = 0x61000019
DT_SCE_STUB_MODULE_NAME = 0x6100001D
DT_SCE_STUB_MODULE_VERSION = 0x6100001F
DT_SCE_STUB_LIBRARY_NAME = 0x61000021
DT_SCE_STUB_LIBRARY_VERSION = 0x61000023
DT_SCE_HASH = 0x61000025
DT_SCE_PLTGOT = 0x61000027
DT_SCE_JMPREL = 0x61000029
DT_SCE_PLTREL = 0x6100002B
DT_SCE_PLTRELSZ = 0x6100002D
DT_SCE_RELA = 0x6100002F
DT_SCE_RELASZ = 0x61000031
DT_SCE_RELAENT = 0x61000033
DT_SCE_STRTAB = 0x61000035
DT_SCE_STRSZ = 0x61000037
DT_SCE_SYMTAB = 0x61000039
DT_SCE_SYMENT = 0x6100003B
DT_SCE_HASHSZ = 0x6100003D
DT_SCE_SYMTABSZ = 0x6100003F

DT_NAMES = {v: k for k, v in globals().items() if k.startswith("DT_") and isinstance(v, int)}

R_X86_64_NONE = 0
R_X86_64_64 = 1
R_X86_64_GLOB_DAT = 6
R_X86_64_JUMP_SLOT = 7
R_X86_64_RELATIVE = 8
R_X86_64_DTPMOD64 = 16
R_X86_64_DTPOFF64 = 17
R_X86_64_TPOFF64 = 18

MODULE_ATTRS = {0x1: "CANT_STOP", 0x2: "EXCLUSIVE_LOAD", 0x4: "EXCLUSIVE_START",
                0x8: "CAN_RESTART", 0x10: "CAN_RELOCATE", 0x20: "CANT_SHARE"}
LIBRARY_ATTRS = {0x1: "AUTO_EXPORT", 0x2: "WEAK_EXPORT", 0x8: "LOOSE_IMPORT"}

PROC_PARAM_MAGIC = 0x4942524F    # 'ORBI'
MODULE_PARAM_MAGIC = 0x3C13F4BF

# Base64 variant used by SCE to encode NIDs and library/module ids.
NID_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-"
_NID_INDEX = {c: i for i, c in enumerate(NID_ALPHABET)}

# GCN shader binaries: code starts with `s_mov_b32 vcc_hi, <literal>` where the
# literal locates the ShaderBinaryInfo footer ("OrbShdr") right after the code.
GCN_SHADER_MARKER = b"\xff\x03\xeb\xbe"
GCN_INFO_MAGIC = b"OrbShdr"
GCN_INFO_SIZE = 28
GCN_FILE_MAGIC = b"Shdr"            # Gnmx ShaderFileHeader preceding the code
GCN_MAX_HEADER_DISTANCE = 0x400
GCN_MAX_GAP = 0x10000
# Container seen wrapping the .sb files in Bloodborne: [magic][size]... 'Shdr'.
# Its size also covers the shader's reflection data (uniform/semantic names).
GCN_CONTAINER_MAGICS = (4444444, 9999999)
GCN_CONTAINER_HEADER = 0x2C

DEFAULT_PIC_BASE = 0x400000

NORETURN_NAMES = frozenset((
    "__stack_chk_fail", "__cxa_throw", "__cxa_rethrow", "__cxa_bad_cast",
    "__cxa_bad_typeid", "__cxa_pure_virtual", "__cxa_call_unexpected",
    "_Unwind_Resume", "_ZSt9terminatev", "_ZSt10unexpectedv", "abort", "exit",
    "_exit", "_Exit", "quick_exit", "longjmp", "_longjmp", "siglongjmp",
    "pthread_exit", "scePthreadExit", "_ZSt14_Xlength_errorPKc",
    "_ZSt14_Xout_of_rangePKc", "_ZSt11_Xbad_allocv",
    "_ZSt18_Xinvalid_argumentPKc", "_ZSt10_Rng_abortPKc",
))

# FreeBSD 9 syscalls + Sony extensions; "-" marks unused numbers.
SYSCALL_NAMES = """
nosys exit fork read write open close wait4 creat link unlink execv chdir fchdir mknod chmod
chown obreak getfsstat lseek getpid mount unmount setuid getuid geteuid ptrace recvmsg sendmsg
recvfrom accept getpeername getsockname access chflags fchflags sync kill stat getppid lstat
dup pipe getegid profil ktrace sigaction getgid sigprocmask getlogin setlogin acct sigpending
sigaltstack ioctl reboot revoke symlink readlink execve umask chroot fstat getkerninfo
getpagesize msync vfork vread vwrite sbrk sstk mmap ovadvise munmap mprotect madvise vhangup
vlimit mincore getgroups setgroups getpgrp setpgid setitimer wait swapon getitimer gethostname
sethostname getdtablesize dup2 getdopt fcntl select setdopt fsync setpriority socket connect
accept getpriority send recv sigreturn bind setsockopt listen vtimes sigvec sigblock sigsetmask
sigsuspend sigstack recvmsg sendmsg vtrace gettimeofday getrusage getsockopt resuba readv
writev settimeofday fchown fchmod recvfrom setreuid setregid rename truncate ftruncate flock
mkfifo sendto shutdown socketpair mkdir rmdir utimes sigreturn adjtime getpeername gethostid
sethostid getrlimit setrlimit killpg setsid quotactl quota getsockname sem_lock sem_wakeup
asyncdaemon nlm_syscall nfssvc getdirentries statfs fstatfs - lgetfh getfh getdomainname
setdomainname uname sysarch rtprio - - semsys msgsys shmsys - pread pwrite setfib ntp_adjtime
sfork getdescriptor setdescriptor - setgid setegid seteuid lfs_bmapv lfs_markv lfs_segclean
lfs_segwait stat fstat lstat pathconf fpathconf - getrlimit setrlimit getdirentries mmap nosys
lseek truncate ftruncate sysctl mlock munlock undelete futimes getpgid newreboot poll - - - - -
- - - - - semctl semget semop semconfig msgctl msgget msgsnd msgrcv shmat shmctl shmdt shmget
clock_gettime clock_settime clock_getres ktimer_create ktimer_delete ktimer_settime
ktimer_gettime ktimer_getoverrun nanosleep ffclock_getcounter ffclock_setestimate
ffclock_getestimate - - - clock_getcpuclockid2 ntp_gettime - minherit rfork openbsd_poll
issetugid lchown aio_read aio_write lio_listio - - - - - - - - - - - - - - getdents - lchmod
lchown lutimes msync nstat nfstat nlstat - - - - - - - - preadv pwritev - - - - - - fhstatfs
fhopen fhstat modnext modstat modfnext modfind kldload kldunload kldfind kldnext kldstat
kldfirstmod getsid setresuid setresgid signanosleep aio_return aio_suspend aio_cancel aio_error
aio_read aio_write lio_listio yield thr_sleep thr_wakeup mlockall munlockall getcwd
sched_setparam sched_getparam sched_setscheduler sched_getscheduler sched_yield
sched_get_priority_max sched_get_priority_min sched_rr_get_interval utrace sendfile kldsym jail
nnpfs_syscall sigprocmask sigsuspend sigaction sigpending sigreturn sigtimedwait sigwaitinfo
acl_get_file acl_set_file acl_get_fd acl_set_fd acl_delete_file acl_delete_fd acl_aclcheck_file
acl_aclcheck_fd extattrctl extattr_set_file extattr_get_file extattr_delete_file
aio_waitcomplete getresuid getresgid kqueue kevent cap_get_proc cap_set_proc cap_get_fd
cap_get_file cap_set_fd cap_set_file - extattr_set_fd extattr_get_fd extattr_delete_fd setugid
nfsclnt eaccess afs3_syscall nmount kse_exit kse_wakeup kse_create kse_thr_interrupt
kse_release mac_get_proc mac_set_proc mac_get_fd mac_get_file mac_set_fd mac_set_file kenv
lchflags uuidgen sendfile mac_syscall getfsstat statfs fstatfs fhstatfs - ksem_close ksem_post
ksem_wait ksem_trywait ksem_init ksem_open ksem_unlink ksem_getvalue ksem_destroy mac_get_pid
mac_get_link mac_set_link extattr_set_link extattr_get_link extattr_delete_link mac_execve
sigaction sigreturn xstat xfstat xlstat getcontext setcontext swapcontext swapoff acl_get_link
acl_set_link acl_delete_link acl_aclcheck_link sigwait thr_create thr_exit thr_self thr_kill -
- jail_attach extattr_list_fd extattr_list_file extattr_list_link kse_switchin ksem_timedwait
thr_suspend thr_wake kldunloadf audit auditon getauid setauid getaudit setaudit getaudit_addr
setaudit_addr auditctl umtx_op thr_new sigqueue kmq_open kmq_setattr kmq_timedreceive
kmq_timedsend kmq_notify kmq_unlink abort2 thr_set_name aio_fsync rtprio_thread - -
getpath_fromfd getpath_fromaddr sctp_peeloff sctp_generic_sendmsg sctp_generic_sendmsg_iov
sctp_generic_recvmsg pread pwrite mmap lseek truncate ftruncate thr_kill2 shm_open shm_unlink
cpuset cpuset_setid cpuset_getid cpuset_getaffinity cpuset_setaffinity faccessat fchmodat
fchownat fexecve fstatat futimesat linkat mkdirat mkfifoat mknodat openat readlinkat renameat
symlinkat unlinkat posix_openpt gssd_syscall jail_get jail_set jail_remove closefrom semctl
msgctl shmctl lpathconf cap_new cap_rights_get cap_enter cap_getmode pdfork pdkill pdgetpid
pdwait4 pselect getloginclass setloginclass rctl_get_racct rctl_get_rules rctl_get_limits
rctl_add_rule rctl_remove_rule posix_fallocate posix_fadvise regmgr_call jitshm_create
jitshm_alias dl_get_list dl_get_info dl_notify_event evf_create evf_delete evf_open evf_close
evf_wait evf_trywait evf_set evf_clear evf_cancel query_memory_protection batch_map osem_create
osem_delete osem_open osem_close osem_wait osem_trywait osem_post osem_cancel namedobj_create
namedobj_delete set_vm_container debug_init suspend_process resume_process opmc_enable
opmc_disable opmc_set_ctl opmc_set_ctr opmc_get_ctr budget_create budget_delete budget_get
budget_set virtual_query mdbg_call sblock_create sblock_delete sblock_enter sblock_exit
sblock_xenter sblock_xexit eport_create eport_delete eport_trigger eport_open eport_close
is_in_sandbox dmem_container get_authinfo mname dynlib_dlopen dynlib_dlclose dynlib_dlsym
dynlib_get_list dynlib_get_info dynlib_load_prx dynlib_unload_prx dynlib_do_copy_relocations
dynlib_prepare_dlclose dynlib_get_proc_param dynlib_process_needed_and_relocate sandbox_path
mdbg_service randomized_path rdup dl_get_metadata workaround8849 is_development_mode
get_self_auth_info dynlib_get_info_ex budget_getid budget_get_ptype
get_paging_stats_of_all_threads get_proc_type_info get_resident_count
prepare_to_suspend_process get_resident_fmem_count thr_get_name set_gpo
get_paging_stats_of_all_objects test_debug_rwmem free_stack suspend_system ipmimgr_call get_gpo
get_vm_map_timestamp opmc_set_hw opmc_get_hw get_cpu_usage_all mmap_dmem physhm_open
physhm_unlink resume_internal_hdd thr_suspend_ucontext thr_resume_ucontext thr_get_ucontext
thr_set_ucontext set_timezone_info set_phys_fmem_limit utc_to_localtime localtime_to_utc
set_uevt get_cpu_usage_proc get_map_statistics set_chicken_switches - -
get_kernel_mem_statistics get_sdk_compiled_version app_state_change dynlib_get_obj_member
budget_get_ptype_of_budget prepare_to_resume_process process_terminate blockpool_open
blockpool_map blockpool_unmap dynlib_get_info_for_libdbg blockpool_batch fdatasync
dynlib_get_list2 dynlib_get_info2 aio_submit aio_multi_delete aio_multi_wait aio_multi_poll
aio_get_data aio_multi_cancel get_bio_usage_all aio_create aio_submit_cmd aio_init
get_page_table_stats dynlib_get_list_for_libdbg blockpool_move virtual_query_all
reserve_2mb_page cpumode_yield get_phys_page_size
""".split()

# mov rax, imm32 / mov eax, imm32 ; mov r10, rcx ; syscall
SYSCALL_RE = re.compile(rb"(?:\x48\xc7\xc0|\xb8)(.{4})\x49\x89\xca\x0f\x05", re.S)

ERRNO_TIL = "ps4_errno_700"
ERRNO_ENUM = "PS4_ERROR_CODES"

LOCAL_TYPES = """
typedef struct Elf64_Sym {
  uint32_t st_name; uint8_t st_info; uint8_t st_other; uint16_t st_shndx;
  uint64_t st_value; uint64_t st_size;
} Elf64_Sym;
typedef struct Elf64_Rela { uint64_t r_offset; uint64_t r_info; int64_t r_addend; } Elf64_Rela;
typedef struct Elf64_Dyn { int64_t d_tag; uint64_t d_val; } Elf64_Dyn;
struct ShaderBinaryInfo {
  char signature[7];
  uint8_t version;
  uint32_t pssl_or_cg : 1;
  uint32_t cached : 1;
  uint32_t type : 4;
  uint32_t source_type : 2;
  uint32_t length : 24;
  uint8_t chunk_usage_base_offset_in_dw;
  uint8_t num_input_usage_slots;
  uint8_t flags;
  uint8_t reserved;
  uint32_t shader_hash_lo;
  uint32_t shader_hash_hi;
  uint32_t crc32;
};
"""

# (offset, C declaration) -- the struct is declared up to the size the module states.
PROC_PARAM_FIELDS = (
    (0x00, "uint64_t size"),
    (0x08, "uint32_t magic"),
    (0x0C, "uint32_t entry_count"),
    (0x10, "uint64_t sdk_version"),
    (0x18, "const char *process_name"),
    (0x20, "const char *user_main_thread_name"),
    (0x28, "const uint32_t *user_main_thread_priority"),
    (0x30, "const uint32_t *user_main_thread_stack_size"),
    (0x38, "void *libc_param"),
    (0x40, "void *kernel_mem_param"),
    (0x48, "void *kernel_fs_param"),
    (0x50, "const uint32_t *process_preload_enabled"),
)
MODULE_PARAM_FIELDS = (
    (0x00, "uint64_t size"),
    (0x08, "uint32_t magic"),
    (0x0C, "uint32_t entry_count"),
    (0x10, "uint64_t sdk_version"),
)


class FormatError(Exception):
    pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def decode_sce_id(text: str) -> int:
    value = 0
    for ch in text:
        value = (value << 6) | _NID_INDEX[ch]
    return value


def sdk_version_str(version: int) -> str:
    return "%x.%03x.%03x" % (version >> 24, (version >> 12) & 0xFFF, version & 0xFFF)


def flags_str(value: int, names: dict) -> str:
    parts = [name for bit, name in names.items() if value & bit]
    rest = value & ~sum(names)
    if rest:
        parts.append("%#x" % rest)
    return "|".join(parts) or "none"


def read_uleb(buf, pos):
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return result, pos


def read_sleb(buf, pos):
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            if byte & 0x40:
                result -= 1 << shift
            return result, pos


_EH_FIXED = {0x00: "<Q", 0x02: "<H", 0x03: "<I", 0x04: "<Q", 0x0A: "<h", 0x0B: "<i", 0x0C: "<q"}


def read_eh_pointer(buf, pos, enc, base_va, data_base=0):
    """Decode a DW_EH_PE_* encoded pointer at buf[pos]; base_va is the VA of buf[0]."""
    fmt = enc & 0x0F
    if fmt == 0x01:
        value, end = read_uleb(buf, pos)
    elif fmt == 0x09:
        value, end = read_sleb(buf, pos)
    elif fmt in _EH_FIXED:
        f = _EH_FIXED[fmt]
        value = struct.unpack_from(f, buf, pos)[0]
        end = pos + struct.calcsize(f)
    else:
        raise ValueError("unsupported pointer encoding %#x" % enc)
    app = enc & 0x70
    if app == 0x10:
        value += base_va + pos
    elif app == 0x30:
        value += data_base
    elif app != 0:
        raise ValueError("unsupported pointer application %#x" % enc)
    return value & MASK64, end


# ---------------------------------------------------------------------------
# Binary parsing (no IDA dependencies)
# ---------------------------------------------------------------------------

@dataclass
class Phdr:
    index: int
    type: int
    flags: int
    offset: int
    vaddr: int
    filesz: int
    memsz: int
    align: int

    @property
    def end(self) -> int:
        return self.vaddr + self.memsz

    def contains(self, va: int, size: int = 1) -> bool:
        return self.vaddr <= va and va + size <= self.vaddr + self.memsz


@dataclass
class SceModule:
    id: int
    name: str
    major: int
    minor: int
    attr: int = 0


@dataclass
class SceLibrary:
    id: int
    name: str
    version: int
    exported: bool
    attr: int = 0


@dataclass
class Symbol:
    index: int
    name: str
    info: int
    shndx: int
    value: int
    size: int
    nid: str | None = None
    lib_id: int | None = None
    mod_id: int | None = None

    @property
    def bind(self) -> int:
        return self.info >> 4

    @property
    def type(self) -> int:
        return self.info & 0xF

    @property
    def defined(self) -> bool:
        return self.shndx != 0


@dataclass
class GcnShader:
    start: int          # link-time VA where the blob starts (file header, else code)
    code: int           # first GCN instruction
    info: int           # ShaderBinaryInfo footer
    end: int            # end of the blob (footer end, or extended over trailing metadata)
    type: int
    length: int
    hash: int
    crc: int
    container_end: int = 0  # end of the wrapping container, when one was recognised


class Container:
    """Maps ELF file offsets to input file offsets (plain ELF or fake SELF)."""

    def __init__(self, data: bytes):
        self.data = data
        self.is_self = data[:4] == SELF_MAGIC
        self.elf_base = 0
        self.entries: list[tuple[int, int, int, int]] = []   # props, offset, filesz, memsz
        self.segment_map: list[tuple[int, int, int, bool]] = []  # elf_off, size, file_off, plain
        if self.is_self:
            if len(data) < 0x20:
                raise FormatError("truncated SELF header")
            count = struct.unpack_from("<H", data, 0x18)[0]
            self.elf_base = 0x20 + count * 0x20
            if self.elf_base + 0x40 > len(data):
                raise FormatError("truncated SELF entry table")
            self.entries = [struct.unpack_from("<QQQQ", data, 0x20 + i * 0x20) for i in range(count)]

    def header(self, offset: int, size: int) -> bytes:
        start = self.elf_base + offset
        if start + size > len(self.data):
            raise FormatError("ELF headers extend past end of file")
        return self.data[start:start + size]

    def bind_segments(self, phdrs: list[Phdr]):
        for props, offset, filesz, _ in self.entries:
            if not props & 0x800:              # entry does not carry segment blocks
                continue
            index = (props >> 20) & 0xFFF
            if index >= len(phdrs):
                continue
            plain = not props & 0xA            # neither encrypted (2) nor compressed (8)
            self.segment_map.append((phdrs[index].offset, filesz, offset, plain))

    def file_offset(self, elf_off: int, size: int) -> int:
        if not self.is_self:
            if elf_off + size > len(self.data):
                raise FormatError("segment data at %#x extends past end of file" % elf_off)
            return elf_off
        for start, length, file_off, plain in self.segment_map:
            if start <= elf_off and elf_off + size <= start + length:
                if not plain:
                    raise FormatError("SELF segment data is encrypted or compressed; "
                                      "decrypt the module first")
                return file_off + (elf_off - start)
        raise FormatError("ELF range %#x+%#x is not present in the SELF" % (elf_off, size))

    def read(self, elf_off: int, size: int) -> bytes:
        start = self.file_offset(elf_off, size)
        return self.data[start:start + size]


class PS4Module:
    def __init__(self, data: bytes):
        self.container = c = Container(data)
        ehdr = c.header(0, 0x40)
        if ehdr[:4] != ELF_MAGIC:
            raise FormatError("no ELF header")
        if ehdr[4] != 2 or ehdr[5] != 1:
            raise FormatError("not a little-endian ELF64 file")
        (self.e_type, machine, _, self.entry, phoff, _, _, _, phentsize, phnum,
         _, _, _) = struct.unpack_from("<HHIQQQIHHHHHH", ehdr, 16)
        if machine != EM_X86_64:
            raise FormatError("not an x86-64 module")
        if phentsize != 56 or not phnum:
            raise FormatError("bad program header table")
        table = c.header(phoff, phnum * 56)
        self.phdrs = [Phdr(i, *struct.unpack_from("<IIQQ8xQQQ", table, i * 56)) for i in range(phnum)]
        c.bind_segments(self.phdrs)

        self.loads = sorted((p for p in self.phdrs if p.type in (PT_LOAD, PT_SCE_RELRO) and p.memsz),
                            key=lambda p: p.vaddr)
        if not self.loads:
            raise FormatError("no loadable segments")
        self.link_base = self.loads[0].vaddr
        self.link_end = max(p.end for p in self.loads)
        self.is_pic = self.link_base == 0

        self.dynlib = b""
        self.dynlib_phdr = self.first(PT_SCE_DYNLIBDATA)
        if self.dynlib_phdr and self.dynlib_phdr.filesz:
            self.dynlib = c.read(self.dynlib_phdr.offset, self.dynlib_phdr.filesz)
        self.dynamic: list[tuple[int, int]] = []
        self.dynamic_offset = None          # offset of the table inside dynlib data
        dyn = self.first(PT_DYNAMIC)
        if dyn and dyn.filesz:
            raw = c.read(dyn.offset, dyn.filesz - dyn.filesz % 16)
            for tag, value in struct.iter_unpack("<QQ", raw):
                self.dynamic.append((tag, value))
                if tag == DT_NULL:
                    break
            if self.dynlib_phdr and self.dynlib_phdr.offset <= dyn.offset < \
                    self.dynlib_phdr.offset + self.dynlib_phdr.filesz:
                self.dynamic_offset = dyn.offset - self.dynlib_phdr.offset
        self._tags: dict[int, int] = {}
        for tag, value in self.dynamic:
            self._tags.setdefault(tag, value)
        self._parse_dynamic()

    # -- generic accessors -------------------------------------------------

    def first(self, ptype: int) -> Phdr | None:
        return next((p for p in self.phdrs if p.type == ptype), None)

    def load_for(self, va: int, size: int = 1) -> Phdr | None:
        for p in self.loads:
            if p.vaddr <= va and va + size <= p.vaddr + p.filesz:
                return p
        return None

    def read_va(self, va: int, size: int) -> bytes | None:
        p = self.load_for(va, size)
        if p is None:
            return None
        return self.container.read(p.offset + (va - p.vaddr), size)

    def segment_bytes(self, p: Phdr) -> bytes:
        return self.container.read(p.offset, p.filesz) if p.filesz else b""

    def tag(self, tag: int, default=None):
        return self._tags.get(tag, default)

    def dynlib_table(self, off_tag: int, size_tag: int) -> bytes:
        off, size = self.tag(off_tag), self.tag(size_tag)
        if off is None or not size or off + size > len(self.dynlib):
            return b""
        return self.dynlib[off:off + size]

    def string(self, offset: int) -> str:
        base = self.tag(DT_SCE_STRTAB)
        size = self.tag(DT_SCE_STRSZ, 0)
        if base is None or offset >= size:
            return ""
        start = base + offset
        end = self.dynlib.find(b"\0", start, base + size)
        if end < 0:
            end = base + size
        return self.dynlib[start:end].decode("utf-8", "replace")

    # -- dynamic section ---------------------------------------------------

    def _parse_dynamic(self):
        # Ids are per kind: imported and exported libraries may reuse the same id.
        self.modules: dict[int, SceModule] = {}
        self.import_libs: dict[int, SceLibrary] = {}
        self.export_libs: dict[int, SceLibrary] = {}
        self.module_info: SceModule | None = None
        self.needed: list[str] = []

        for tag, value in self.dynamic:
            low, ident = value & 0xFFFFFFFF, value >> 48
            if tag in (DT_SCE_NEEDED_MODULE, DT_SCE_MODULE_INFO):
                mod = SceModule(ident, self.string(low), (value >> 40) & 0xFF, (value >> 32) & 0xFF)
                if tag == DT_SCE_MODULE_INFO:
                    self.module_info = mod
                else:
                    self.modules[ident] = mod
            elif tag in (DT_SCE_IMPORT_LIB, DT_SCE_EXPORT_LIB):
                exported = tag == DT_SCE_EXPORT_LIB
                libs = self.export_libs if exported else self.import_libs
                libs[ident] = SceLibrary(ident, self.string(low), (value >> 32) & 0xFFFF, exported)
            elif tag == DT_NEEDED:
                self.needed.append(self.string(value))
        for tag, value in self.dynamic:
            ident = value >> 48
            if tag == DT_SCE_MODULE_ATTR:
                if self.module_info and self.module_info.id == ident:
                    self.module_info.attr = value & 0xFFFF
            elif tag in (DT_SCE_IMPORT_LIB_ATTR, DT_SCE_EXPORT_LIB_ATTR):
                libs = self.export_libs if tag == DT_SCE_EXPORT_LIB_ATTR else self.import_libs
                if ident in libs:
                    libs[ident].attr = value & 0xFFFF

        self.symbols: list[Symbol] = []
        raw = self.dynlib_table(DT_SCE_SYMTAB, DT_SCE_SYMTABSZ)
        for index, (name, info, _, shndx, value, size) in enumerate(
                struct.iter_unpack("<IBBHQQ", raw[:len(raw) - len(raw) % 24])):
            sym = Symbol(index, self.string(name) if name else "", info, shndx, value, size)
            parts = sym.name.split("#")
            if len(parts) == 3 and len(parts[0]) == 11:
                try:
                    sym.lib_id = decode_sce_id(parts[1])
                    sym.mod_id = decode_sce_id(parts[2])
                    sym.nid = parts[0]
                except KeyError:
                    pass
            self.symbols.append(sym)

        def relocs(off_tag, size_tag):
            raw = self.dynlib_table(off_tag, size_tag)
            return list(struct.iter_unpack("<QQq", raw[:len(raw) - len(raw) % 24]))

        self.rela = relocs(DT_SCE_RELA, DT_SCE_RELASZ)
        self.jmprel = relocs(DT_SCE_JMPREL, DT_SCE_PLTRELSZ)

    def describe_dynamic(self, tag: int, value: int) -> str:
        name = DT_NAMES.get(tag, "DT_%#x" % tag)
        low, ident = value & 0xFFFFFFFF, value >> 48
        if tag in (DT_NEEDED, DT_SONAME, DT_SCE_ORIGINAL_FILENAME):
            return "%s: %s" % (name, self.string(value))
        if tag in (DT_SCE_NEEDED_MODULE, DT_SCE_MODULE_INFO):
            return "%s: %s id=%d v%d.%d" % (name, self.string(low), ident,
                                            (value >> 40) & 0xFF, (value >> 32) & 0xFF)
        if tag in (DT_SCE_IMPORT_LIB, DT_SCE_EXPORT_LIB):
            return "%s: %s id=%d v%d" % (name, self.string(low), ident, (value >> 32) & 0xFFFF)
        if tag in (DT_SCE_IMPORT_LIB_ATTR, DT_SCE_EXPORT_LIB_ATTR):
            return "%s: id=%d %s" % (name, ident, flags_str(value & 0xFFFF, LIBRARY_ATTRS))
        if tag == DT_SCE_MODULE_ATTR:
            return "%s: id=%d %s" % (name, ident, flags_str(value & 0xFFFF, MODULE_ATTRS))
        return "%s: %#x" % (name, value)

    # -- other metadata ----------------------------------------------------

    def fingerprint(self) -> bytes | None:
        off = self.tag(DT_SCE_FINGERPRINT)
        if off is None or off + 20 > len(self.dynlib):
            return None
        return self.dynlib[off:off + 20]

    def lib_versions(self) -> list[tuple[str, int]]:
        p = self.first(PT_SCE_LIBVERSION)
        out = []
        if not p or not p.filesz:
            return out
        try:
            raw = self.container.read(p.offset, p.filesz)
        except FormatError:
            return out
        pos = 0
        while pos < len(raw):
            length = raw[pos]
            item = raw[pos + 1:pos + 1 + length]
            pos += 1 + length
            name, sep, version = item.partition(b":")
            if not length or not sep or len(version) != 4:
                break
            out.append((name.decode("ascii", "replace"), int.from_bytes(version, "big")))
        return out

    def sce_comment(self) -> str | None:
        p = self.first(PT_SCE_COMMENT)
        if not p or p.filesz < 12:
            return None
        try:
            raw = self.container.read(p.offset, p.filesz)
        except FormatError:
            return None
        length = struct.unpack_from("<I", raw, 8)[0]
        return raw[12:12 + length].split(b"\0")[0].decode("utf-8", "replace") or None

    # -- .eh_frame ---------------------------------------------------------

    def eh_frame(self):
        """Return (eh_frame_start, eh_frame_end, [(start, size), ...]) as link VAs, or None."""
        hdr = self.first(PT_GNU_EH_FRAME)
        if not hdr or hdr.filesz < 12:
            return None
        seg = self.load_for(hdr.vaddr, hdr.filesz)
        if seg is None:
            return None
        buf = self.segment_bytes(seg)
        base = seg.vaddr
        pos = hdr.vaddr - base
        if buf[pos] != 1:
            return None
        enc_count, enc_table = buf[pos + 2], buf[pos + 3]
        try:
            frame_va, table = read_eh_pointer(buf, pos + 4, buf[pos + 1], base, hdr.vaddr)
            count = 0
            if enc_count != 0xFF and enc_table != 0xFF:
                count, table = read_eh_pointer(buf, table, enc_count, base, hdr.vaddr)
        except (ValueError, struct.error):
            return None
        if not seg.vaddr <= frame_va < seg.vaddr + seg.filesz:
            return None
        start = frame_va - base
        limit = pos if start < pos else len(buf)
        cies: dict[int, int | None] = {}
        fdes, end = self._walk_eh_frame(buf, base, start, limit, cies)
        frame_end = base + end
        # The hdr search table only lists live FDEs; a raw walk also returns the
        # dead ones left behind by LTO/--gc-sections, whose pc-relative start
        # then points into .eh_frame itself.
        if count and table + 8 * count <= len(buf):
            try:
                fdes = self._hdr_fdes(buf, base, table, count, enc_table, hdr.vaddr, cies)
            except (ValueError, IndexError, struct.error):
                pass
        code = [(p.vaddr, p.vaddr + p.filesz) for p in self.loads if p.flags & PF_X]
        fdes = [(s, n) for s, n in fdes
                if not frame_va <= s < frame_end and not hdr.vaddr <= s < hdr.end
                and any(lo <= s < hi for lo, hi in code)]
        return frame_va, frame_end, fdes

    def _hdr_fdes(self, buf, base, pos, count, enc, hdr_va, cies):
        if enc == 0x3B:                                   # datarel | sdata4, the usual case
            pairs = [(hdr_va + a, hdr_va + b) for a, b in
                     struct.iter_unpack("<ii", buf[pos:pos + 8 * count])]
        else:
            pairs = []
            for _ in range(count):
                a, pos = read_eh_pointer(buf, pos, enc, base, hdr_va)
                b, pos = read_eh_pointer(buf, pos, enc, base, hdr_va)
                pairs.append((a, b))
        fdes = []
        for start, fde_va in pairs:
            size = 0
            off = fde_va - base
            if 0 <= off < len(buf) - 8:
                body = off + 4
                cie = body - struct.unpack_from("<I", buf, body)[0]
                if cie not in cies:
                    cies[cie] = self._parse_cie(buf, cie + 8, base) if 0 <= cie < off else None
                if cies[cie] is not None:
                    enc_fde = cies[cie]
                    _, p = read_eh_pointer(buf, body + 4, enc_fde, base)
                    size, _ = read_eh_pointer(buf, p, enc_fde & 0x0F, base)
            fdes.append((start & MASK64, size))
        return fdes

    @staticmethod
    def _parse_cie(buf, pos, base):
        version = buf[pos]
        pos += 1
        end = buf.index(b"\0", pos)
        aug = buf[pos:end].decode("ascii", "replace")
        pos = end + 1
        if "eh" in aug:
            pos += 8
        _, pos = read_uleb(buf, pos)
        _, pos = read_sleb(buf, pos)
        if version == 1:
            pos += 1
        else:
            _, pos = read_uleb(buf, pos)
        enc = 0
        if aug.startswith("z"):
            _, pos = read_uleb(buf, pos)
            for ch in aug[1:]:
                if ch == "R":
                    enc = buf[pos]
                    pos += 1
                elif ch == "L":
                    pos += 1
                elif ch == "P":
                    penc = buf[pos]
                    _, pos = read_eh_pointer(buf, pos + 1, penc & 0x7F, base)
                elif ch not in "SB":
                    break
        return enc

    def _walk_eh_frame(self, buf, base, pos, limit, cies):
        fdes = []
        while pos + 4 <= limit:
            length = struct.unpack_from("<I", buf, pos)[0]
            if length == 0:
                pos += 4
                break
            body = pos + 4
            if length == 0xFFFFFFFF:
                length = struct.unpack_from("<Q", buf, body)[0]
                body += 8
            nxt = body + length
            if nxt > limit:
                break
            cie_ptr = struct.unpack_from("<I", buf, body)[0]
            try:
                if cie_ptr == 0:
                    cies[pos] = self._parse_cie(buf, body + 4, base)
                else:
                    cie = body - cie_ptr
                    if cie not in cies:
                        cies[cie] = self._parse_cie(buf, cie + 8, base) if 0 <= cie < pos else None
                    enc = cies[cie]
                    if enc is not None:
                        start, p = read_eh_pointer(buf, body + 4, enc, base)
                        size, _ = read_eh_pointer(buf, p, enc & 0x0F, base)
                        if start:
                            fdes.append((start, size))
            except (ValueError, IndexError, struct.error):
                cies.setdefault(pos, None)
            pos = nxt
        return fdes, pos

    # -- GCN shaders -------------------------------------------------------

    def scan_shaders(self) -> list[GcnShader]:
        found = []
        for seg in self.loads:
            buf = self.segment_bytes(seg)
            base = seg.vaddr
            prev_end = 0
            pos = buf.find(GCN_SHADER_MARKER)
            while pos >= 0:
                nxt = pos + 4
                if pos + 8 <= len(buf):
                    literal = struct.unpack_from("<I", buf, pos + 4)[0]
                    info = pos + (literal + 1) * 8
                    if info + GCN_INFO_SIZE <= len(buf) and buf[info:info + 7] == GCN_INFO_MAGIC:
                        bits, = struct.unpack_from("<I", buf, info + 8)
                        shash, crc = struct.unpack_from("<QI", buf, info + 16)
                        start = pos
                        hdr = buf.rfind(GCN_FILE_MAGIC, max(prev_end, pos - GCN_MAX_HEADER_DISTANCE), pos)
                        if hdr >= 0 and 0 < struct.unpack_from("<H", buf, hdr + 4)[0] < 0x40:
                            start = hdr
                        end = info + GCN_INFO_SIZE
                        container_end = 0
                        wrap = start - GCN_CONTAINER_HEADER
                        if start == hdr and wrap >= prev_end:
                            magic, size = struct.unpack_from("<II", buf, wrap)
                            if magic in GCN_CONTAINER_MAGICS and end <= wrap + size <= len(buf):
                                start, container_end = wrap, base + wrap + size
                        found.append(GcnShader(base + start, base + pos, base + info, base + end,
                                               (bits >> 2) & 0xF, bits >> 8, shash, crc,
                                               container_end))
                        prev_end = nxt = end
                pos = buf.find(GCN_SHADER_MARKER, nxt)
        return found


# ---------------------------------------------------------------------------
# NID database
# ---------------------------------------------------------------------------

_NID_CACHE: dict[str, dict[str, str]] = {}


def nid_db_paths() -> list[str]:
    dirs = []
    here = globals().get("__file__")
    if here:
        dirs.append(os.path.dirname(os.path.abspath(here)))
    dirs.append(os.path.join(ida_diskio.get_user_idadir(), "loaders"))
    dirs.append(ida_diskio.idadir("loaders"))
    return [os.path.join(d, "aerolib.csv") for d in dirs]


def load_nids() -> dict[str, str]:
    for path in nid_db_paths():
        if not os.path.isfile(path):
            continue
        if path in _NID_CACHE:
            return _NID_CACHE[path]
        nids = {}
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                nid, _, name = line.strip().partition(" ")
                if len(nid) == 11 and name:
                    nids[nid] = name
        _NID_CACHE[path] = nids
        return nids
    return {}


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

@dataclass
class Options:
    base: int
    skip_shaders: bool = True
    eh_frame: bool = True
    dynlib: bool = True
    errno: bool = True
    syscalls: bool = True

    FLAGS = ("skip_shaders", "eh_frame", "dynlib", "errno", "syscalls")

    @classmethod
    def defaults(cls, mod: PS4Module) -> "Options":
        return cls(base=DEFAULT_PIC_BASE if mod.is_pic else mod.link_base)

    def ask(self, mod: PS4Module) -> bool:
        F = ida_kernwin.Form
        base_hint = "position-independent, linked at 0" if mod.is_pic else \
            "linked at %#x" % mod.link_base

        class _Form(F):
            def __init__(self):
                super().__init__(r"""STARTITEM 0
BUTTON YES* OK
BUTTON CANCEL Cancel
ps4ida

<#Address the lowest segment is loaded at#Image base (%s):{base}>

<#Mark GCN shader blobs as opaque data so they are never disassembled as x86#Skip embedded GCN shaders:{skip_shaders}>
<#Queue every .eh_frame FDE start as a function#Seed functions from .eh_frame:{eh_frame}>
<#Map the SCE dynlib data (symbol/relocation/string tables) into an extra segment#Map SCE dynlib tables:{dynlib}>
<#Turn 0x80xxxxxx immediates into PS4_ERROR_CODES members once auto-analysis finishes#Apply SCE error codes:{errno}>
<#Comment syscall wrappers with the syscall name#Comment syscalls:{syscalls}>{flags}>
""" % base_hint, {
                    "base": F.NumericInput(tp=F.FT_ADDR, value=0),
                    "flags": F.ChkGroupControl(Options.FLAGS),
                })

        form = _Form()
        form.Compile()
        form.base.value = self.base
        form.flags.value = sum(1 << i for i, n in enumerate(self.FLAGS) if getattr(self, n))
        ok = form.Execute() == 1
        if ok:
            self.base = form.base.value
            for i, name in enumerate(self.FLAGS):
                setattr(self, name, bool(form.flags.value & (1 << i)))
        form.Free()
        return ok


# ---------------------------------------------------------------------------
# IDA side
# ---------------------------------------------------------------------------

STATE_NODE = "$ ps4ida"


def log(text: str):
    ida_kernwin.msg("[ps4ida] %s\n" % text)


@dataclass
class Piece:
    start: int
    end: int
    name: str
    sclass: str
    perm: int


def carve(lo: int, hi: int, default: tuple[str, str], perm: int, carves) -> list[Piece] | None:
    """Split [lo, hi) into named pieces; returns None if the carves overlap."""
    pieces = []
    cur = lo
    for start, end, name, sclass in sorted(c for c in carves if c[1] > c[0]):
        start, end = max(start, lo), min(end, hi)
        if start >= end:
            continue
        if start < cur:
            return None
        if start > cur:
            pieces.append(Piece(cur, start, default[0], default[1], perm))
        pieces.append(Piece(start, end, name, sclass, perm))
        cur = end
    if cur < hi:
        pieces.append(Piece(cur, hi, default[0], default[1], perm))
    return pieces


class PostAnalysis(ida_idp.IDB_Hooks):
    """Runs once when the initial auto-analysis queue drains."""

    live: set = set()

    def __init__(self, fdes: list[tuple[int, int]]):
        super().__init__()
        self.fdes = fdes                    # (ea, size) of every unwind-described function
        self.enum_tid = BADADDR
        self.candidates: list[tuple[int, int]] = []
        PostAnalysis.live.add(self)

    def closebase(self):
        # Database closed before analysis finished: never run on another one.
        self.unhook()
        PostAnalysis.live.discard(self)

    def auto_empty_finally(self):
        self.unhook()
        PostAnalysis.live.discard(self)
        # never break analysis because of a cleanup pass
        try:
            self.create_missed_functions()
        except Exception as exc:
            log("FDE function pass failed: %s" % exc)
        if self.candidates:
            try:
                self.apply_error_codes()
            except Exception as exc:
                log("error-code pass failed: %s" % exc)

    def create_missed_functions(self):
        """FDE starts IDA could not turn into functions on its own (typically
        code falling into an inline jump table) are created with FDE bounds."""
        created = 0
        for ea, size in self.fdes:
            if size and ida_funcs.get_func(ea) is None and ida_funcs.add_func(ea, ea + size):
                created += 1
        if created:
            log("created %d functions from .eh_frame bounds" % created)

    def apply_error_codes(self):
        t0 = time.perf_counter()
        insn = ida_ua.insn_t()
        done = 0
        seen = set()
        for ea, value in self.candidates:
            head = ida_bytes.get_item_head(ea)
            if head in seen or not ida_bytes.is_code(ida_bytes.get_flags(head)):
                continue
            if not ida_ua.decode_insn(insn, head):
                continue
            for n in range(ida_ida.UA_MAXOP):
                op = insn.ops[n]
                if op.type == ida_ua.o_void:
                    break
                if op.type == ida_ua.o_imm and op.value & 0xFFFFFFFF == value:
                    if ida_bytes.op_enum(head, n, self.enum_tid, 0):
                        done += 1
                        seen.add(head)
                    break
        log("applied %s to %d operands in %.2fs" % (ERRNO_ENUM, done, time.perf_counter() - t0))


class Loader:
    def __init__(self, li, mod: PS4Module, opts: Options):
        self.li = li
        self.m = mod
        self.o = opts
        self.delta = opts.base - mod.link_base
        self.nids = load_nids()
        self.text: list[tuple[int, int]] = []          # final VAs of code segments
        self.fdes: list[tuple[int, int]] = []          # link VAs
        self.eh_range: tuple[int, int] | None = None
        self.plt_stubs: dict[int, int] = {}            # symbol index -> link VA of stub
        self.plt_range: tuple[int, int] | None = None
        self.got_range: tuple[int, int] | None = None
        self.externs: dict[int, int] = {}              # symbol index -> final VA
        self.image_end = mod.link_end + self.delta
        self.reloc_sites: list[int] = []               # link VAs, sorted
        self.param_sizes: dict[int, int] = {}
        self.sdk_version: int | None = None
        self.errno_til = False
        self.stats: dict[str, object] = {}
        self.timings: dict[str, float] = {}

    # -- utilities ---------------------------------------------------------

    def va(self, link_va: int) -> int:
        return (link_va + self.delta) & MASK64

    def in_text(self, ea: int) -> bool:
        i = bisect.bisect_right(self.text, (ea, MASK64)) - 1
        return i >= 0 and self.text[i][0] <= ea < self.text[i][1]

    def sym_name(self, sym: Symbol) -> str:
        if sym.nid:
            name = self.nids.get(sym.nid)
            if name:
                return name
            return "nid_" + sym.nid.replace("+", "_").replace("-", "_")
        return sym.name

    def set_name(self, ea: int, name: str, flags: int = 0):
        ida_name.set_name(ea, name, ida_name.SN_NOCHECK | ida_name.SN_NOWARN | ida_name.SN_FORCE | flags)

    def library(self, sym: Symbol) -> SceLibrary | None:
        return (self.m.export_libs if sym.defined else self.m.import_libs).get(sym.lib_id)

    def library_of(self, sym: Symbol) -> str:
        lib = self.library(sym)
        return lib.name if lib else "unknown"

    def sym_comment(self, sym: Symbol) -> str:
        if not sym.nid:
            return ""
        parts = ["NID %s" % sym.nid]
        lib = self.library(sym)
        if sym.defined:
            mod = self.m.module_info if self.m.module_info and self.m.module_info.id == sym.mod_id else None
        else:
            mod = self.m.modules.get(sym.mod_id)
        if lib:
            parts.append("library %s" % lib.name)
        if mod:
            parts.append("module %s" % mod.name)
        return ", ".join(parts)

    def step(self, label: str, fn):
        ida_kernwin.replace_wait_box("ps4ida: %s" % label)
        t0 = time.perf_counter()
        fn()
        self.timings[label] = time.perf_counter() - t0

    # -- driver ------------------------------------------------------------

    def run(self):
        self.post = PostAnalysis([])
        ida_kernwin.show_wait_box("HIDECANCEL\nps4ida: starting")
        try:
            self.step("processor", self.setup_processor)
            self.step("layout", self.compute_layout)
            self.step("segments", self.create_segments)
            self.step("types", self.declare_types)
            self.step("imports", self.create_imports)
            self.step("relocations", self.apply_relocations)
            self.step("PLT", self.process_plt)
            self.step("exports", self.process_exports)
            self.step("entry points", self.process_entry_points)
            self.step("params", self.process_params)
            self.step("shaders", self.process_shaders)
            self.step("functions", self.seed_functions)
            if self.o.dynlib:
                self.step("dynlib data", self.map_dynlib)
            if self.o.syscalls:
                self.step("syscalls", self.comment_syscalls)
            if self.o.errno:
                self.step("error codes", self.schedule_error_codes)
            self.step("summary", self.write_summary)
            self.save_state()
            if self.post.fdes or self.post.candidates:
                self.post.hook()
            else:
                PostAnalysis.live.discard(self.post)
        finally:
            ida_kernwin.hide_wait_box()
        log("timings: " + ", ".join("%s %.2fs" % kv for kv in self.timings.items()))

    def reload(self):
        """NEF_RELOAD: refresh segment bytes from the file and re-apply relocations."""
        self.delta = ida_nalt.get_imagebase() - self.m.link_base
        for p in self.m.loads:
            if p.filesz:
                off = self.m.container.file_offset(p.offset, p.filesz)
                self.li.file2base(off, self.va(p.vaddr), self.va(p.vaddr + p.filesz),
                                  ida_loader.FILEREG_PATCHABLE)
        self.compute_layout()
        for i in range(ida_segment.get_segm_qty()):
            seg = ida_segment.getnseg(i)
            if seg.type == ida_segment.SEG_CODE:
                self.text.append((seg.start_ea, seg.end_ea))
        self.text.sort()
        ext = ida_segment.get_segm_by_name("extern")
        if ext:
            imports = [s for s in self.m.symbols if s.index and not s.defined and s.name]
            self.externs = {s.index: ext.start_ea + 8 * i for i, s in enumerate(imports)}
        self.apply_relocations(annotate=False)

    # -- steps -------------------------------------------------------------

    def setup_processor(self):
        ida_idp.set_processor_type("metapc", ida_idp.SETPROC_LOADER)
        ida_ida.inf_set_app_bitness(64)
        ida_ida.inf_set_filetype(ida_ida.f_ELF)

        cc = ida_ida.compiler_info_t()
        cc.id = ida_typeinf.COMP_GNU
        cc.cm = ida_typeinf.CM_N64 | ida_typeinf.CM_M_NN | ida_typeinf.CM_CC_FASTCALL
        cc.size_i, cc.size_b, cc.size_e, cc.defalign = 4, 1, 4, 0
        cc.size_s, cc.size_l, cc.size_ll, cc.size_ldbl = 2, 8, 8, 16
        ida_typeinf.set_compiler(cc, ida_typeinf.SETCOMP_OVERRIDE)
        ida_typeinf.add_til("gnulnx_x64", ida_typeinf.ADDTIL_DEFAULT)
        self.errno_til = ida_typeinf.add_til(ERRNO_TIL, ida_typeinf.ADDTIL_SILENT) in (
            ida_typeinf.ADDTIL_OK, ida_typeinf.ADDTIL_COMP)

        ida_ida.inf_set_demnames(ida_ida.DEMNAM_GCC3 | ida_ida.DEMNAM_NAME)
        # Keep unreferenced code and don't coagulate unexplored data into blobs
        # (that is what used to swallow vtables).
        ida_ida.inf_set_af(ida_ida.inf_get_af() & ~(ida_ida.AF_UNK | ida_ida.AF_FINAL))

    def compute_layout(self):
        m = self.m
        eh = m.eh_frame()
        if eh:
            self.eh_range = (eh[0], eh[1])
            self.fdes = eh[2]

        # PLT: GOT slots of JUMP_SLOT relocations initially point at `push n`
        # inside the stub; the stub itself is `jmp [rip+slot]` right before it.
        slots = {}
        for off, info, _ in m.jmprel:
            if info & 0xFFFFFFFF == R_X86_64_JUMP_SLOT:
                slots[off] = info >> 32
        for slot, symidx in slots.items():
            raw = m.read_va(slot, 8)
            if raw is None:
                continue
            stub = struct.unpack("<Q", raw)[0] - 6
            code = m.read_va(stub, 6)
            if code and code[:2] == b"\xff\x25" and \
                    stub + 6 + struct.unpack_from("<i", code, 2)[0] == slot:
                self.plt_stubs[symidx] = stub
        if self.plt_stubs:
            lo, hi = min(self.plt_stubs.values()), max(self.plt_stubs.values()) + 16
            plt0 = m.read_va(lo - 16, 2)
            if plt0 == b"\xff\x35":
                lo -= 16
            self.plt_range = (lo, hi)

        pltgot = m.tag(DT_SCE_PLTGOT)
        if pltgot is not None and slots:
            self.got_range = (pltgot, max(pltgot + 24, max(slots) + 8))

        # End of code in the executable segment: everything the unwinder, the
        # PLT and the dynamic entries know about.
        code_ends = [s + n for s, n in self.fdes]
        if self.plt_range:
            code_ends.append(self.plt_range[1])
        for v in (m.entry, m.tag(DT_INIT), m.tag(DT_FINI)):
            if v:
                code_ends.append(v + 1)
        self.code_ends = code_ends

    def plan_segments(self) -> list[Piece]:
        m = self.m
        pieces: list[Piece] = []
        hdr = m.first(PT_GNU_EH_FRAME)
        param = m.first(PT_SCE_PROCPARAM) or m.first(PT_SCE_MODULE_PARAM)
        for p in m.loads:
            perm = (ida_segment.SEGPERM_READ if p.flags & PF_R else 0) | \
                   (ida_segment.SEGPERM_WRITE if p.flags & PF_W else 0) | \
                   (ida_segment.SEGPERM_EXEC if p.flags & PF_X else 0)
            lo, hi = p.vaddr, p.end
            if p.flags & PF_X:
                ends = [e for e in self.code_ends if lo < e <= hi]
                plan = None
                if ends and self.fdes:
                    code_hi = max(ends)
                    carves = []
                    if self.plt_range and lo <= self.plt_range[0] and self.plt_range[1] <= code_hi:
                        carves += [(lo, self.plt_range[0], ".text", "CODE"),
                                   (*self.plt_range, ".plt", "CODE"),
                                   (self.plt_range[1], code_hi, ".text", "CODE")]
                    else:
                        carves.append((lo, code_hi, ".text", "CODE"))
                    if self.eh_range:
                        carves.append((*self.eh_range, ".eh_frame", "CONST"))
                    if hdr and p.contains(hdr.vaddr, hdr.memsz):
                        carves.append((hdr.vaddr, hdr.end, ".eh_frame_hdr", "CONST"))
                    # .rodata & friends sit in the R-X mapping; give them R so IDA
                    # treats them as data, like a section-based ELF load would.
                    plan = carve(lo, hi, (".rodata", "CONST"), perm, carves)
                    if plan:
                        for piece in plan:
                            if piece.sclass != "CODE":
                                piece.perm = ida_segment.SEGPERM_READ
                pieces += plan or [Piece(lo, hi, ".text", "CODE", perm)]
            elif p.type == PT_SCE_RELRO:
                pieces.append(Piece(lo, hi, ".data.rel.ro", "DATA", perm))
            else:
                data_hi = lo + p.filesz
                carves = []
                if param and p.contains(param.vaddr, param.memsz):
                    name = ".sce_process_param" if param.type == PT_SCE_PROCPARAM else ".sce_module_param"
                    carves.append((param.vaddr, param.end, name, "DATA"))
                if self.got_range and lo <= self.got_range[0] and self.got_range[1] <= data_hi:
                    carves.append((*self.got_range, ".got.plt", "DATA"))
                pieces += carve(lo, data_hi, (".data", "DATA"), perm, carves) or \
                    [Piece(lo, data_hi, ".data", "DATA", perm)]
                if p.memsz > p.filesz:
                    pieces.append(Piece(data_hi, hi, ".bss", "BSS", perm))
        return pieces

    def add_segment(self, start, end, name, sclass, perm, seg_type=None):
        seg = ida_segment.segment_t()
        seg.start_ea = start
        seg.end_ea = end
        seg.sel = ida_segment.setup_selector(0)
        seg.bitness = 2
        seg.align = ida_segment.saRelPara
        seg.comb = ida_segment.scPub
        seg.perm = perm
        if seg_type is not None:
            seg.type = seg_type
        flags = ida_segment.ADDSEG_NOSREG | ida_segment.ADDSEG_NOTRUNC | ida_segment.ADDSEG_QUIET
        if not ida_segment.add_segm_ex(seg, name, sclass, flags):
            raise RuntimeError("cannot create segment %s at %#x" % (name, start))
        return ida_segment.getseg(start)

    def create_segments(self):
        types = {"CODE": ida_segment.SEG_CODE, "DATA": ida_segment.SEG_DATA,
                 "CONST": ida_segment.SEG_DATA, "BSS": ida_segment.SEG_BSS}
        for piece in self.plan_segments():
            start, end = self.va(piece.start), self.va(piece.end)
            self.add_segment(start, end, piece.name, piece.sclass, piece.perm, types[piece.sclass])
            if piece.sclass == "CODE":
                self.text.append((start, end))
        self.text.sort()
        for p in self.m.loads:
            if p.filesz:
                off = self.m.container.file_offset(p.offset, p.filesz)
                self.li.file2base(off, self.va(p.vaddr), self.va(p.vaddr + p.filesz),
                                  ida_loader.FILEREG_PATCHABLE)
        ida_nalt.set_imagebase(self.va(self.m.link_base))

    def declare_types(self):
        decls = LOCAL_TYPES
        for type_name, fields, ptype in (("SceProcParam", PROC_PARAM_FIELDS, PT_SCE_PROCPARAM),
                                         ("SceModuleParam", MODULE_PARAM_FIELDS, PT_SCE_MODULE_PARAM)):
            p = self.m.first(ptype)
            raw = self.m.read_va(p.vaddr, 8) if p and p.memsz >= 8 else None
            if raw is None:
                continue
            size = min(struct.unpack("<Q", raw)[0], p.memsz)
            members, used = [], 0
            for off, decl in fields:
                end = off + (4 if decl.startswith("uint32_t ") else 8)
                if end > size:
                    break
                members.append(decl)
                used = end
            if size > used:
                members.append("uint8_t unknown[%d]" % (size - used))
            decls += "struct %s {\n  %s;\n};\n" % (type_name, ";\n  ".join(members))
            self.param_sizes[ptype] = size
        errors = ida_typeinf.parse_decls(None, decls, None, ida_typeinf.HTI_DCL)
        if errors:
            log("%d errors while declaring local types" % errors)

    def named_type(self, name: str) -> ida_typeinf.tinfo_t | None:
        tif = ida_typeinf.tinfo_t()
        return tif if tif.get_named_type(None, name) else None

    def apply_type(self, ea: int, name: str, count: int = 0) -> bool:
        tif = self.named_type(name)
        if tif is None:
            return False
        if count:
            arr = ida_typeinf.tinfo_t()
            if not arr.create_array(tif, count):
                return False
            tif = arr
        ida_bytes.del_items(ea, ida_bytes.DELIT_SIMPLE, tif.get_size())
        return ida_typeinf.apply_tinfo(ea, tif, ida_typeinf.TINFO_DEFINITE)

    def create_imports(self):
        imports = [s for s in self.m.symbols if s.index and not s.defined and s.name]
        if not imports:
            return
        start = align_up(self.image_end, 0x1000)
        end = start + 8 * len(imports)
        self.add_segment(start, end, "extern", "XTRN", 0, ida_segment.SEG_XTRN)
        self.image_end = end

        by_library: dict[str, list[tuple[int, str]]] = {}
        for i, sym in enumerate(imports):
            ea = start + 8 * i
            self.externs[sym.index] = ea
            name = self.sym_name(sym)
            # Functions reached through the PLT get the plain name on the stub;
            # the extern keeps the __imp_ form (PE-style, demangles fine).
            extern_name = "__imp_" + name if sym.index in self.plt_stubs else name
            self.set_name(ea, extern_name)
            cmt = self.sym_comment(sym)
            if cmt:
                ida_bytes.set_cmt(ea, cmt, True)
            if sym.index not in self.plt_stubs:
                ida_typeinf.apply_named_type(ea, name)
            by_library.setdefault(self.library_of(sym), []).append((ea, name))

        for library, entries in sorted(by_library.items()):
            node = ida_netnode.netnode()
            node.create()
            for ea, name in entries:
                ida_loader.set_import_name(node.index(), ea, name)
            ida_loader.import_module(library, None, node.index(), None, None)
        self.stats["imports"] = len(imports)

    def symbol_value(self, index: int) -> int | None:
        if index in self.externs:
            return self.externs[index]
        if 0 < index < len(self.m.symbols):
            sym = self.m.symbols[index]
            if sym.defined:
                return self.va(sym.value)
        return None

    def apply_relocations(self, annotate: bool = True):
        put_qword = ida_bytes.put_qword
        fixup = ida_fixup.fixup_data_t(ida_fixup.FIXUP_OFF64)
        in_text = self.in_text
        delta = self.delta
        sites = []
        unknown: dict[int, int] = {}
        count = 0
        for table in (self.m.rela, self.m.jmprel):
            for off, info, addend in table:
                rtype, symidx = info & 0xFFFFFFFF, info >> 32
                ea = (off + delta) & MASK64
                if rtype == R_X86_64_RELATIVE:
                    value = (addend + delta) & MASK64
                elif rtype in (R_X86_64_64, R_X86_64_GLOB_DAT, R_X86_64_JUMP_SLOT):
                    target = self.symbol_value(symidx)
                    if target is None:
                        unknown[rtype] = unknown.get(rtype, 0) + 1
                        continue
                    value = (target + addend) & MASK64 if rtype == R_X86_64_64 else target
                elif rtype in (R_X86_64_DTPMOD64, R_X86_64_DTPOFF64, R_X86_64_TPOFF64):
                    if annotate:
                        sym = self.m.symbols[symidx] if 0 < symidx < len(self.m.symbols) else None
                        what = {R_X86_64_DTPMOD64: "TLS module id",
                                R_X86_64_DTPOFF64: "TLS offset (DTPOFF64)",
                                R_X86_64_TPOFF64: "TLS offset (TPOFF64)"}[rtype]
                        ida_bytes.create_qword(ea, 8, True)
                        ida_bytes.set_cmt(ea, what + (" of " + self.sym_name(sym) if sym else ""), False)
                    continue
                elif rtype == R_X86_64_NONE:
                    continue
                else:
                    unknown[rtype] = unknown.get(rtype, 0) + 1
                    continue
                put_qword(ea, value)
                fixup.off = value
                fixup.set(ea)
                sites.append(off)
                count += 1
                if annotate and not in_text(ea):
                    ida_bytes.create_qword(ea, 8, True)
                    ida_offset.op_plain_offset(ea, 0, 0)
                    if rtype == R_X86_64_GLOB_DAT:
                        self.set_name(ea, self.sym_name(self.m.symbols[symidx]) + "_ptr")
        sites.sort()
        self.reloc_sites = sites
        self.stats["relocations"] = count
        if unknown:
            log("unhandled relocations: " + ", ".join("type %d x%d" % kv for kv in sorted(unknown.items())))

    def process_plt(self):
        if self.plt_range:
            lo = self.va(self.plt_range[0])
            if lo != self.va(min(self.plt_stubs.values())):
                self.set_name(lo, "_PLT0")
                ida_auto.auto_make_proc(lo)
        if self.got_range:
            self.set_name(self.va(self.got_range[0]), "_GLOBAL_OFFSET_TABLE_")
        for symidx, stub in self.plt_stubs.items():
            ea = self.va(stub)
            sym = self.m.symbols[symidx]
            name = self.sym_name(sym)
            self.set_name(ea, name)
            ida_ua.create_insn(ea)
            if ida_funcs.add_func(ea, BADADDR):
                pfn = ida_funcs.get_func(ea)
                if pfn and name in NORETURN_NAMES:
                    pfn.flags |= ida_funcs.FUNC_NORET
                    ida_funcs.update_func(pfn)
            ida_typeinf.apply_named_type(ea, name)
            cmt = self.sym_comment(sym)
            if cmt:
                ida_bytes.set_cmt(ea, cmt, True)

    def process_exports(self):
        count = 0
        for sym in self.m.symbols:
            if not sym.defined or not sym.value or sym.bind not in (STB_GLOBAL, STB_WEAK):
                continue
            if sym.type not in (STT_FUNC, STT_OBJECT, STT_NOTYPE):
                continue
            ea = self.va(sym.value)
            if not ida_segment.getseg(ea):
                continue
            name = self.sym_name(sym)
            is_func = sym.type == STT_FUNC
            self.set_name(ea, name)
            ida_entry.add_entry(ea, ea, ida_name.get_name(ea), is_func)
            if is_func:
                ida_auto.auto_make_proc(ea)
            cmt = self.sym_comment(sym)
            if cmt:
                ida_bytes.set_cmt(ea, cmt, True)
            count += 1
        self.stats["exports"] = count

    def process_entry_points(self):
        m = self.m
        if m.entry and m.load_for(m.entry):
            ea = self.va(m.entry)
            ida_entry.add_entry(ea, ea, "_start", True)
            # start_ea is recomputed from cs:ip, so cs must be a flat selector
            ida_ida.inf_set_start_cs(ida_segment.getseg(ea).sel)
            ida_ida.inf_set_start_ip(ea)
            ida_ida.inf_set_start_ea(ea)
        for tag, name in ((DT_INIT, "_init"), (DT_FINI, "_fini")):
            v = m.tag(tag)
            if v and m.load_for(v):
                ea = self.va(v)
                ida_entry.add_entry(ea, ea, name, True)
        for arr_tag, size_tag, name in ((DT_PREINIT_ARRAY, DT_PREINIT_ARRAYSZ, "__preinit_array"),
                                        (DT_INIT_ARRAY, DT_INIT_ARRAYSZ, "__init_array"),
                                        (DT_FINI_ARRAY, DT_FINI_ARRAYSZ, "__fini_array")):
            v, size = m.tag(arr_tag), m.tag(size_tag, 0)
            if not v or not size or not m.load_for(v, size):
                continue
            ea = self.va(v)
            self.set_name(ea, name + "_start")
            for slot in range(ea, ea + size - size % 8, 8):
                ida_bytes.create_qword(slot, 8, True)
                ida_offset.op_plain_offset(slot, 0, 0)
                target = ida_bytes.get_qword(slot)
                if self.in_text(target):
                    ida_auto.auto_make_proc(target)
        tls = m.first(PT_TLS)
        if tls and m.load_for(tls.vaddr):
            ida_bytes.set_cmt(self.va(tls.vaddr),
                              "TLS template: %#x initialised + %#x zeroed bytes, align %#x" % (
                                  tls.filesz, tls.memsz - tls.filesz, tls.align), False)

    def process_params(self):
        for ptype, type_name, label, magic in (
                (PT_SCE_PROCPARAM, "SceProcParam", "sce_process_param", PROC_PARAM_MAGIC),
                (PT_SCE_MODULE_PARAM, "SceModuleParam", "sce_module_param", MODULE_PARAM_MAGIC)):
            p = self.m.first(ptype)
            if not p or not self.m.load_for(p.vaddr, 0x18):
                continue
            ea = self.va(p.vaddr)
            self.set_name(ea, label)
            self.apply_type(ea, type_name)
            got_magic, _, sdk = struct.unpack("<IIQ", self.m.read_va(p.vaddr + 8, 16))
            if got_magic == magic:
                self.sdk_version = sdk
                ida_bytes.set_cmt(ea, "SDK %s" % sdk_version_str(sdk), False)
            if ptype == PT_SCE_PROCPARAM:
                libc = ida_bytes.get_qword(ea + 0x38) if self.param_sizes.get(ptype, 0) >= 0x40 else 0
                if libc and ida_segment.getseg(libc):
                    self.set_name(libc, "sce_libc_param")

    def process_shaders(self):
        shaders = self.m.scan_shaders()
        if not shaders:
            return
        # Glue the reflection data sitting between consecutive shader blobs onto
        # the preceding blob, unless something that looks like real data or code
        # (relocated pointer, function start) lives in the gap.
        barriers = sorted(set(self.reloc_sites) | {s for s, _ in self.fdes})

        def clear(lo, hi):
            i = bisect.bisect_left(barriers, lo)
            return i == len(barriers) or barriers[i] >= hi

        for a, b in zip(shaders, shaders[1:] + [None]):
            if a.container_end > a.end and (b is None or a.container_end <= b.start) \
                    and clear(a.end, a.container_end):
                a.end = a.container_end
        for a, b in zip(shaders, shaders[1:]):
            if 0 < b.start - a.end <= GCN_MAX_GAP and self.m.load_for(a.end, b.start - a.end) \
                    and clear(a.end, b.start):
                a.end = b.start

        skipped = total = 0
        for sh in shaders:
            i = bisect.bisect_left(self.reloc_sites, sh.start)
            if i < len(self.reloc_sites) and self.reloc_sites[i] < sh.end:
                skipped += 1                         # carries pointers: not a pure blob
                continue
            start, code, info, end = (self.va(x) for x in (sh.start, sh.code, sh.info, sh.end))
            self.set_name(start, "gcn_shader_%016X" % sh.hash if sh.hash else "gcn_shader")
            ida_bytes.set_cmt(start, "GCN shader: type %d, code %#x bytes @ %#x, hash %016X, crc %08X"
                              % (sh.type, sh.length, code, sh.hash, sh.crc), False)
            if self.o.skip_shaders:
                ida_bytes.del_items(start, ida_bytes.DELIT_SIMPLE, end - start)
                ida_bytes.create_byte(start, info - start, True)
                self.apply_type(info, "ShaderBinaryInfo")
                tail = info + GCN_INFO_SIZE
                if end > tail:
                    ida_bytes.create_byte(tail, end - tail, True)
            total += end - start
        self.stats["shaders"] = "%d (%#x bytes)" % (len(shaders) - skipped, total)

    def seed_functions(self):
        if self.o.eh_frame:
            make = ida_auto.auto_make_proc
            in_text = self.in_text
            for start, size in self.fdes:
                ea = self.va(start)
                if in_text(ea):
                    make(ea)
                    self.post.fdes.append((ea, size))
            self.stats["fdes"] = len(self.fdes)
        if not self.fdes:
            # No unwind info: relocated pointers into code are the next best hint.
            for off, info, addend in self.m.rela:
                if info & 0xFFFFFFFF == R_X86_64_RELATIVE:
                    ea = self.va(addend)
                    if self.in_text(ea):
                        ida_auto.auto_make_proc(ea)

    def map_dynlib(self):
        m = self.m
        if not m.dynlib:
            return
        start = align_up(self.image_end, 0x1000)
        end = start + len(m.dynlib)
        self.add_segment(start, end, ".sce_dynlibdata", "CONST", ida_segment.SEGPERM_READ,
                         ida_segment.SEG_DATA)
        self.image_end = end
        ida_loader.mem2base(m.dynlib, start, m.container.file_offset(m.dynlib_phdr.offset, 0))

        fp = m.fingerprint()
        if fp:
            ea = start + m.tag(DT_SCE_FINGERPRINT)
            self.set_name(ea, "sce_fingerprint")
            ida_bytes.create_byte(ea, 20, True)
            ida_bytes.set_cmt(ea, fp.hex().upper(), False)

        def table(off_tag, size_tag, name, type_name, entsize):
            off, size = m.tag(off_tag), m.tag(size_tag, 0)
            if off is None or not size or off + size > len(m.dynlib):
                return
            self.set_name(start + off, name)
            if type_name:
                self.apply_type(start + off, type_name, size // entsize)

        table(DT_SCE_SYMTAB, DT_SCE_SYMTABSZ, "sce_dynsym", "Elf64_Sym", 24)
        table(DT_SCE_RELA, DT_SCE_RELASZ, "sce_rela", "Elf64_Rela", 24)
        table(DT_SCE_JMPREL, DT_SCE_PLTRELSZ, "sce_jmprel", "Elf64_Rela", 24)
        table(DT_SCE_STRTAB, DT_SCE_STRSZ, "sce_dynstr", None, 1)
        table(DT_SCE_HASH, DT_SCE_HASHSZ, "sce_hash", None, 4)

        if m.dynamic_offset is not None:
            ea = start + m.dynamic_offset
            self.set_name(ea, "_DYNAMIC")
            dyn_t = self.named_type("Elf64_Dyn")
            for i, (tag, value) in enumerate(m.dynamic):
                item = ea + 16 * i
                if dyn_t is not None:
                    ida_typeinf.apply_tinfo(item, dyn_t, ida_typeinf.TINFO_DEFINITE)
                ida_bytes.set_cmt(item, m.describe_dynamic(tag, value), False)

    def comment_syscalls(self):
        fde_starts = {s for s, _ in self.fdes}
        count = 0
        for p in self.m.loads:
            if not p.flags & PF_X:
                continue
            buf = self.m.segment_bytes(p)
            for match in SYSCALL_RE.finditer(buf):
                number = struct.unpack("<i", match.group(1))[0]
                if not 0 <= number < len(SYSCALL_NAMES):
                    continue
                name = SYSCALL_NAMES[number]
                link = p.vaddr + match.start()
                ea = self.va(link)
                if not self.in_text(ea):
                    continue
                label = "sys_%s" % name if name != "-" else "sys_%d" % number
                ida_bytes.set_cmt(self.va(link + match.end() - match.start() - 2),
                                  "syscall %d: %s" % (number, label), False)
                if link in fde_starts and not ida_name.get_name(ea):
                    self.set_name(ea, "__" + label)
                count += 1
        self.stats["syscalls"] = count

    def schedule_error_codes(self):
        if not self.errno_til:
            log("%s.til not found; skipping error-code pass" % ERRNO_TIL)
            return
        tif = self.named_type(ERRNO_ENUM)
        if tif is None or not tif.is_enum():
            return
        tid = tif.force_tid()           # copies the enum into local types
        edm = ida_typeinf.enum_type_data_t()
        if tid == BADADDR or not tif.get_enum_details(edm):
            return
        values = {e.value & 0xFFFFFFFF for e in edm}
        candidates = []
        for p in self.m.loads:
            if not p.flags & PF_X:
                continue
            buf = self.m.segment_bytes(p)
            pos = buf.find(b"\x80", 3)
            while pos >= 0:
                value = int.from_bytes(buf[pos - 3:pos + 1], "little")
                if value in values:
                    ea = self.va(p.vaddr + pos - 3)
                    if self.in_text(ea):
                        candidates.append((ea, value))
                pos = buf.find(b"\x80", pos + 1)
        self.post.enum_tid = tid
        self.post.candidates = candidates
        self.stats["errno candidates"] = len(candidates)

    def write_summary(self):
        m = self.m
        lines = ["PlayStation 4 %s" % ET_NAMES.get(m.e_type, "module %#x" % m.e_type)]
        if m.container.is_self:
            lines.append("Container: fake SELF")
        if m.module_info:
            lines.append("Module: %s v%d.%d%s" % (
                m.module_info.name, m.module_info.major, m.module_info.minor,
                ", attr " + flags_str(m.module_info.attr, MODULE_ATTRS) if m.module_info.attr else ""))
        original = m.tag(DT_SCE_ORIGINAL_FILENAME)
        original = m.string(original) if original is not None else None
        if original:
            lines.append("Original file: %s" % original)
        comment = m.sce_comment()
        if comment and comment != original:
            lines.append("Build path: %s" % comment)
        if self.sdk_version:
            lines.append("SDK: %s" % sdk_version_str(self.sdk_version))
        fp = m.fingerprint()
        if fp:
            lines.append("Fingerprint: %s" % fp.hex().upper())
        lines.append("Image base: %#x (linked at %#x)" % (self.va(m.link_base), m.link_base))
        if m.modules:
            lines.append("Needed modules:")
            for mod in sorted(m.modules.values(), key=lambda x: x.id):
                lines.append("  [%2d] %s v%d.%d" % (mod.id, mod.name, mod.major, mod.minor))
        libs = sorted(m.import_libs.values(), key=lambda x: x.id) + \
            sorted(m.export_libs.values(), key=lambda x: x.id)
        if libs:
            lines.append("Libraries:")
            for lib in libs:
                lines.append("  [%2d] %s %s v%d%s" % (
                    lib.id, "export" if lib.exported else "import", lib.name, lib.version,
                    ", " + flags_str(lib.attr, LIBRARY_ATTRS) if lib.attr else ""))
        versions = m.lib_versions()
        if versions:
            lines.append("Linked library versions:")
            seen = set()
            for name, version in versions:
                if (name, version) not in seen:
                    seen.add((name, version))
                    lines.append("  %s %08X" % (name, version))
        for line in lines:
            ida_lines.add_pgm_cmt(line.replace("%", "%%"))
        log(" | ".join(lines[:4]))
        log("stats: " + ", ".join("%s=%s" % kv for kv in self.stats.items()))

    def save_state(self):
        node = ida_netnode.netnode(STATE_NODE, 0, True)
        node.altset(0, self.va(self.m.link_base))


# ---------------------------------------------------------------------------
# Loader entry points
# ---------------------------------------------------------------------------

def _probe(li) -> tuple[int, bool] | None:
    """Cheap check used by accept_file: returns (e_type, is_self) for PS4 modules."""
    li.seek(0)
    head = li.read(0x20)
    if len(head) < 0x20:
        return None
    elf_off = 0
    is_self = head[:4] == SELF_MAGIC
    if is_self:
        elf_off = 0x20 + struct.unpack_from("<H", head, 0x18)[0] * 0x20
    elif head[:4] != ELF_MAGIC:
        return None
    li.seek(elf_off)
    ehdr = li.read(0x40)
    if len(ehdr) < 0x40 or ehdr[:4] != ELF_MAGIC or ehdr[4] != 2 or ehdr[5] != 1:
        return None
    e_type, machine = struct.unpack_from("<HH", ehdr, 16)
    phoff = struct.unpack_from("<Q", ehdr, 32)[0]
    phentsize, phnum = struct.unpack_from("<HH", ehdr, 54)
    if machine != EM_X86_64:
        return None
    if e_type in (ET_SCE_EXEC, ET_SCE_REPLAY_EXEC, ET_SCE_RELEXEC, ET_SCE_STUBLIB,
                  ET_SCE_DYNEXEC, ET_SCE_DYNAMIC):
        return e_type, is_self
    if e_type not in (ET_EXEC, ET_DYN) or phentsize != 56 or not 0 < phnum < 64:
        return None
    # Plain ELF types are only claimed when SCE dynamic data is present, so
    # FreeBSD/Linux binaries and the PS4 kernel are left to other loaders.
    li.seek(elf_off + phoff)
    table = li.read(phnum * 56)
    for i in range(len(table) // 56):
        if struct.unpack_from("<I", table, i * 56)[0] == PT_SCE_DYNLIBDATA:
            return e_type, is_self
    return None


def accept_file(li, filename):
    try:
        probe = _probe(li)
    except Exception:
        return 0
    if probe is None:
        return 0
    e_type, is_self = probe
    return {
        "format": "PlayStation 4 %s (%s)" % (ET_NAMES.get(e_type, "module"),
                                             "fake SELF" if is_self else "ELF"),
        "processor": "metapc",
        "options": 1 | ida_loader.ACCEPT_FIRST,
    }


def load_file(li, neflags, fmt):
    li.seek(0)
    data = li.read(li.size())
    try:
        mod = PS4Module(data)
    except FormatError as exc:
        ida_kernwin.warning("ps4ida: %s" % exc)
        return 0

    if neflags & ida_loader.NEF_RELOAD:
        Loader(li, mod, Options.defaults(mod)).reload()
        return 1

    opts = Options.defaults(mod)
    if neflags & ida_loader.NEF_MAN and not ida_kernwin.cvar.batch:
        if not opts.ask(mod):
            return 0
    try:
        Loader(li, mod, opts).run()
    except FormatError as exc:
        ida_kernwin.warning("ps4ida: %s" % exc)
        return 0
    return 1
