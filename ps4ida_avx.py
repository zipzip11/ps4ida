"""
ps4ida_avx.py -- Hex-Rays microcode filter that lifts AVX (VEX) code.

The x64 decompiler understands SSE but leaves VEX-encoded instructions
(vmovaps, vxorps, vmulps, ...) as __asm blocks. This filter lifts them:

* 128-bit VEX instructions are rewritten into their SSE twin and handed to the
  decompiler's own SSE lifter (types, _mm_* intrinsics, constant propagation):
      vaddps d, s1, s2   ->  movaps d, s1 ; addps d, s2
* vpshufd/vmovdqa feeding float code are typed as their float twins
  (_mm_shuffle_ps, movaps) to avoid __m128i casts.
* Lane-extract idioms (vpshufd/vpermilps/shufps with only the low selector
  set, vmovhlps/vunpckhpd x, y, y) become plain lane moves, so the decompiler
  sees `v.m128_f32[3]` instead of a shuffle intrinsic.
* roundss/roundsd with a fixed mode become floorf/ceilf/truncf/rintf calls.
* Everything else -- 256-bit ymm code, non-commutative ops whose destination
  is the second source, 4-operand blends, broadcasts, 128-bit lane inserts and
  extracts -- becomes a call to the matching _mm_* / _mm256_* intrinsic.

The decompiler keeps xmmN and ymmN in separate registers: ymm writes mirror
their low half into xmmN, and ymm reads refresh their low half from xmmN only
where an xmm-only write can reach them (per-function reaching definitions).
VEX-128 zeroing of bits 255:128 is not modelled.

Install: copy to <IDAUSR>/plugins. Edit > Plugins > "ps4ida AVX lifter"
toggles it for the current database.
"""

import ida_allins
import ida_bytes
import ida_funcs
import ida_gdl
import ida_hexrays as hr
import ida_ida
import ida_idaapi
import ida_idp
import ida_kernwin
import ida_typeinf
import ida_ua

NN = ida_allins


def _ids(names: str) -> set:
    return {getattr(NN, "NN_" + n) for n in names.split() if hasattr(NN, "NN_" + n)}


def _twins(names: str) -> dict:
    """Map 'vfoo' -> 'foo' instruction ids for every pair this IDA knows."""
    out = {}
    for name in names.split():
        vex, sse = getattr(NN, "NN_" + name, None), getattr(NN, "NN_" + name[1:], None)
        if vex is not None and sse is not None:
            out[vex] = sse
    return out


_CMP = "eq lt le unord neq nlt nle ord"

# Same operand list as the SSE instruction.
SAME_FORM = _twins("""
    vmovaps vmovups vmovdqa vmovdqu vmovapd vmovupd vmovd vmovq vmovntps vmovntpd vmovntdq vlddqu
    vmovhps vmovlps vmovhpd vmovlpd vmovshdup vmovsldup vmovddup vmovmskps vmovmskpd vpmovmskb
    vucomiss vcomiss vucomisd vcomisd vptest
    vcvttss2si vcvtss2si vcvttsd2si vcvtsd2si vcvtdq2ps vcvttps2dq vcvtps2dq vcvtps2pd vcvtpd2ps
    vcvtdq2pd vcvttpd2dq vcvtpd2dq
    vsqrtps vsqrtpd vrsqrtps vrcpps vroundps vroundpd
    vpshufd vpshuflw vpshufhw vpextrb vpextrw vpextrd vpextrq vextractps
    vpmovsxbw vpmovsxbd vpmovsxbq vpmovsxwd vpmovsxwq vpmovsxdq
    vpmovzxbw vpmovzxbd vpmovzxbq vpmovzxwd vpmovzxwq vpmovzxdq
    vpabsb vpabsw vpabsd vphminposuw vmovntdqa vstmxcsr vldmxcsr
""")

# VEX "d, s1, s2[, imm]" == "movaps d, s1 ; OP d, s2[, imm]".
NDS_FORM = _twins("""
    vaddps vsubps vmulps vdivps vminps vmaxps vandps vandnps vorps vxorps
    vaddpd vsubpd vmulpd vdivpd vminpd vmaxpd vandpd vandnpd vorpd vxorpd
    vaddss vsubss vmulss vdivss vminss vmaxss vsqrtss vrcpss vrsqrtss vcmpss
    vaddsd vsubsd vmulsd vdivsd vminsd vmaxsd vsqrtsd vcmpsd
    vaddsubps vaddsubpd vhaddps vhaddpd vhsubps vhsubpd vdpps vdppd vblendps vblendpd
    vshufps vshufpd vunpcklps vunpckhps vunpcklpd vunpckhpd vinsertps vmovhlps vmovlhps
    vcmpps vcmppd vcvtsi2ss vcvtsi2sd vcvtss2sd vcvtsd2ss
    vpxor vpand vpandn vpor
    vpaddb vpaddw vpaddd vpaddq vpsubb vpsubw vpsubd vpsubq vpaddsb vpaddsw vpaddusb vpaddusw
    vpsubsb vpsubsw vpsubusb vpsubusw vpmulld vpmullw vpmulhw vpmulhuw vpmuludq vpmuldq vpmaddwd
    vpcmpeqb vpcmpeqw vpcmpeqd vpcmpeqq vpcmpgtb vpcmpgtw vpcmpgtd vpcmpgtq
    vpminsb vpminsw vpminsd vpminub vpminuw vpminud vpmaxsb vpmaxsw vpmaxsd vpmaxub vpmaxuw vpmaxud
    vpavgb vpavgw vpsadbw vpsignb vpsignw vpsignd vphaddw vphaddd vphsubw vphsubd
    vpsllw vpslld vpsllq vpsrlw vpsrld vpsrlq vpsraw vpsrad vpslldq vpsrldq
    vpunpcklbw vpunpckhbw vpunpcklwd vpunpckhwd vpunpckldq vpunpckhdq vpunpcklqdq vpunpckhqdq
    vpacksswb vpackssdw vpackuswb vpackusdw vpalignr vpshufb vpblendw
    vpinsrb vpinsrw vpinsrd vpinsrq vmovlps vmovhps vmovlpd vmovhpd
""" + " ".join("vcmp%s%s" % (p, k) for p in _CMP.split() for k in ("ps", "pd", "ss", "sd")))

# Packed ops where "d, s1, d" can be lifted as "OP d, s1".
COMMUTATIVE = set(_twins("""
    vaddps vmulps vandps vorps vxorps vaddpd vmulpd vandpd vorpd
    vpxor vpand vpor vpaddb vpaddw vpaddd vpaddq vpmulld vpmullw vpmuludq
    vpcmpeqb vpcmpeqw vpcmpeqd vpcmpeqq vpavgb vpavgw
    vpminsb vpminsw vpminsd vpminub vpminuw vpminud vpmaxsb vpmaxsw vpmaxsd vpmaxub vpmaxuw vpmaxud
    vpaddsb vpaddsw vpaddusb vpaddusw vpmulhw vpmulhuw vpmuldq vpmaddwd vpsadbw
""")) | set(_twins(" ".join("vcmp%sps vcmp%spd" % (p, p) for p in "eq neq unord ord".split())))

# "OP d, s, s" whose result is 0 whatever s holds.
ZERO_IDIOM = _ids("vxorps vxorpd vpxor vpsubb vpsubw vpsubd vpsubq vandnps vandnpd vpandn")

SCALAR_MOVES = {NN.NN_vmovss: NN.NN_movss, NN.NN_vmovsd: NN.NN_movsd}

# Scalar float arithmetic lifted straight to float microcode when the
# destination is also the second source (no SSE rewrite expresses that).
SCALAR_FP = {
    NN.NN_vaddss: (hr.m_fadd, 4), NN.NN_vaddsd: (hr.m_fadd, 8),
    NN.NN_vsubss: (hr.m_fsub, 4), NN.NN_vsubsd: (hr.m_fsub, 8),
    NN.NN_vmulss: (hr.m_fmul, 4), NN.NN_vmulsd: (hr.m_fmul, 8),
    NN.NN_vdivss: (hr.m_fdiv, 4), NN.NN_vdivsd: (hr.m_fdiv, 8),
}
SCALAR_CVT = {NN.NN_vcvtss2sd: (4, 8), NN.NN_vcvtsd2ss: (8, 4)}
# ... and the libm equivalents (args: s1.lo, s2.lo; sqrt takes s2 only)
SCALAR_CALL = {
    NN.NN_vsqrtss: ("sqrtf", 4, 1), NN.NN_vsqrtsd: ("sqrt", 8, 1),
    NN.NN_vminss: ("fminf", 4, 2), NN.NN_vmaxss: ("fmaxf", 4, 2),
    NN.NN_vminsd: ("fmin", 8, 2), NN.NN_vmaxsd: ("fmax", 8, 2),
}

# pshufd-style shuffles: d.lane[i] = s.lane[(imm >> 2i) & 3]
LANE_SHUFFLE = _ids("pshufd vpshufd vpermilps")
SHUFPS_SAME = _ids("shufps vshufps")
HIGH_HALF = _ids("vmovhlps vunpckhpd movhlps unpckhpd")
PERMILPD = _ids("vpermilpd")
ROUND_SCALAR = {NN.NN_roundss: 4, NN.NN_roundsd: 8, NN.NN_vroundss: 4, NN.NN_vroundsd: 8}
ROUND_NAMES = ("rint", "floor", "ceil", "trunc")

# Intrinsic fallbacks: itype -> (name without _mm_/_mm256_ prefix, result, args)
# Type tokens: V = vector of the instruction width, X = 128-bit vector,
# suffix = element kind (ps/pd/i); f/d = float/double scalar; I = immediate; n = int.
INTRINSICS = {}


def _intr(names: str, kind: str, result: str, args: tuple, fmt: str = "{}_{k}"):
    for name in names.split():
        itype = getattr(NN, "NN_v" + name + ("" if kind == "i" else kind), None)
        if itype is not None:
            base = name if kind == "i" else fmt.format(name, k=kind)
            INTRINSICS[itype] = (base, result, args)


for _k in ("ps", "pd"):
    _V = "V" + _k
    _intr("add sub mul div min max and or xor addsub hadd hsub", _k, _V, (_V, _V))
    _intr("unpckl unpckh", _k, _V, (_V, _V))
    _intr("shuf blend", _k, _V, (_V, _V, "I"))
    _intr("sqrt rsqrt rcp", _k, _V, (_V,))
    _intr("round", _k, _V, (_V, "I"))
    _intr("blendv", _k, _V, (_V, _V, _V))
    for _p in _CMP.split():
        _intr("cmp" + _p, _k, _V, (_V, _V))
INTRINSICS[NN.NN_vandnps] = ("andnot_ps", "Vps", ("Vps", "Vps"))
INTRINSICS[NN.NN_vandnpd] = ("andnot_pd", "Vpd", ("Vpd", "Vpd"))
INTRINSICS[NN.NN_vunpcklps] = ("unpacklo_ps", "Vps", ("Vps", "Vps"))
INTRINSICS[NN.NN_vunpckhps] = ("unpackhi_ps", "Vps", ("Vps", "Vps"))
INTRINSICS[NN.NN_vunpcklpd] = ("unpacklo_pd", "Vpd", ("Vpd", "Vpd"))
INTRINSICS[NN.NN_vunpckhpd] = ("unpackhi_pd", "Vpd", ("Vpd", "Vpd"))
INTRINSICS[NN.NN_vshufps] = ("shuffle_ps", "Vps", ("Vps", "Vps", "I"))
INTRINSICS[NN.NN_vshufpd] = ("shuffle_pd", "Vpd", ("Vpd", "Vpd", "I"))
INTRINSICS[NN.NN_vdpps] = ("dp_ps", "Vps", ("Vps", "Vps", "I"))
INTRINSICS[NN.NN_vmovshdup] = ("movehdup_ps", "Vps", ("Vps",))
INTRINSICS[NN.NN_vmovsldup] = ("moveldup_ps", "Vps", ("Vps",))
INTRINSICS[NN.NN_vmovddup] = ("movedup_pd", "Vpd", ("Vpd",))
INTRINSICS[NN.NN_vperm2f128] = ("permute2f128_ps", "Vps", ("Vps", "Vps", "I"))
INTRINSICS[NN.NN_vinsertf128] = ("insertf128_ps", "Vps", ("Vps", "Xps", "I"))
INTRINSICS[NN.NN_vextractf128] = ("extractf128_ps", "Xps", ("Vps", "I"))
INTRINSICS[NN.NN_vbroadcastss] = ("set1_ps", "Vps", ("f",))
INTRINSICS[NN.NN_vbroadcastsd] = ("set1_pd", "Vpd", ("d",))
INTRINSICS[NN.NN_vbroadcastf128] = ("broadcast_ps", "Vps", ("Xps",))
INTRINSICS[NN.NN_vcvtdq2ps] = ("cvtepi32_ps", "Vps", ("Vi",))
INTRINSICS[NN.NN_vcvttps2dq] = ("cvttps_epi32", "Vi", ("Vps",))
INTRINSICS[NN.NN_vcvtps2dq] = ("cvtps_epi32", "Vi", ("Vps",))
INTRINSICS[NN.NN_vcvtps2pd] = ("cvtps_pd", "Vpd", ("Xps",))
INTRINSICS[NN.NN_vcvtpd2ps] = ("cvtpd_ps", "Xps", ("Vpd",))
INTRINSICS[NN.NN_vcvtdq2pd] = ("cvtepi32_pd", "Vpd", ("Xi",))
INTRINSICS[NN.NN_vcvttpd2dq] = ("cvttpd_epi32", "Xi", ("Vpd",))
INTRINSICS[NN.NN_vmovmskps] = ("movemask_ps", "n", ("Vps",))
INTRINSICS[NN.NN_vmovmskpd] = ("movemask_pd", "n", ("Vpd",))
# 128-bit integer ops, "vpXXX" -> "_mm_XXX" (shift counts given as an
# immediate switch to the slli/srli/srai names in _intrinsic)
INT_INTRINSICS = """
    vpaddb add_epi8 vpaddw add_epi16 vpaddd add_epi32 vpaddq add_epi64
    vpsubb sub_epi8 vpsubw sub_epi16 vpsubd sub_epi32 vpsubq sub_epi64
    vpaddsb adds_epi8 vpaddsw adds_epi16 vpaddusb adds_epu8 vpaddusw adds_epu16
    vpsubsb subs_epi8 vpsubsw subs_epi16 vpsubusb subs_epu8 vpsubusw subs_epu16
    vpmullw mullo_epi16 vpmulld mullo_epi32 vpmulhw mulhi_epi16 vpmulhuw mulhi_epu16
    vpmuludq mul_epu32 vpmuldq mul_epi32 vpmaddwd madd_epi16 vpsadbw sad_epu8
    vpand and_si128 vpandn andnot_si128 vpor or_si128 vpxor xor_si128
    vpcmpeqb cmpeq_epi8 vpcmpeqw cmpeq_epi16 vpcmpeqd cmpeq_epi32 vpcmpeqq cmpeq_epi64
    vpcmpgtb cmpgt_epi8 vpcmpgtw cmpgt_epi16 vpcmpgtd cmpgt_epi32 vpcmpgtq cmpgt_epi64
    vpminsb min_epi8 vpminsw min_epi16 vpminsd min_epi32 vpminub min_epu8 vpminuw min_epu16
    vpminud min_epu32 vpmaxsb max_epi8 vpmaxsw max_epi16 vpmaxsd max_epi32 vpmaxub max_epu8
    vpmaxuw max_epu16 vpmaxud max_epu32 vpavgb avg_epu8 vpavgw avg_epu16
    vpsignb sign_epi8 vpsignw sign_epi16 vpsignd sign_epi32
    vphaddw hadd_epi16 vphaddd hadd_epi32 vphsubw hsub_epi16 vphsubd hsub_epi32
    vpunpcklbw unpacklo_epi8 vpunpckhbw unpackhi_epi8 vpunpcklwd unpacklo_epi16
    vpunpckhwd unpackhi_epi16 vpunpckldq unpacklo_epi32 vpunpckhdq unpackhi_epi32
    vpunpcklqdq unpacklo_epi64 vpunpckhqdq unpackhi_epi64
    vpacksswb packs_epi16 vpackssdw packs_epi32 vpackuswb packus_epi16 vpackusdw packus_epi32
    vpshufb shuffle_epi8
    vpsllw sll_epi16 vpslld sll_epi32 vpsllq sll_epi64 vpsrlw srl_epi16 vpsrld srl_epi32
    vpsrlq srl_epi64 vpsraw sra_epi16 vpsrad sra_epi32
""".split()
for _name, _base in zip(INT_INTRINSICS[::2], INT_INTRINSICS[1::2]):
    if hasattr(NN, "NN_" + _name):
        INTRINSICS[getattr(NN, "NN_" + _name)] = (_base, "Vi", ("Vi", "Vi"))
for _name, _base in (("vpslldq", "slli_si128"), ("vpsrldq", "srli_si128")):
    INTRINSICS[getattr(NN, "NN_" + _name)] = (_base, "Vi", ("Vi", "I"))
INTRINSICS[NN.NN_vpblendw] = ("blend_epi16", "Vi", ("Vi", "Vi", "I"))
INTRINSICS[NN.NN_vdppd] = ("dp_pd", "Vpd", ("Vpd", "Vpd", "I"))
INTRINSICS[NN.NN_vpalignr] = ("alignr_epi8", "Vi", ("Vi", "Vi", "I"))
INTRINSICS[NN.NN_vpermilps] = ("permute_ps", "Vps", ("Vps", "I"))
INTRINSICS[NN.NN_vpermilpd] = ("permute_pd", "Vpd", ("Vpd", "I"))
INTRINSICS[NN.NN_vpblendvb] = ("blendv_epi8", "Vi", ("Vi", "Vi", "Vi"))
for _p in _CMP.split():
    for _k, _v in (("ss", "Vps"), ("sd", "Vpd")):
        if hasattr(NN, "NN_vcmp%s%s" % (_p, _k)):
            INTRINSICS[getattr(NN, "NN_vcmp%s%s" % (_p, _k))] = ("cmp%s_%s" % (_p, _k), _v, (_v, _v))

XMM_FIRST = ida_idp.str2reg("xmm0")
YMM_FIRST = ida_idp.str2reg("ymm0")
YMM_TO_XMM = YMM_FIRST - XMM_FIRST
CALLS = _ids("call callfi callni")
ZERO_UPPER = _ids("vzeroupper vzeroall")
VEC_MOVES = _ids("vmovaps vmovups vmovapd vmovupd vmovdqa vmovdqu")
VEC_TYPES = {("ps", 16): "__m128", ("pd", 16): "__m128d", ("i", 16): "__m128i",
             ("ps", 32): "__m256", ("pd", 32): "__m256d", ("i", 32): "__m256i"}

HANDLED = set(SAME_FORM) | set(NDS_FORM) | set(SCALAR_MOVES) | set(INTRINSICS) | VEC_MOVES | \
    LANE_SHUFFLE | SHUFPS_SAME | HIGH_HALF | PERMILPD | set(ROUND_SCALAR) | _ids("vinsertps") | \
    _ids("movdqa movdqu vzeroupper vzeroall vmaskmovdqu")


def _same_reg(a: ida_ua.op_t, b: ida_ua.op_t) -> bool:
    return a.type == ida_ua.o_reg and b.type == ida_ua.o_reg and a.reg == b.reg


def _copy(op: ida_ua.op_t) -> ida_ua.op_t:
    c = ida_ua.op_t()
    c.assign(op)
    return c


def _is_ymm(op: ida_ua.op_t) -> bool:
    return op.type != ida_ua.o_void and op.dtype == ida_ua.dt_byte32


FLOAT_SUFFIX = ("ps", "ss", "pd", "sd")
INT_SHUFFLES = _ids("pshufd vpshufd")
INT_LOADS = _ids("vmovdqa vmovdqu movdqa movdqu")


def float_consumer(ea: int, reg: int, limit: int = 32, depth: int = 3) -> bool:
    """Is the next instruction that reads `reg` a floating-point one?

    Compilers freely use integer-domain shuffles/loads (vpshufd, vmovdqa) on
    float vectors; typing them by their consumer avoids __m128i casts."""
    insn = ida_ua.insn_t()
    end = ida_funcs.get_func(ea)
    end = end.end_ea if end else ea + 0x100
    for _ in range(limit):
        ea = ida_bytes.next_head(ea, end)
        if ea == ida_idaapi.BADADDR or not ida_ua.decode_insn(insn, ea):
            return False
        feature = insn.get_canon_feature()
        reads = writes = False
        for i in range(ida_ida.UA_MAXOP):
            op = insn.ops[i]
            if op.type == ida_ua.o_void:
                break
            if op.type == ida_ua.o_reg and op.reg in (reg, reg + YMM_TO_XMM):
                reads |= ida_idp.has_cf_use(feature, i)
                writes |= ida_idp.has_cf_chg(feature, i)
        if reads:
            if insn.itype in INT_SHUFFLES and depth and insn.ops[0].type == ida_ua.o_reg:
                return float_consumer(ea, insn.ops[0].reg, limit, depth - 1)   # look through
            mnem = ida_ua.print_insn_mnem(ea) or ""
            return mnem.endswith(FLOAT_SUFFIX)
        if writes or feature & ida_idp.CF_STOP or ida_idp.is_call_insn(insn) or \
                feature & ida_idp.CF_JUMP:
            return False
    return False


class YmmFlow:
    """Which ymm reads can see a value last written through xmmN.

    Per register, a definition is X (xmm-only write, call, function entry) or
    Y (256-bit write). A ymm read needs its low half refreshed from xmmN only
    if an X definition reaches it."""

    X, Y = 1, 2

    def __init__(self, pfn):
        self.sync = {}                            # (ea, n) -> "zext" | "low"
        chart = ida_gdl.FlowChart(pfn, flags=ida_gdl.FC_PREDS | ida_gdl.FC_NOEXT)
        blocks = list(chart)
        insns = {b.id: list(self._decode(b)) for b in blocks}
        everything_x = sum(self.X << (2 * n) for n in range(16))
        state = {b.id: 0 for b in blocks}
        if blocks:
            state[blocks[0].id] = everything_x
        work = [b.id for b in blocks]
        succs = {b.id: [s.id for s in b.succs()] for b in blocks}
        while work:
            bid = work.pop()
            out = self._transfer(insns[bid], state[bid], record=False)
            for s in succs[bid]:
                merged = state[s] | out
                if merged != state[s]:
                    state[s] = merged
                    work.append(s)
        for b in blocks:
            self._transfer(insns[b.id], state[b.id], record=True)

    @staticmethod
    def _decode(block):
        insn = ida_ua.insn_t()
        ea = block.start_ea
        while ea < block.end_ea:
            if ida_bytes.is_code(ida_bytes.get_flags(ea)) and ida_ua.decode_insn(insn, ea):
                feature = insn.get_canon_feature()
                reads, writes = [], []
                for i in range(ida_ida.UA_MAXOP):
                    op = insn.ops[i]
                    if op.type == ida_ua.o_void:
                        break
                    if op.type != ida_ua.o_reg:
                        continue
                    if XMM_FIRST <= op.reg < XMM_FIRST + 16:
                        n, wide = op.reg - XMM_FIRST, False
                    elif YMM_FIRST <= op.reg < YMM_FIRST + 16:
                        n, wide = op.reg - YMM_FIRST, True
                    else:
                        continue
                    if ida_idp.has_cf_use(feature, i):
                        reads.append((n, wide))
                    if ida_idp.has_cf_chg(feature, i):
                        writes.append((n, wide))
                yield insn.ea, insn.itype in CALLS or insn.itype in ZERO_UPPER, reads, writes
            ea = ida_bytes.next_head(ea, block.end_ea)

    def _transfer(self, insns, state, record):
        for ea, is_call, reads, writes in insns:
            if record:
                for n, wide in reads:
                    kinds = state >> (2 * n) & 3
                    if wide and kinds & self.X:
                        # VEX-128 writes zero bits 255:128; if 256-bit writes can
                        # also reach here, only the low half can be refreshed.
                        self.sync[(ea, n)] = "low" if kinds & self.Y else "zext"
            if is_call:
                state = sum(self.X << (2 * n) for n in range(16))
                continue
            for n, wide in writes:
                state = state & ~(3 << (2 * n)) | ((self.Y if wide else self.X) << (2 * n))
        return state

    def sync_mode(self, ea, n):
        return self.sync.get((ea, n))


class Emitter:
    """Microcode building blocks for one instruction."""

    def __init__(self, cdg, lifter, flow=None):
        self.cdg = cdg
        self.insn = cdg.insn
        self.lifter = lifter
        self.flow = flow

    @staticmethod
    def reg(mreg: int, size: int) -> hr.mop_t:
        mop = hr.mop_t()
        mop.make_reg(mreg, size)
        return mop

    def mov(self, src: hr.mop_t, dst: hr.mop_t, fp: bool = False):
        ins = self.cdg.emit(hr.m_mov, src, hr.mop_t(), dst)
        if fp and ins is not None:
            ins.iprops |= hr.IPROP_FPINSN

    def copy_upper(self, src: int, dst: int, low: int):
        """Copy bytes low..16 of one xmm register into another (4/8-byte pieces)."""
        for off, size in ((4, 4), (8, 8)) if low == 4 else ((8, 8),):
            self.mov(self.reg(src + off, size), self.reg(dst + off, size))

    def xmm_half(self, op: ida_ua.op_t) -> hr.mop_t:
        """Low 128 bits of a ymm register operand (always mirrored in xmmN)."""
        return self.reg(hr.reg2mreg(op.reg - YMM_TO_XMM), 16)

    def ymm_read(self, op: ida_ua.op_t, size: int) -> hr.mop_t:
        ymm = hr.reg2mreg(op.reg)
        mode = self.flow.sync_mode(self.insn.ea, op.reg - YMM_FIRST) if self.flow else "low"
        if mode == "zext":
            m128, m256 = self.lifter.tif("Xps", 32), self.lifter.tif("Vps", 32)
            self.mov(self.call("_mm256_zextps128_ps256", m256, [(self.xmm_half(op), m128)]), self.reg(ymm, 32))
        elif mode == "low":
            self.mov(self.xmm_half(op), self.reg(ymm, 16))
        return self.reg(ymm, size)

    def read(self, n: int, size: int) -> hr.mop_t:
        op = self.insn.ops[n]
        if op.type == ida_ua.o_reg:
            if _is_ymm(op):
                return self.ymm_read(op, size)
            return self.reg(hr.reg2mreg(op.reg), size)
        if op.type == ida_ua.o_imm:
            mop = hr.mop_t()
            mop.make_number(op.value, size)
            return mop
        return self.reg(self.cdg.load_operand(n), size)

    def write(self, n: int, value: hr.mop_t):
        op = self.insn.ops[n]
        if op.type != ida_ua.o_reg:
            self.cdg.store_operand(n, value)
            return
        if _is_ymm(op):
            # the decompiler keeps xmmN and ymmN apart: mirror the low half
            # (dead copies are optimised away)
            ymm = hr.reg2mreg(op.reg)
            self.mov(value, self.reg(ymm, 32))
            m128, m256 = self.lifter.tif("Xps", 32), self.lifter.tif("Vps", 32)
            self.mov(self.call("_mm256_castps256_ps128", m128, [(self.reg(ymm, 32), m256)]), self.xmm_half(op))
        else:
            self.mov(value, self.reg(hr.reg2mreg(op.reg), value.size))

    def call(self, name: str, result: ida_typeinf.tinfo_t, args, pure: bool = True) -> hr.mop_t:
        """Nested call to a helper, usable as a source operand."""
        info = hr.mcallinfo_t()
        info.callee = ida_idaapi.BADADDR
        info.solid_args = len(args)
        info.cc = ida_typeinf.CM_CC_FASTCALL
        info.return_type = result
        info.role = hr.ROLE_UNK
        info.flags = hr.FCI_FINAL | hr.FCI_PROP | (hr.FCI_SPLOK | hr.FCI_PURE if pure else 0)
        for mop, tif in args:
            arg = hr.mcallarg_t()
            arg.copy_mop(mop)
            arg.type = tif
            info.args.push_back(arg)
        insn = hr.minsn_t(self.insn.ea)
        insn.opcode = hr.m_call
        insn.l.make_helper(name)
        size = 0 if result.is_void() else result.get_size()
        insn.d._make_callinfo(info)
        insn.d.size = size
        mop = hr.mop_t()
        mop.make_insn(insn)
        mop.size = size
        return mop


class AvxLifter(hr.microcode_filter_t):
    def __init__(self):
        super().__init__()
        self.types = {}
        self.flows = {}

    def flow(self, cdg):
        pfn = ida_funcs.get_func(cdg.insn.ea)
        if pfn is None:
            return None
        key = (pfn.start_ea, pfn.end_ea, pfn.size())
        if key not in self.flows:
            if len(self.flows) > 64:
                self.flows.clear()
            self.flows[key] = YmmFlow(pfn)
        return self.flows[key]

    def tif(self, token: str, width: int) -> ida_typeinf.tinfo_t:
        key = (token, width)
        if key not in self.types:
            tif = ida_typeinf.tinfo_t()
            if token == "f":
                tif.create_simple_type(ida_typeinf.BTF_FLOAT)
            elif token == "d":
                tif.create_simple_type(ida_typeinf.BTF_DOUBLE)
            elif token in ("I", "n"):
                tif.create_simple_type(ida_typeinf.BTF_INT32)
            else:
                size = 16 if token[0] == "X" else width
                tif.get_named_type(None, VEC_TYPES[(token[1:], size)])
            self.types[key] = tif
        return self.types[key]

    @staticmethod
    def token_size(token: str, width: int) -> int:
        return {"f": 4, "d": 8, "I": 4, "n": 4}.get(token) or (16 if token[0] == "X" else width)

    def match(self, cdg):
        return cdg.insn.itype in HANDLED

    def apply(self, cdg):
        insn = cdg.insn
        saved = ida_ua.insn_t()
        saved.assign(insn)
        try:
            return self._lift(cdg, saved)
        except Exception as exc:          # never take the decompiler down
            ida_kernwin.msg("ps4ida AVX lifter: %#x: %s\n" % (insn.ea, exc))
            return hr.MERR_INSN
        finally:
            insn.assign(saved)

    # -- strategies --------------------------------------------------------

    def _lift(self, cdg, saved) -> int:
        insn = cdg.insn
        ops = [_copy(insn.ops[i]) for i in range(ida_ida.UA_MAXOP) if insn.ops[i].type != ida_ua.o_void]
        itype = insn.itype
        if itype in (NN.NN_vzeroupper, NN.NN_vzeroall):
            return hr.MERR_OK            # upper halves are not modelled (see YmmFlow)
        if itype == NN.NN_vmaskmovdqu and len(ops) == 2:
            return self._maskmov(cdg)
        ymm = any(_is_ymm(op) for op in ops)

        if not ymm:
            rc = self._lanes(cdg, itype, ops)
            if rc is not None:
                return rc
            rc = self._float_domain(cdg, itype, ops)
            if rc is not None:
                return rc
            if itype in ROUND_SCALAR:
                return self._round(cdg, itype, ops)
            rc = self._sse_twin(cdg, itype, ops)
            insn.assign(saved)
            if rc != hr.MERR_INSN:
                return rc
        elif itype in VEC_MOVES and len(ops) == 2:
            em = Emitter(cdg, self, self.flow(cdg))
            em.write(0, em.read(1, 32))
            return hr.MERR_OK
        elif itype in ZERO_IDIOM and len(ops) == 3 and _same_reg(ops[1], ops[2]):
            em = Emitter(cdg, self)
            kind = "pd" if itype == NN.NN_vxorpd else "ps"
            em.write(0, em.call("_mm256_setzero_" + kind, self.tif("V" + kind, 32), []))
            return hr.MERR_OK
        return self._intrinsic(cdg, itype, ops, 32 if ymm else 16)

    def _gen_as(self, cdg, itype, ops) -> int:
        """Rewrite the current instruction in place and run the stock lifter on it."""
        insn = cdg.insn
        insn.itype = itype
        for i in range(ida_ida.UA_MAXOP):
            if i < len(ops):
                insn.ops[i].assign(ops[i])
                insn.ops[i].n = i
            else:
                insn.ops[i].type = ida_ua.o_void
        cdg.prepare_gen_micro()
        return cdg.gen_micro()

    def _float_domain(self, cdg, itype, ops):
        """Integer-domain shuffle/load feeding float code -> float-typed twin."""
        if not ops or ops[0].type != ida_ua.o_reg:
            return None
        if itype in INT_SHUFFLES and len(ops) == 3 and ops[2].type == ida_ua.o_imm:
            if not float_consumer(cdg.insn.ea, ops[0].reg):
                return None
            d, s, imm = ops
            if not _same_reg(d, s):
                rc = self._gen_as(cdg, NN.NN_movaps, [d, s])
                if rc != hr.MERR_OK:
                    return rc
            return self._gen_as(cdg, NN.NN_shufps, [d, d, imm])
        if itype in INT_LOADS and len(ops) == 2 and float_consumer(cdg.insn.ea, ops[0].reg):
            if ops[1].type in (ida_ua.o_mem, ida_ua.o_displ, ida_ua.o_phrase):
                # typed load, so lanes read back as .m128_f32[i]
                em = Emitter(cdg, self)
                ptr = ida_typeinf.tinfo_t()
                ptr.create_ptr(self.tif("f", 16))
                name = "_mm_loadu_ps" if itype in _ids("vmovdqu movdqu") else "_mm_load_ps"
                addr = em.reg(cdg.load_effective_address(1), 8)
                em.write(0, em.call(name, self.tif("Xps", 16), [(addr, ptr)]))
                return hr.MERR_OK
            return self._gen_as(cdg, NN.NN_movaps, ops)
        return None

    def _sse_twin(self, cdg, itype, ops) -> int:
        if itype in SAME_FORM and (len(ops) == 2 or itype not in NDS_FORM):
            return self._gen_as(cdg, SAME_FORM[itype], ops)     # vmovlps m64, x / x, m64
        if itype in SCALAR_MOVES:
            if len(ops) == 2:                       # load / store form
                return self._gen_as(cdg, SCALAR_MOVES[itype], ops)
            sse = SCALAR_MOVES[itype]               # register merge form
        elif itype in NDS_FORM:
            sse = NDS_FORM[itype]
        else:
            return hr.MERR_INSN
        if len(ops) < 3:
            return hr.MERR_INSN

        d, s1, s2, rest = ops[0], ops[1], ops[2], ops[3:]
        if itype in ZERO_IDIOM and _same_reg(s1, s2):
            return self._gen_as(cdg, sse, [d, d])
        if _same_reg(d, s1):
            return self._gen_as(cdg, sse, [d, s2] + rest)
        if _same_reg(d, s2):
            if itype in COMMUTATIVE:
                return self._gen_as(cdg, sse, [d, s1] + rest)
            if s1.type == ida_ua.o_reg and itype in SCALAR_FP:
                code, size = SCALAR_FP[itype]
                dst = hr.reg2mreg(d.reg)
                ins = cdg.emit(code, size, hr.reg2mreg(s1.reg), dst, dst, 0)
                if ins is not None:
                    ins.iprops |= hr.IPROP_FPINSN
                return hr.MERR_OK
            if s1.type == ida_ua.o_reg and itype in SCALAR_CALL:
                name, size, argc = SCALAR_CALL[itype]
                em = Emitter(cdg, self)
                dst = hr.reg2mreg(d.reg)
                tif = self.tif("f" if size == 4 else "d", 16)
                args = [(em.reg(hr.reg2mreg(s1.reg), size), tif)] if argc == 2 else []
                args.append((em.reg(dst, size), tif))
                em.mov(em.call(name, tif, args), em.reg(dst, size))
                em.copy_upper(hr.reg2mreg(s1.reg), dst, size)
                return hr.MERR_OK
            if s1.type == ida_ua.o_reg and itype in SCALAR_MOVES:
                em = Emitter(cdg, self)          # low part already in place
                em.copy_upper(hr.reg2mreg(s1.reg), hr.reg2mreg(d.reg), 4 if itype == NN.NN_vmovss else 8)
                return hr.MERR_OK
            if s1.type == ida_ua.o_reg and itype in SCALAR_CVT:
                src_size, dst_size = SCALAR_CVT[itype]
                em = Emitter(cdg, self)
                dst = hr.reg2mreg(d.reg)
                low = em.reg(dst, src_size)          # convert first, then merge s1's upper part
                ins = cdg.emit(hr.m_f2f, low, hr.mop_t(), em.reg(dst, dst_size))
                if ins is not None:
                    ins.iprops |= hr.IPROP_FPINSN
                em.copy_upper(hr.reg2mreg(s1.reg), dst, dst_size)
                return hr.MERR_OK
            return hr.MERR_INSN
        rc = self._gen_as(cdg, NN.NN_movaps, [d, s1])
        if rc != hr.MERR_OK:
            return rc
        return self._gen_as(cdg, sse, [d, s2] + rest)

    def _lane_plan(self, itype, ops):
        """Lane-level rewrite: [(dst_off, src_operand or None for zero, src_off, size)]."""
        last = ops[-1]
        imm = last.value & 0xFF if last.type == ida_ua.o_imm else None
        extract = imm is not None and imm & 3 and not imm >> 2   # lane k -> 0, rest = lane 0
        if itype in LANE_SHUFFLE and len(ops) == 3 and extract:
            return [(0, 1, 4 * imm, 4), (4, 1, 0, 4), (8, 1, 0, 4), (12, 1, 0, 4)]
        if itype in SHUFPS_SAME and extract:
            srcs = ops[1:-1]
            if all(_same_reg(s, srcs[0]) for s in srcs) and (len(srcs) == 2 or _same_reg(ops[0], srcs[0])):
                return [(0, 1, 4 * imm, 4), (4, 1, 0, 4), (8, 1, 0, 4), (12, 1, 0, 4)]
        if itype in HIGH_HALF:
            srcs = ops[1:]
            if all(_same_reg(s, srcs[0]) for s in srcs) and (len(srcs) == 2 or _same_reg(ops[0], srcs[0])):
                return [(0, 1, 8, 8), (8, 1, 8, 8)]
            if itype == NN.NN_vmovhlps and len(ops) == 3 and _same_reg(ops[0], ops[2]):
                return [(0, 2, 8, 8), (8, 1, 8, 8)]
        if itype == NN.NN_vmovlhps and len(ops) == 3 and _same_reg(ops[0], ops[2]):
            return [(0, 1, 0, 8), (8, 2, 0, 8)]
        if itype in PERMILPD and len(ops) == 3 and imm is not None:
            return [(0, 1, 8 * (imm & 1), 8), (8, 1, 8 * (imm >> 1 & 1), 8)]
        if itype == NN.NN_vinsertps and len(ops) == 4 and imm is not None:
            src_off = 4 * (imm >> 6) if ops[2].type == ida_ua.o_reg else 0
            plan = [(4 * i, 1, 4 * i, 4) for i in range(4)]
            plan[imm >> 4 & 3] = (4 * (imm >> 4 & 3), 2, src_off, 4)
            return [(off, None, 0, 4) if imm >> (off // 4) & 1 else (off, src, soff, size)
                    for off, src, soff, size in plan]
        return None

    def _lanes(self, cdg, itype, ops):
        """Lane-level idioms -> per-lane moves (None: not an idiom)."""
        plan = self._lane_plan(itype, ops)
        if plan is None or ops[0].type != ida_ua.o_reg:
            return None
        em = Emitter(cdg, self)
        dst = hr.reg2mreg(ops[0].reg)
        # float lanes moved as floats, so the decompiler types them as such
        fp = itype in _ids("vinsertps vpermilpd vmovhlps vmovlhps vunpckhpd movhlps unpckhpd") or \
            float_consumer(cdg.insn.ea, ops[0].reg)
        bases, kregs = {}, []
        for n in {src for _, src, _, _ in plan if src is not None}:
            op = ops[n]
            if op.type == ida_ua.o_reg:
                base = hr.reg2mreg(op.reg)
                if base == dst:                    # snapshot: dst is written lane by lane
                    kreg = cdg.mba.alloc_kreg(16)
                    em.mov(em.reg(base, 16), em.reg(kreg, 16))
                    kregs.append(kreg)
                    base = kreg
            else:
                base = cdg.load_operand(n)
            bases[n] = base
        for doff, src, soff, size in plan:
            if src is None:
                zero = hr.mop_t()
                zero.make_number(0, size)
                em.mov(zero, em.reg(dst + doff, size))
            elif bases[src] + soff != dst + doff:
                em.mov(em.reg(bases[src] + soff, size), em.reg(dst + doff, size), fp)
        for kreg in kregs:
            cdg.mba.free_kreg(kreg, 16)
        return hr.MERR_OK

    def _maskmov(self, cdg) -> int:
        """vmaskmovdqu x, mask: byte-masked store to [rdi] -> _mm_maskmoveu_si128()."""
        em = Emitter(cdg, self)
        m128i = self.tif("Xi", 16)
        ptr = ida_typeinf.tinfo_t()
        ptr.create_ptr(ida_typeinf.tinfo_t(ida_typeinf.BTF_CHAR))
        rdi = em.reg(hr.reg2mreg(ida_idp.str2reg("rdi")), 8)
        void = ida_typeinf.tinfo_t(ida_typeinf.BT_VOID)
        call = em.call("_mm_maskmoveu_si128", void,
                       [(em.read(0, 16), m128i), (em.read(1, 16), m128i), (rdi, ptr)], pure=False)
        cdg.emit(hr.m_call, call.d.l, hr.mop_t(), call.d.d)
        return hr.MERR_OK

    def _round(self, cdg, itype, ops) -> int:
        size = ROUND_SCALAR[itype]
        vex = itype in (NN.NN_vroundss, NN.NN_vroundsd)
        if ops[-1].type != ida_ua.o_imm or ops[0].type != ida_ua.o_reg:
            return hr.MERR_INSN
        imm = ops[-1].value & 0xF
        name = "rint" if imm & 4 else ROUND_NAMES[imm & 3]
        tif = self.tif("f" if size == 4 else "d", 16)
        em = Emitter(cdg, self)
        dst = hr.reg2mreg(ops[0].reg)
        src_n = 2 if vex else 1
        value = em.read(src_n, size)
        result = em.call(name + ("f" if size == 4 else ""), tif, [(value, tif)])
        if vex and not _same_reg(ops[0], ops[1]):
            em.copy_upper(hr.reg2mreg(ops[1].reg), dst, size)
        em.mov(result, em.reg(dst, size))
        return hr.MERR_OK

    def _intrinsic(self, cdg, itype, ops, width) -> int:
        spec = INTRINSICS.get(itype)
        if spec is None:
            return hr.MERR_INSN
        base, result, args = spec
        if base[:4] in ("sll_", "srl_", "sra_") and len(ops) == 3 and ops[2].type == ida_ua.o_imm:
            base, args = base[:3] + "i_" + base[4:], ("Vi", "I")      # vpslld x, y, 4
        # vextractf128 / vcvtpd2ps write 128 bits from a 256-bit source
        prefix = "_mm256_" if width == 32 else "_mm_"
        if width == 16 and base in ("permute2f128_ps", "insertf128_ps", "extractf128_ps", "broadcast_ps"):
            return hr.MERR_INSN
        srcs = list(range(1, len(ops)))
        if len(srcs) != len(args):
            return hr.MERR_INSN
        em = Emitter(cdg, self, self.flow(cdg) if width == 32 else None)
        if itype == NN.NN_vinsertf128 and ops[3].type == ida_ua.o_imm and ops[3].value & 1 \
                and ops[1].type == ida_ua.o_reg:
            # only the low half of the source survives: build from two halves
            m128 = self.tif("Xps", 32)
            value = em.call("_mm256_set_m128", self.tif("Vps", 32),
                            [(em.read(2, 16), m128), (em.xmm_half(ops[1]), m128)])
            em.write(0, value)
            return hr.MERR_OK
        call_args = []
        for n, token in zip(srcs, args):
            tif = self.tif(token, width)
            call_args.append((em.read(n, self.token_size(token, width)), tif))
        value = em.call(prefix + base, self.tif(result, width), call_args)
        em.write(0, value)
        return hr.MERR_OK


class AvxLifterModule(ida_idaapi.plugmod_t):
    def __init__(self):
        super().__init__()
        self.filter = AvxLifter()
        self.enabled = hr.install_microcode_filter(self.filter, True)

    def run(self, arg):
        self.enabled = not self.enabled
        hr.install_microcode_filter(self.filter, self.enabled)
        ida_kernwin.msg("ps4ida AVX lifter %s (re-decompile to see the effect)\n"
                        % ("enabled" if self.enabled else "disabled"))
        return True

    def __del__(self):
        if self.enabled:
            hr.install_microcode_filter(self.filter, False)


class AvxLifterPlugin(ida_idaapi.plugin_t):
    flags = ida_idaapi.PLUGIN_MULTI
    comment = "Lift AVX instructions in the Hex-Rays decompiler"
    help = "Toggles the AVX microcode lifter for this database"
    wanted_name = "ps4ida AVX lifter"
    wanted_hotkey = ""

    def init(self):
        if ida_idp.ph_get_id() != ida_idp.PLFM_386 or not ida_ida.inf_is_64bit():
            return None
        if not hr.init_hexrays_plugin():
            return None
        return AvxLifterModule()


def PLUGIN_ENTRY():
    return AvxLifterPlugin()
