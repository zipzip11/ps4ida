"""
ps4ida_avx.py -- Hex-Rays microcode filter that lifts 128-bit AVX (VEX) code.

The x64 decompiler understands SSE but leaves VEX-encoded instructions
(vmovaps, vxorps, vmulps, ...) as __asm blocks. Most VEX-128 instructions are
the SSE instruction with a non-destructive third operand, so this filter
rewrites each one into its SSE twin and lets the decompiler's own SSE lifter
generate the microcode (types, _mm_* intrinsics, constant propagation):

    vmovaps x, m          ->  movaps x, m
    vxorps  x, y, y       ->  xorps  x, x                (zero idiom)
    vaddps  d, s1, s2     ->  movaps d, s1 ; addps d, s2
    vaddps  d, s1, d      ->  addps  d, s1               (commutative only)
    vsubss  d, s1, d      ->  d.lo = s1.lo - d.lo        (direct float microcode)

Not lifted (stay __asm): 256-bit ymm forms, non-commutative packed ops whose
destination is the second source, and 4-operand blends.

Approximations: VEX-128 zeroes bits 255:128 of the destination and scalar ops
copy the upper lanes from s1; neither is modelled when it can't be expressed
through the SSE twin. Both are invisible in practice for compiler-generated
scalar/128-bit code.

Install: copy to <IDAUSR>/plugins. Edit > Plugins > "ps4ida AVX lifter" toggles it
for the current database.
"""

import ida_allins
import ida_hexrays
import ida_ida
import ida_idaapi
import ida_idp
import ida_kernwin
import ida_ua


def _twins(names: str) -> dict:
    """Map 'vfoo' -> 'foo' instruction ids for every pair this IDA knows."""
    out = {}
    for name in names.split():
        vex, sse = getattr(ida_allins, "NN_" + name, None), getattr(ida_allins, "NN_" + name[1:], None)
        if vex is not None and sse is not None:
            out[vex] = sse
    return out


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
    vpabsb vpabsw vpabsd vphminposuw
""")

# VEX "d, s1, s2[, imm]" == "movaps d, s1 ; OP d, s2[, imm]".
NDS_FORM = _twins("""
    vaddps vsubps vmulps vdivps vminps vmaxps vandps vandnps vorps vxorps
    vaddpd vsubpd vmulpd vdivpd vminpd vmaxpd vandpd vandnpd vorpd vxorpd
    vaddss vsubss vmulss vdivss vminss vmaxss vsqrtss vrcpss vrsqrtss vroundss vcmpss
    vaddsd vsubsd vmulsd vdivsd vminsd vmaxsd vsqrtsd vroundsd vcmpsd
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
    vpinsrb vpinsrw vpinsrd vpinsrq
""")

# Packed ops where "d, s1, d" can be lifted as "OP d, s1".
COMMUTATIVE = set(_twins("""
    vaddps vmulps vandps vorps vxorps vaddpd vmulpd vandpd vorpd
    vpxor vpand vpor vpaddb vpaddw vpaddd vpaddq vpmulld vpmullw vpmuludq
    vpcmpeqb vpcmpeqw vpcmpeqd vpcmpeqq vpavgb vpavgw
"""))

# "OP d, s, s" whose result is 0 whatever s holds.
ZERO_IDIOM = set(_twins("vxorps vxorpd vpxor vpsubb vpsubw vpsubd vpsubq vandnps vandnpd vpandn"))

SCALAR_MOVES = {ida_allins.NN_vmovss: ida_allins.NN_movss, ida_allins.NN_vmovsd: ida_allins.NN_movsd}

# Scalar arithmetic is lifted straight to float microcode when the destination
# is also the second source (no SSE rewrite expresses that).
SCALAR_FP = {
    ida_allins.NN_vaddss: (ida_hexrays.m_fadd, 4), ida_allins.NN_vaddsd: (ida_hexrays.m_fadd, 8),
    ida_allins.NN_vsubss: (ida_hexrays.m_fsub, 4), ida_allins.NN_vsubsd: (ida_hexrays.m_fsub, 8),
    ida_allins.NN_vmulss: (ida_hexrays.m_fmul, 4), ida_allins.NN_vmulsd: (ida_hexrays.m_fmul, 8),
    ida_allins.NN_vdivss: (ida_hexrays.m_fdiv, 4), ida_allins.NN_vdivsd: (ida_hexrays.m_fdiv, 8),
}

HANDLED = set(SAME_FORM) | set(NDS_FORM) | set(SCALAR_MOVES)


def _same_reg(a: ida_ua.op_t, b: ida_ua.op_t) -> bool:
    return a.type == ida_ua.o_reg and b.type == ida_ua.o_reg and a.reg == b.reg


def _copy(op: ida_ua.op_t) -> ida_ua.op_t:
    c = ida_ua.op_t()
    c.assign(op)
    return c


class AvxLifter(ida_hexrays.microcode_filter_t):
    def match(self, cdg):
        insn = cdg.insn
        if insn.itype not in HANDLED:
            return False
        # SSE twins only exist for the 128-bit forms.
        return not any(op.type == ida_ua.o_reg and op.dtype == ida_ua.dt_byte32 for op in insn.ops)

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

    def apply(self, cdg):
        insn = cdg.insn
        saved = ida_ua.insn_t()
        saved.assign(insn)
        try:
            return self._lift(cdg)
        finally:
            insn.assign(saved)

    def _lift(self, cdg) -> int:
        insn = cdg.insn
        itype = insn.itype
        ops = []
        for i in range(ida_ida.UA_MAXOP):
            if insn.ops[i].type == ida_ua.o_void:
                break
            ops.append(_copy(insn.ops[i]))

        if itype in SAME_FORM:
            return self._gen_as(cdg, SAME_FORM[itype], ops)
        if itype in SCALAR_MOVES:
            if len(ops) == 2:                       # load / store form
                return self._gen_as(cdg, SCALAR_MOVES[itype], ops)
            sse = SCALAR_MOVES[itype]               # register merge form
        else:
            sse = NDS_FORM[itype]
        if len(ops) < 3:
            return ida_hexrays.MERR_INSN

        d, s1, s2, rest = ops[0], ops[1], ops[2], ops[3:]
        if itype in ZERO_IDIOM and _same_reg(s1, s2):
            return self._gen_as(cdg, sse, [d, d])
        if _same_reg(d, s1):
            return self._gen_as(cdg, sse, [d, s2] + rest)
        if _same_reg(d, s2):
            if itype in COMMUTATIVE:
                return self._gen_as(cdg, sse, [d, s1] + rest)
            if itype in SCALAR_FP and s1.type == ida_ua.o_reg:
                code, size = SCALAR_FP[itype]
                dst = ida_hexrays.reg2mreg(d.reg)
                ins = cdg.emit(code, size, ida_hexrays.reg2mreg(s1.reg), dst, dst, 0)
                if ins is not None:
                    ins.iprops |= ida_hexrays.IPROP_FPINSN
                return ida_hexrays.MERR_OK
            return ida_hexrays.MERR_INSN
        rc = self._gen_as(cdg, ida_allins.NN_movaps, [d, s1])
        if rc != ida_hexrays.MERR_OK:
            return rc
        return self._gen_as(cdg, sse, [d, s2] + rest)


class AvxLifterModule(ida_idaapi.plugmod_t):
    def __init__(self):
        super().__init__()
        self.filter = AvxLifter()
        self.enabled = ida_hexrays.install_microcode_filter(self.filter, True)

    def run(self, arg):
        self.enabled = not self.enabled
        ida_hexrays.install_microcode_filter(self.filter, self.enabled)
        ida_kernwin.msg("ps4ida AVX lifter %s (re-decompile to see the effect)\n"
                        % ("enabled" if self.enabled else "disabled"))
        return True

    def __del__(self):
        if self.enabled:
            ida_hexrays.install_microcode_filter(self.filter, False)


class AvxLifterPlugin(ida_idaapi.plugin_t):
    flags = ida_idaapi.PLUGIN_MULTI
    comment = "Lift 128-bit AVX instructions in the Hex-Rays decompiler"
    help = "Toggles the AVX microcode lifter for this database"
    wanted_name = "ps4ida AVX lifter"
    wanted_hotkey = ""

    def init(self):
        if ida_idp.ph_get_id() != ida_idp.PLFM_386 or not ida_ida.inf_is_64bit():
            return None
        if not ida_hexrays.init_hexrays_plugin():
            return None
        return AvxLifterModule()


def PLUGIN_ENTRY():
    return AvxLifterPlugin()
