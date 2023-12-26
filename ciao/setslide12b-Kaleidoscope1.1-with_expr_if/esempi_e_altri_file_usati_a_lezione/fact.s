	.text
	.file	"fact.ll"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function fact
.LCPI0_0:
	.quad	0x4000000000000000              # double 2
.LCPI0_1:
	.quad	0x4008000000000000              # double 3
.LCPI0_2:
	.quad	0x3ff0000000000000              # double 1
	.text
	.globl	fact
	.p2align	4, 0x90
	.type	fact,@function
fact:                                   # @fact
	.cfi_startproc
# %bb.0:                                # %entry
	movsd	%xmm0, -8(%rsp)
	ucomisd	.LCPI0_0(%rip), %xmm0
	jae	.LBB0_4
# %bb.1:                                # %trueexp
	movsd	-8(%rsp), %xmm0                 # xmm0 = mem[0],zero
	ucomisd	.LCPI0_2(%rip), %xmm0
	jae	.LBB0_3
# %bb.2:                                # %trueexp5
	movsd	-8(%rsp), %xmm0                 # xmm0 = mem[0],zero
	retq
.LBB0_4:                                # %falseexp9
	movsd	-8(%rsp), %xmm0                 # xmm0 = mem[0],zero
	mulsd	.LCPI0_1(%rip), %xmm0
	retq
.LBB0_3:                                # %falseexp
	movsd	-8(%rsp), %xmm0                 # xmm0 = mem[0],zero
	addsd	%xmm0, %xmm0
	retq
.Lfunc_end0:
	.size	fact, .Lfunc_end0-fact
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
