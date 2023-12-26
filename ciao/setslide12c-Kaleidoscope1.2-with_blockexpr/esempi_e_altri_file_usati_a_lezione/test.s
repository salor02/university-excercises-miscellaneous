	.text
	.file	"test.ll"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3, 0x0                          # -- Begin function a
.LCPI0_0:
	.quad	0x3ff0000000000000              # double 1
	.text
	.globl	a
	.p2align	4, 0x90
	.type	a,@function
a:                                      # @a
	.cfi_startproc
# %bb.0:                                # %entry
	movsd	%xmm0, -8(%rsp)
	addsd	.LCPI0_0(%rip), %xmm0
	retq
.Lfunc_end0:
	.size	a, .Lfunc_end0-a
	.cfi_endproc
                                        # -- End function
	.globl	b                               # -- Begin function b
	.p2align	4, 0x90
	.type	b,@function
b:                                      # @b
	.cfi_startproc
# %bb.0:                                # %entry
	subq	$24, %rsp
	.cfi_def_cfa_offset 32
	movsd	%xmm0, 8(%rsp)                  # 8-byte Spill
	movsd	%xmm0, 16(%rsp)
	callq	a@PLT
	addsd	8(%rsp), %xmm0                  # 8-byte Folded Reload
	addq	$24, %rsp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	b, .Lfunc_end1-b
	.cfi_endproc
                                        # -- End function
	.section	".note.GNU-stack","",@progbits
