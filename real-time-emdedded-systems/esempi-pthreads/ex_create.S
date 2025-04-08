	.file	"ex_create.c"
.globl pippo
	.data
	.align 4
	.type	pippo, @object
	.size	pippo, 4
pippo:
	.long	5
	.section	.rodata
.LC0:
	.string	"mythread: parameter=%d\n"
	.align 4
.LC1:
	.string	"mythread: main_id==mythread_id:%d\n"
	.text
.globl body
	.type	body, @function
body:
	pushl	%ebp
	movl	%esp, %ebp
	subl	$24, %esp
	movl	8(%ebp), %eax
	movl	(%eax), %eax
	movl	%eax, -8(%ebp)
	movl	-8(%ebp), %eax
	movl	%eax, 4(%esp)
	movl	$.LC0, (%esp)
	call	printf
	call	pthread_self
	movl	%eax, -4(%ebp)
	movl	main_id, %edx
	movl	-4(%ebp), %eax
	movl	%eax, 4(%esp)
	movl	%edx, (%esp)
	call	pthread_equal
	movl	%eax, 4(%esp)
	movl	$.LC1, (%esp)
	call	printf
	movl	$5678, %eax
	leave
	ret
	.size	body, .-body
	.section	.rodata
.LC2:
	.string	"main: before pthread_create\n"
.LC3:
	.string	"main: after pthread_create\n"
.LC4:
	.string	"main: returnvalue is %d\n"
	.text
.globl main
	.type	main, @function
main:
	leal	4(%esp), %ecx
	andl	$-16, %esp
	pushl	-4(%ecx)
	pushl	%ebp
	movl	%esp, %ebp
	pushl	%ecx
	subl	$84, %esp
	movl	%gs:20, %eax
	movl	%eax, -8(%ebp)
	xorl	%eax, %eax
	movl	$1234, -56(%ebp)
	leal	-44(%ebp), %eax
	movl	%eax, (%esp)
	call	pthread_attr_init
	movl	$.LC2, (%esp)
	call	puts
	call	pthread_self
	movl	%eax, main_id
	leal	-56(%ebp), %eax
	movl	%eax, 12(%esp)
	movl	$body, 8(%esp)
	leal	-44(%ebp), %eax
	movl	%eax, 4(%esp)
	leal	-52(%ebp), %eax
	movl	%eax, (%esp)
	call	pthread_create
	movl	%eax, -48(%ebp)
	movl	$.LC3, (%esp)
	call	puts
	leal	-44(%ebp), %eax
	movl	%eax, (%esp)
	call	pthread_attr_destroy
	movl	-52(%ebp), %edx
	leal	-60(%ebp), %eax
	movl	%eax, 4(%esp)
	movl	%edx, (%esp)
	call	pthread_join
	movl	-60(%ebp), %eax
	movl	%eax, 4(%esp)
	movl	$.LC4, (%esp)
	call	printf
	movl	$0, %eax
	movl	-8(%ebp), %edx
	xorl	%gs:20, %edx
	je	.L5
	call	__stack_chk_fail
.L5:
	addl	$84, %esp
	popl	%ecx
	popl	%ebp
	leal	-4(%ecx), %esp
	ret
	.size	main, .-main
	.comm	main_id,4,4
	.ident	"GCC: (GNU) 4.1.2 20060928 (prerelease) (Ubuntu 4.1.1-13ubuntu5)"
	.section	.note.GNU-stack,"",@progbits
