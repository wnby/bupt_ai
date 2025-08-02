.text
.global thread_switch
.align 8
thread_switch:
    # 保存整数寄存器上下文
    pop     %rax 
    movq    %rax, (%rdi)         # 保存返回地址
    movq    %rsp, 0x8(%rdi)      # 保存栈指针 rsp
    movq    %rbp, 0x10(%rdi)     # 保存基址指针 rbp
    movq    %rbx, 0x18(%rdi)     # 保存 rbx
    movq    %r12, 0x20(%rdi)     # 保存 r12
    movq    %r13, 0x28(%rdi)     # 保存 r13
    movq    %r14, 0x30(%rdi)     # 保存 r14
    movq    %r15, 0x38(%rdi)     # 保存 r15

    # 保存 SSE 寄存器上下文
    movdqu  %xmm0,  0x40(%rdi)
    movdqu  %xmm1,  0x50(%rdi)
    movdqu  %xmm2,  0x60(%rdi)
    movdqu  %xmm3,  0x70(%rdi)
    movdqu  %xmm4,  0x80(%rdi)
    movdqu  %xmm5,  0x90(%rdi)
    movdqu  %xmm6,  0xA0(%rdi)
    movdqu  %xmm7,  0xB0(%rdi)
    movdqu  %xmm8,  0xC0(%rdi)
    movdqu  %xmm9,  0xD0(%rdi)
    movdqu  %xmm10, 0xE0(%rdi)
    movdqu  %xmm11, 0xF0(%rdi)
    movdqu  %xmm12, 0x100(%rdi)
    movdqu  %xmm13, 0x110(%rdi)
    movdqu  %xmm14, 0x120(%rdi)
    movdqu  %xmm15, 0x130(%rdi)

    # 恢复整数寄存器上下文
    movq    0x38(%rsi), %r15
    movq    0x30(%rsi), %r14
    movq    0x28(%rsi), %r13
    movq    0x20(%rsi), %r12
    movq    0x18(%rsi), %rbx
    movq    0x10(%rsi), %rbp
    movq    0x8(%rsi), %rsp
    movq    (%rsi), %rax         # 恢复返回地址
    push    %rax                 # 将返回地址压入栈

    # 恢复 SSE 寄存器上下文
    movdqu  0x130(%rsi), %xmm15
    movdqu  0x120(%rsi), %xmm14
    movdqu  0x110(%rsi), %xmm13
    movdqu  0x100(%rsi), %xmm12
    movdqu  0xF0(%rsi), %xmm11
    movdqu  0xE0(%rsi), %xmm10
    movdqu  0xD0(%rsi), %xmm9
    movdqu  0xC0(%rsi), %xmm8
    movdqu  0xB0(%rsi), %xmm7
    movdqu  0xA0(%rsi), %xmm6
    movdqu  0x90(%rsi), %xmm5
    movdqu  0x80(%rsi), %xmm4
    movdqu  0x70(%rsi), %xmm3
    movdqu  0x60(%rsi), %xmm2
    movdqu  0x50(%rsi), %xmm1
    movdqu  0x40(%rsi), %xmm0

    movq    %rsi, %r10           # 恢复其他参数
    movq    0x40(%r10), %rdi
    movq    0x48(%r10), %rsi
    movq    0x50(%r10), %rdx
    ret
