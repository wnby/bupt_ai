#include "mmu.h"

  # vectors.S sends all traps here.
.globl alltraps
alltraps:
  # Build trap frame.
  pushl %ds
  pushl %es
  pushl %fs
  pushl %gs
  pushal

  # Set up data segments.
  movw $(SEG_KDATA<<3), %ax
  movw %ax, %ds
  movw %ax, %es

  # Call trap(tf), where tf=%esp
  # 这里压入栈顶指针, 栈顶恰好形成了struct trapframe
  # 可以思考一下CPU硬件、vector0、alltraps先后向内核栈压入值的顺序, 然后比较struct trapframe结构体的内部结构
  pushl %esp
  call trap
  addl $4, %esp
  # 此处函数没有ret
  # Return falls through to trapret...
.globl trapret
trapret:
  popal
  popl %gs
  popl %fs
  popl %es
  popl %ds
  addl $0x8, %esp  # trapno and errcode
  iret
