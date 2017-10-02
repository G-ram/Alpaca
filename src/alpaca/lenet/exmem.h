#ifndef EXMEM_H
#define EXMEM_H

static inline unsigned int read_addr(unsigned long address){
	if(address >= 0x10000) {}
	unsigned int result;
	unsigned int sr, flash;
	asm("mov r2,%0":"=r"(sr):);	// save SR before disabling IRQ
	asm("dint");
	asm("nop");
	asm("movx.a %1,%0":"=r"(flash):"m"(address));
	asm("movx.w @%1, %0":"=r"(result):"r"(flash));
	asm("mov %0,r2"::"r"(sr));	// restore previous SR and IRQ state
	asm("nop");
	asm("eint");
	return result;	// aligned address -> low-byte contains result
}

#endif