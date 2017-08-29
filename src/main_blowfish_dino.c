#include <msp430.h>

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <libmspbuiltins/builtins.h>
#include <libio/log.h>
#include <libmsp/mem.h>
#include <libmsp/periph.h>
#include <libmsp/clock.h>
#include <libmsp/watchdog.h>
#include <libmsp/gpio.h>
#include <libmspmath/msp-math.h>

#ifdef CONFIG_EDB
#include <libedb/edb.h>
#else
#define ENERGY_GUARD_BEGIN()
#define ENERGY_GUARD_END()
#endif

#ifdef DINO
#include <libdino/dino.h>
#endif

#include "pins.h"
#define LENGTH 13

bool test11 = 0;
unsigned overflow=0;
//__attribute__((interrupt(TIMERB1_VECTOR))) 
__attribute__((interrupt(51))) 
void TimerB1_ISR(void){
	TBCTL &= ~(0x0002);
	if(TBCTL && 0x0001){
		overflow++;
		TBCTL |= 0x0004;
		TBCTL |= (0x0002);
		TBCTL &= ~(0x0001);	
	}
}
__attribute__((section("__interrupt_vector_timer0_b1"),aligned(2)))
void(*__vector_timer0_b1)(void) = TimerB1_ISR;
static __nv unsigned curtask;

/* This is for progress reporting only */
#define SET_CURTASK(t) curtask = t

#define TASK_INIT                   1
#define TASK_SET_UKEY               2
#define TASK_INIT_KEY               3
#define TASK_INIT_S                 4
#define TASK_SET_KEY                5
#define TASK_ENCRYPT                6
#define TASK_ENCRYPT_END            7
#define TASK_START_ENCRYPT          8
#define TASK_START_ENCRYPT2     9

#ifdef DINO

#define TASK_BOUNDARY(t) \
        DINO_TASK_BOUNDARY(NULL); \
        SET_CURTASK(t); \

#define DINO_MANUAL_RESTORE_NONE() \
        DINO_MANUAL_REVERT_BEGIN() \
        DINO_MANUAL_REVERT_END() \

#define DINO_MANUAL_RESTORE_PTR(nm, type) \
        DINO_MANUAL_REVERT_BEGIN() \
        DINO_MANUAL_REVERT_PTR(type, nm); \
        DINO_MANUAL_REVERT_END() \

#define DINO_MANUAL_RESTORE_VAL(nm, label) \
        DINO_MANUAL_REVERT_BEGIN() \
        DINO_MANUAL_REVERT_VAL(nm, label); \
        DINO_MANUAL_REVERT_END() \

#else // !DINO

#define TASK_BOUNDARY(t) SET_CURTASK(t)

#define DINO_RESTORE_CHECK()
#define DINO_MANUAL_VERSION_PTR(...)
#define DINO_MANUAL_VERSION_VAL(...)
#define DINO_MANUAL_RESTORE_NONE()
#define DINO_MANUAL_RESTORE_PTR(...)
#define DINO_MANUAL_RESTORE_VAL(...)
#define DINO_MANUAL_REVERT_BEGIN(...)
#define DINO_MANUAL_REVERT_END(...)
#define DINO_MANUAL_REVERT_VAL(...)

#endif // !DINO

static void init_hw()
{
	msp_watchdog_disable();
	msp_gpio_unlock();
	msp_clock_setup();
}
static __ro_nv const char cp[32] = {'1','2','3','4','5','6','7','8','9','0',
	'A','B','C','D','E','F','F','E','D','C','B','A',
	'0','9','8','7','6','5','4','3','2','1'}; //mimicing 16byte hex key (0x1234_5678_90ab_cdef_fedc_ba09_8765_4321)
static __ro_nv const char indata[LENGTH] = {'H','e','l','l','o',',',' ','w','o','r','l','d','!'};
static __ro_nv const uint32_t init_key[18] = {
	0x243f6a88L, 0x85a308d3L, 0x13198a2eL, 0x03707344L,
	0xa4093822L, 0x299f31d0L, 0x082efa98L, 0xec4e6c89L,
	0x452821e6L, 0x38d01377L, 0xbe5466cfL, 0x34e90c6cL,
	0xc0ac29b7L, 0xc97c50ddL, 0x3f84d5b5L, 0xb5470917L,
	0x9216d5d9L, 0x8979fb1b
};
uint32_t __nv s0[256], s1[256], s2[256], s3[256];
static __ro_nv const uint32_t init_s0[256] = {
	0xd1310ba6L, 0x98dfb5acL, 0x2ffd72dbL, 0xd01adfb7L, 
	0xb8e1afedL, 0x6a267e96L, 0xba7c9045L, 0xf12c7f99L, 
	0x24a19947L, 0xb3916cf7L, 0x0801f2e2L, 0x858efc16L, 
	0x636920d8L, 0x71574e69L, 0xa458fea3L, 0xf4933d7eL, 
	0x0d95748fL, 0x728eb658L, 0x718bcd58L, 0x82154aeeL, 
	0x7b54a41dL, 0xc25a59b5L, 0x9c30d539L, 0x2af26013L, 
	0xc5d1b023L, 0x286085f0L, 0xca417918L, 0xb8db38efL, 
	0x8e79dcb0L, 0x603a180eL, 0x6c9e0e8bL, 0xb01e8a3eL, 
	0xd71577c1L, 0xbd314b27L, 0x78af2fdaL, 0x55605c60L, 
	0xe65525f3L, 0xaa55ab94L, 0x57489862L, 0x63e81440L, 
	0x55ca396aL, 0x2aab10b6L, 0xb4cc5c34L, 0x1141e8ceL, 
	0xa15486afL, 0x7c72e993L, 0xb3ee1411L, 0x636fbc2aL, 
	0x2ba9c55dL, 0x741831f6L, 0xce5c3e16L, 0x9b87931eL, 
	0xafd6ba33L, 0x6c24cf5cL, 0x7a325381L, 0x28958677L, 
	0x3b8f4898L, 0x6b4bb9afL, 0xc4bfe81bL, 0x66282193L, 
	0x61d809ccL, 0xfb21a991L, 0x487cac60L, 0x5dec8032L, 
	0xef845d5dL, 0xe98575b1L, 0xdc262302L, 0xeb651b88L, 
	0x23893e81L, 0xd396acc5L, 0x0f6d6ff3L, 0x83f44239L, 
	0x2e0b4482L, 0xa4842004L, 0x69c8f04aL, 0x9e1f9b5eL, 
	0x21c66842L, 0xf6e96c9aL, 0x670c9c61L, 0xabd388f0L, 
	0x6a51a0d2L, 0xd8542f68L, 0x960fa728L, 0xab5133a3L, 
	0x6eef0b6cL, 0x137a3be4L, 0xba3bf050L, 0x7efb2a98L, 
	0xa1f1651dL, 0x39af0176L, 0x66ca593eL, 0x82430e88L, 
	0x8cee8619L, 0x456f9fb4L, 0x7d84a5c3L, 0x3b8b5ebeL, 
	0xe06f75d8L, 0x85c12073L, 0x401a449fL, 0x56c16aa6L, 
	0x4ed3aa62L, 0x363f7706L, 0x1bfedf72L, 0x429b023dL, 
	0x37d0d724L, 0xd00a1248L, 0xdb0fead3L, 0x49f1c09bL, 
	0x075372c9L, 0x80991b7bL, 0x25d479d8L, 0xf6e8def7L, 
	0xe3fe501aL, 0xb6794c3bL, 0x976ce0bdL, 0x04c006baL, 
	0xc1a94fb6L, 0x409f60c4L, 0x5e5c9ec2L, 0x196a2463L, 
	0x68fb6fafL, 0x3e6c53b5L, 0x1339b2ebL, 0x3b52ec6fL, 
	0x6dfc511fL, 0x9b30952cL, 0xcc814544L, 0xaf5ebd09L, 
	0xbee3d004L, 0xde334afdL, 0x660f2807L, 0x192e4bb3L, 
	0xc0cba857L, 0x45c8740fL, 0xd20b5f39L, 0xb9d3fbdbL, 
	0x5579c0bdL, 0x1a60320aL, 0xd6a100c6L, 0x402c7279L, 
	0x679f25feL, 0xfb1fa3ccL, 0x8ea5e9f8L, 0xdb3222f8L, 
	0x3c7516dfL, 0xfd616b15L, 0x2f501ec8L, 0xad0552abL, 
	0x323db5faL, 0xfd238760L, 0x53317b48L, 0x3e00df82L, 
	0x9e5c57bbL, 0xca6f8ca0L, 0x1a87562eL, 0xdf1769dbL, 
	0xd542a8f6L, 0x287effc3L, 0xac6732c6L, 0x8c4f5573L, 
	0x695b27b0L, 0xbbca58c8L, 0xe1ffa35dL, 0xb8f011a0L, 
	0x10fa3d98L, 0xfd2183b8L, 0x4afcb56cL, 0x2dd1d35bL, 
	0x9a53e479L, 0xb6f84565L, 0xd28e49bcL, 0x4bfb9790L, 
	0xe1ddf2daL, 0xa4cb7e33L, 0x62fb1341L, 0xcee4c6e8L, 
	0xef20cadaL, 0x36774c01L, 0xd07e9efeL, 0x2bf11fb4L, 
	0x95dbda4dL, 0xae909198L, 0xeaad8e71L, 0x6b93d5a0L, 
	0xd08ed1d0L, 0xafc725e0L, 0x8e3c5b2fL, 0x8e7594b7L, 
	0x8ff6e2fbL, 0xf2122b64L, 0x8888b812L, 0x900df01cL, 
	0x4fad5ea0L, 0x688fc31cL, 0xd1cff191L, 0xb3a8c1adL, 
	0x2f2f2218L, 0xbe0e1777L, 0xea752dfeL, 0x8b021fa1L, 
	0xe5a0cc0fL, 0xb56f74e8L, 0x18acf3d6L, 0xce89e299L, 
	0xb4a84fe0L, 0xfd13e0b7L, 0x7cc43b81L, 0xd2ada8d9L, 
	0x165fa266L, 0x80957705L, 0x93cc7314L, 0x211a1477L, 
	0xe6ad2065L, 0x77b5fa86L, 0xc75442f5L, 0xfb9d35cfL, 
	0xebcdaf0cL, 0x7b3e89a0L, 0xd6411bd3L, 0xae1e7e49L, 
	0x00250e2dL, 0x2071b35eL, 0x226800bbL, 0x57b8e0afL, 
	0x2464369bL, 0xf009b91eL, 0x5563911dL, 0x59dfa6aaL, 
	0x78c14389L, 0xd95a537fL, 0x207d5ba2L, 0x02e5b9c5L, 
	0x83260376L, 0x6295cfa9L, 0x11c81968L, 0x4e734a41L, 
	0xb3472dcaL, 0x7b14a94aL, 0x1b510052L, 0x9a532915L, 
	0xd60f573fL, 0xbc9bc6e4L, 0x2b60a476L, 0x81e67400L, 
	0x08ba6fb5L, 0x571be91fL, 0xf296ec6bL, 0x2a0dd915L, 
	0xb6636521L, 0xe7b9f9b6L, 0xff34052eL, 0xc5855664L, 
	0x53b02d5dL, 0xa99f8fa1L, 0x08ba4799L, 0x6e85076aL, 
};

static __ro_nv const uint32_t init_s1[256] = {
	0x4b7a70e9L, 0xb5b32944L, 0xdb75092eL, 0xc4192623L, 
	0xad6ea6b0L, 0x49a7df7dL, 0x9cee60b8L, 0x8fedb266L, 
	0xecaa8c71L, 0x699a17ffL, 0x5664526cL, 0xc2b19ee1L, 
	0x193602a5L, 0x75094c29L, 0xa0591340L, 0xe4183a3eL, 
	0x3f54989aL, 0x5b429d65L, 0x6b8fe4d6L, 0x99f73fd6L, 
	0xa1d29c07L, 0xefe830f5L, 0x4d2d38e6L, 0xf0255dc1L, 
	0x4cdd2086L, 0x8470eb26L, 0x6382e9c6L, 0x021ecc5eL, 
	0x09686b3fL, 0x3ebaefc9L, 0x3c971814L, 0x6b6a70a1L, 
	0x687f3584L, 0x52a0e286L, 0xb79c5305L, 0xaa500737L, 
	0x3e07841cL, 0x7fdeae5cL, 0x8e7d44ecL, 0x5716f2b8L, 
	0xb03ada37L, 0xf0500c0dL, 0xf01c1f04L, 0x0200b3ffL, 
	0xae0cf51aL, 0x3cb574b2L, 0x25837a58L, 0xdc0921bdL, 
	0xd19113f9L, 0x7ca92ff6L, 0x94324773L, 0x22f54701L, 
	0x3ae5e581L, 0x37c2dadcL, 0xc8b57634L, 0x9af3dda7L, 
	0xa9446146L, 0x0fd0030eL, 0xecc8c73eL, 0xa4751e41L, 
	0xe238cd99L, 0x3bea0e2fL, 0x3280bba1L, 0x183eb331L, 
	0x4e548b38L, 0x4f6db908L, 0x6f420d03L, 0xf60a04bfL, 
	0x2cb81290L, 0x24977c79L, 0x5679b072L, 0xbcaf89afL, 
	0xde9a771fL, 0xd9930810L, 0xb38bae12L, 0xdccf3f2eL, 
	0x5512721fL, 0x2e6b7124L, 0x501adde6L, 0x9f84cd87L, 
	0x7a584718L, 0x7408da17L, 0xbc9f9abcL, 0xe94b7d8cL, 
	0xec7aec3aL, 0xdb851dfaL, 0x63094366L, 0xc464c3d2L, 
	0xef1c1847L, 0x3215d908L, 0xdd433b37L, 0x24c2ba16L, 
	0x12a14d43L, 0x2a65c451L, 0x50940002L, 0x133ae4ddL, 
	0x71dff89eL, 0x10314e55L, 0x81ac77d6L, 0x5f11199bL, 
	0x043556f1L, 0xd7a3c76bL, 0x3c11183bL, 0x5924a509L, 
	0xf28fe6edL, 0x97f1fbfaL, 0x9ebabf2cL, 0x1e153c6eL, 
	0x86e34570L, 0xeae96fb1L, 0x860e5e0aL, 0x5a3e2ab3L, 
	0x771fe71cL, 0x4e3d06faL, 0x2965dcb9L, 0x99e71d0fL, 
	0x803e89d6L, 0x5266c825L, 0x2e4cc978L, 0x9c10b36aL, 
	0xc6150ebaL, 0x94e2ea78L, 0xa5fc3c53L, 0x1e0a2df4L, 
	0xf2f74ea7L, 0x361d2b3dL, 0x1939260fL, 0x19c27960L, 
	0x5223a708L, 0xf71312b6L, 0xebadfe6eL, 0xeac31f66L, 
	0xe3bc4595L, 0xa67bc883L, 0xb17f37d1L, 0x018cff28L, 
	0xc332ddefL, 0xbe6c5aa5L, 0x65582185L, 0x68ab9802L, 
	0xeecea50fL, 0xdb2f953bL, 0x2aef7dadL, 0x5b6e2f84L, 
	0x1521b628L, 0x29076170L, 0xecdd4775L, 0x619f1510L, 
	0x13cca830L, 0xeb61bd96L, 0x0334fe1eL, 0xaa0363cfL, 
	0xb5735c90L, 0x4c70a239L, 0xd59e9e0bL, 0xcbaade14L, 
	0xeecc86bcL, 0x60622ca7L, 0x9cab5cabL, 0xb2f3846eL, 
	0x648b1eafL, 0x19bdf0caL, 0xa02369b9L, 0x655abb50L, 
	0x40685a32L, 0x3c2ab4b3L, 0x319ee9d5L, 0xc021b8f7L, 
	0x9b540b19L, 0x875fa099L, 0x95f7997eL, 0x623d7da8L, 
	0xf837889aL, 0x97e32d77L, 0x11ed935fL, 0x16681281L, 
	0x0e358829L, 0xc7e61fd6L, 0x96dedfa1L, 0x7858ba99L, 
	0x57f584a5L, 0x1b227263L, 0x9b83c3ffL, 0x1ac24696L, 
	0xcdb30aebL, 0x532e3054L, 0x8fd948e4L, 0x6dbc3128L, 
	0x58ebf2efL, 0x34c6ffeaL, 0xfe28ed61L, 0xee7c3c73L, 
	0x5d4a14d9L, 0xe864b7e3L, 0x42105d14L, 0x203e13e0L, 
	0x45eee2b6L, 0xa3aaabeaL, 0xdb6c4f15L, 0xfacb4fd0L, 
	0xc742f442L, 0xef6abbb5L, 0x654f3b1dL, 0x41cd2105L, 
	0xd81e799eL, 0x86854dc7L, 0xe44b476aL, 0x3d816250L, 
	0xcf62a1f2L, 0x5b8d2646L, 0xfc8883a0L, 0xc1c7b6a3L, 
	0x7f1524c3L, 0x69cb7492L, 0x47848a0bL, 0x5692b285L, 
	0x095bbf00L, 0xad19489dL, 0x1462b174L, 0x23820e00L, 
	0x58428d2aL, 0x0c55f5eaL, 0x1dadf43eL, 0x233f7061L, 
	0x3372f092L, 0x8d937e41L, 0xd65fecf1L, 0x6c223bdbL, 
	0x7cde3759L, 0xcbee7460L, 0x4085f2a7L, 0xce77326eL, 
	0xa6078084L, 0x19f8509eL, 0xe8efd855L, 0x61d99735L, 
	0xa969a7aaL, 0xc50c06c2L, 0x5a04abfcL, 0x800bcadcL, 
	0x9e447a2eL, 0xc3453484L, 0xfdd56705L, 0x0e1e9ec9L, 
	0xdb73dbd3L, 0x105588cdL, 0x675fda79L, 0xe3674340L, 
	0xc5c43465L, 0x713e38d8L, 0x3d28f89eL, 0xf16dff20L, 
	0x153e21e7L, 0x8fb03d4aL, 0xe6e39f2bL, 0xdb83adf7L, 
};

static __ro_nv const uint32_t init_s2[256] = {
	0xe93d5a68L, 0x948140f7L, 0xf64c261cL, 0x94692934L, 
	0x411520f7L, 0x7602d4f7L, 0xbcf46b2eL, 0xd4a20068L, 
	0xd4082471L, 0x3320f46aL, 0x43b7d4b7L, 0x500061afL, 
	0x1e39f62eL, 0x97244546L, 0x14214f74L, 0xbf8b8840L, 
	0x4d95fc1dL, 0x96b591afL, 0x70f4ddd3L, 0x66a02f45L, 
	0xbfbc09ecL, 0x03bd9785L, 0x7fac6dd0L, 0x31cb8504L, 
	0x96eb27b3L, 0x55fd3941L, 0xda2547e6L, 0xabca0a9aL, 
	0x28507825L, 0x530429f4L, 0x0a2c86daL, 0xe9b66dfbL, 
	0x68dc1462L, 0xd7486900L, 0x680ec0a4L, 0x27a18deeL, 
	0x4f3ffea2L, 0xe887ad8cL, 0xb58ce006L, 0x7af4d6b6L, 
	0xaace1e7cL, 0xd3375fecL, 0xce78a399L, 0x406b2a42L, 
	0x20fe9e35L, 0xd9f385b9L, 0xee39d7abL, 0x3b124e8bL, 
	0x1dc9faf7L, 0x4b6d1856L, 0x26a36631L, 0xeae397b2L, 
	0x3a6efa74L, 0xdd5b4332L, 0x6841e7f7L, 0xca7820fbL, 
	0xfb0af54eL, 0xd8feb397L, 0x454056acL, 0xba489527L, 
	0x55533a3aL, 0x20838d87L, 0xfe6ba9b7L, 0xd096954bL, 
	0x55a867bcL, 0xa1159a58L, 0xcca92963L, 0x99e1db33L, 
	0xa62a4a56L, 0x3f3125f9L, 0x5ef47e1cL, 0x9029317cL, 
	0xfdf8e802L, 0x04272f70L, 0x80bb155cL, 0x05282ce3L, 
	0x95c11548L, 0xe4c66d22L, 0x48c1133fL, 0xc70f86dcL, 
	0x07f9c9eeL, 0x41041f0fL, 0x404779a4L, 0x5d886e17L, 
	0x325f51ebL, 0xd59bc0d1L, 0xf2bcc18fL, 0x41113564L, 
	0x257b7834L, 0x602a9c60L, 0xdff8e8a3L, 0x1f636c1bL, 
	0x0e12b4c2L, 0x02e1329eL, 0xaf664fd1L, 0xcad18115L, 
	0x6b2395e0L, 0x333e92e1L, 0x3b240b62L, 0xeebeb922L, 
	0x85b2a20eL, 0xe6ba0d99L, 0xde720c8cL, 0x2da2f728L, 
	0xd0127845L, 0x95b794fdL, 0x647d0862L, 0xe7ccf5f0L, 
	0x5449a36fL, 0x877d48faL, 0xc39dfd27L, 0xf33e8d1eL, 
	0x0a476341L, 0x992eff74L, 0x3a6f6eabL, 0xf4f8fd37L, 
	0xa812dc60L, 0xa1ebddf8L, 0x991be14cL, 0xdb6e6b0dL, 
	0xc67b5510L, 0x6d672c37L, 0x2765d43bL, 0xdcd0e804L, 
	0xf1290dc7L, 0xcc00ffa3L, 0xb5390f92L, 0x690fed0bL, 
	0x667b9ffbL, 0xcedb7d9cL, 0xa091cf0bL, 0xd9155ea3L, 
	0xbb132f88L, 0x515bad24L, 0x7b9479bfL, 0x763bd6ebL, 
	0x37392eb3L, 0xcc115979L, 0x8026e297L, 0xf42e312dL, 
	0x6842ada7L, 0xc66a2b3bL, 0x12754cccL, 0x782ef11cL, 
	0x6a124237L, 0xb79251e7L, 0x06a1bbe6L, 0x4bfb6350L, 
	0x1a6b1018L, 0x11caedfaL, 0x3d25bdd8L, 0xe2e1c3c9L, 
	0x44421659L, 0x0a121386L, 0xd90cec6eL, 0xd5abea2aL, 
	0x64af674eL, 0xda86a85fL, 0xbebfe988L, 0x64e4c3feL, 
	0x9dbc8057L, 0xf0f7c086L, 0x60787bf8L, 0x6003604dL, 
	0xd1fd8346L, 0xf6381fb0L, 0x7745ae04L, 0xd736fcccL, 
	0x83426b33L, 0xf01eab71L, 0xb0804187L, 0x3c005e5fL, 
	0x77a057beL, 0xbde8ae24L, 0x55464299L, 0xbf582e61L, 
	0x4e58f48fL, 0xf2ddfda2L, 0xf474ef38L, 0x8789bdc2L, 
	0x5366f9c3L, 0xc8b38e74L, 0xb475f255L, 0x46fcd9b9L, 
	0x7aeb2661L, 0x8b1ddf84L, 0x846a0e79L, 0x915f95e2L, 
	0x466e598eL, 0x20b45770L, 0x8cd55591L, 0xc902de4cL, 
	0xb90bace1L, 0xbb8205d0L, 0x11a86248L, 0x7574a99eL, 
	0xb77f19b6L, 0xe0a9dc09L, 0x662d09a1L, 0xc4324633L, 
	0xe85a1f02L, 0x09f0be8cL, 0x4a99a025L, 0x1d6efe10L, 
	0x1ab93d1dL, 0x0ba5a4dfL, 0xa186f20fL, 0x2868f169L, 
	0xdcb7da83L, 0x573906feL, 0xa1e2ce9bL, 0x4fcd7f52L, 
	0x50115e01L, 0xa70683faL, 0xa002b5c4L, 0x0de6d027L, 
	0x9af88c27L, 0x773f8641L, 0xc3604c06L, 0x61a806b5L, 
	0xf0177a28L, 0xc0f586e0L, 0x006058aaL, 0x30dc7d62L, 
	0x11e69ed7L, 0x2338ea63L, 0x53c2dd94L, 0xc2c21634L, 
	0xbbcbee56L, 0x90bcb6deL, 0xebfc7da1L, 0xce591d76L, 
	0x6f05e409L, 0x4b7c0188L, 0x39720a3dL, 0x7c927c24L, 
	0x86e3725fL, 0x724d9db9L, 0x1ac15bb4L, 0xd39eb8fcL, 
	0xed545578L, 0x08fca5b5L, 0xd83d7cd3L, 0x4dad0fc4L, 
	0x1e50ef5eL, 0xb161e6f8L, 0xa28514d9L, 0x6c51133cL, 
	0x6fd5c7e7L, 0x56e14ec4L, 0x362abfceL, 0xddc6c837L, 
	0xd79a3234L, 0x92638212L, 0x670efa8eL, 0x406000e0L, 
};

static __ro_nv const uint32_t init_s3[256] = {
	0x3a39ce37L, 0xd3faf5cfL, 0xabc27737L, 0x5ac52d1bL, 
	0x5cb0679eL, 0x4fa33742L, 0xd3822740L, 0x99bc9bbeL, 
	0xd5118e9dL, 0xbf0f7315L, 0xd62d1c7eL, 0xc700c47bL, 
	0xb78c1b6bL, 0x21a19045L, 0xb26eb1beL, 0x6a366eb4L, 
	0x5748ab2fL, 0xbc946e79L, 0xc6a376d2L, 0x6549c2c8L, 
	0x530ff8eeL, 0x468dde7dL, 0xd5730a1dL, 0x4cd04dc6L, 
	0x2939bbdbL, 0xa9ba4650L, 0xac9526e8L, 0xbe5ee304L, 
	0xa1fad5f0L, 0x6a2d519aL, 0x63ef8ce2L, 0x9a86ee22L, 
	0xc089c2b8L, 0x43242ef6L, 0xa51e03aaL, 0x9cf2d0a4L, 
	0x83c061baL, 0x9be96a4dL, 0x8fe51550L, 0xba645bd6L, 
	0x2826a2f9L, 0xa73a3ae1L, 0x4ba99586L, 0xef5562e9L, 
	0xc72fefd3L, 0xf752f7daL, 0x3f046f69L, 0x77fa0a59L, 
	0x80e4a915L, 0x87b08601L, 0x9b09e6adL, 0x3b3ee593L, 
	0xe990fd5aL, 0x9e34d797L, 0x2cf0b7d9L, 0x022b8b51L, 
	0x96d5ac3aL, 0x017da67dL, 0xd1cf3ed6L, 0x7c7d2d28L, 
	0x1f9f25cfL, 0xadf2b89bL, 0x5ad6b472L, 0x5a88f54cL, 
	0xe029ac71L, 0xe019a5e6L, 0x47b0acfdL, 0xed93fa9bL, 
	0xe8d3c48dL, 0x283b57ccL, 0xf8d56629L, 0x79132e28L, 
	0x785f0191L, 0xed756055L, 0xf7960e44L, 0xe3d35e8cL, 
	0x15056dd4L, 0x88f46dbaL, 0x03a16125L, 0x0564f0bdL, 
	0xc3eb9e15L, 0x3c9057a2L, 0x97271aecL, 0xa93a072aL, 
	0x1b3f6d9bL, 0x1e6321f5L, 0xf59c66fbL, 0x26dcf319L, 
	0x7533d928L, 0xb155fdf5L, 0x03563482L, 0x8aba3cbbL, 
	0x28517711L, 0xc20ad9f8L, 0xabcc5167L, 0xccad925fL, 
	0x4de81751L, 0x3830dc8eL, 0x379d5862L, 0x9320f991L, 
	0xea7a90c2L, 0xfb3e7bceL, 0x5121ce64L, 0x774fbe32L, 
	0xa8b6e37eL, 0xc3293d46L, 0x48de5369L, 0x6413e680L, 
	0xa2ae0810L, 0xdd6db224L, 0x69852dfdL, 0x09072166L, 
	0xb39a460aL, 0x6445c0ddL, 0x586cdecfL, 0x1c20c8aeL, 
	0x5bbef7ddL, 0x1b588d40L, 0xccd2017fL, 0x6bb4e3bbL, 
	0xdda26a7eL, 0x3a59ff45L, 0x3e350a44L, 0xbcb4cdd5L, 
	0x72eacea8L, 0xfa6484bbL, 0x8d6612aeL, 0xbf3c6f47L, 
	0xd29be463L, 0x542f5d9eL, 0xaec2771bL, 0xf64e6370L, 
	0x740e0d8dL, 0xe75b1357L, 0xf8721671L, 0xaf537d5dL, 
	0x4040cb08L, 0x4eb4e2ccL, 0x34d2466aL, 0x0115af84L, 
	0xe1b00428L, 0x95983a1dL, 0x06b89fb4L, 0xce6ea048L, 
	0x6f3f3b82L, 0x3520ab82L, 0x011a1d4bL, 0x277227f8L, 
	0x611560b1L, 0xe7933fdcL, 0xbb3a792bL, 0x344525bdL, 
	0xa08839e1L, 0x51ce794bL, 0x2f32c9b7L, 0xa01fbac9L, 
	0xe01cc87eL, 0xbcc7d1f6L, 0xcf0111c3L, 0xa1e8aac7L, 
	0x1a908749L, 0xd44fbd9aL, 0xd0dadecbL, 0xd50ada38L, 
	0x0339c32aL, 0xc6913667L, 0x8df9317cL, 0xe0b12b4fL, 
	0xf79e59b7L, 0x43f5bb3aL, 0xf2d519ffL, 0x27d9459cL, 
	0xbf97222cL, 0x15e6fc2aL, 0x0f91fc71L, 0x9b941525L, 
	0xfae59361L, 0xceb69cebL, 0xc2a86459L, 0x12baa8d1L, 
	0xb6c1075eL, 0xe3056a0cL, 0x10d25065L, 0xcb03a442L, 
	0xe0ec6e0eL, 0x1698db3bL, 0x4c98a0beL, 0x3278e964L, 
	0x9f1f9532L, 0xe0d392dfL, 0xd3a0342bL, 0x8971f21eL, 
	0x1b0a7441L, 0x4ba3348cL, 0xc5be7120L, 0xc37632d8L, 
	0xdf359f8dL, 0x9b992f2eL, 0xe60b6f47L, 0x0fe3f11dL, 
	0xe54cda54L, 0x1edad891L, 0xce6279cfL, 0xcd3e7e6fL, 
	0x1618b166L, 0xfd2c1d05L, 0x848fd2c5L, 0xf6fb2299L, 
	0xf523f357L, 0xa6327623L, 0x93a83531L, 0x56cccd02L, 
	0xacf08162L, 0x5a75ebb5L, 0x6e163697L, 0x88d273ccL, 
	0xde966292L, 0x81b949d0L, 0x4c50901bL, 0x71c65614L, 
	0xe6c6c7bdL, 0x327a140aL, 0x45e1d006L, 0xc3f27b9aL, 
	0xc9aa53fdL, 0x62a80f00L, 0xbb25bfe2L, 0x35bdd2f6L, 
	0x71126905L, 0xb2040222L, 0xb6cbcf7cL, 0xcd769c2bL, 
	0x53113ec0L, 0x1640e3d3L, 0x38abbd60L, 0x2547adf0L, 
	0xba38209cL, 0xf746ce76L, 0x77afa1c5L, 0x20756060L, 
	0x85cbfe4eL, 0x8ae88dd8L, 0x7aaaf9b0L, 0x4cf9aa7eL, 
	0x1948c25cL, 0x02fb8a8cL, 0x01c36ae4L, 0xd6ebe1f9L, 
	0x90d4f869L, 0xa65cdea0L, 0x3f09252dL, 0xc208e69fL, 
	0xb74e6132L, 0xce77e25bL, 0x578fdfe3L, 0x3ac372e6L, 
};
#if VERBOSE > 0
void print_long(uint32_t l) {
	LOG("%04x", (unsigned)((l>>16) & 0xffff));
	LOG("%04x\r\n",l & 0xffff);
}
#endif
void init()
{
#ifdef BOARD_MSP_TS430
	TBCTL &= 0xE6FF; //set 12,11 bit to zero (16bit) also 8 to zero (SMCLK)
	TBCTL |= 0x0200; //set 9 to one (SMCLK)
	TBCTL |= 0x00C0; //set 7-6 bit to 11 (divider = 8);
	TBCTL &= 0xFFEF; //set bit 4 to zero
	TBCTL |= 0x0020; //set bit 5 to one (5-4=10: continuous mode)
	TBCTL |= 0x0002; //interrupt enable
#endif
    //WISP_init();
	init_hw();
#ifdef CONFIG_EDB
   // debug_setup();
    edb_init();
#endif

    INIT_CONSOLE();

    __enable_interrupt();
#if 0
    GPIO(PORT_LED_1, DIR) |= BIT(PIN_LED_1);
    GPIO(PORT_LED_2, DIR) |= BIT(PIN_LED_2);
#if defined(PORT_LED_3)
    GPIO(PORT_LED_3, DIR) |= BIT(PIN_LED_3);
#endif

#if defined(PORT_LED_3) // when available, this LED indicates power-on
    GPIO(PORT_LED_3, OUT) |= BIT(PIN_LED_3);
#endif

#ifdef SHOW_PROGRESS_ON_LED
    blink(1, SEC_TO_CYCLES * 5, LED1 | LED2);
#endif
#endif
    EIF_PRINTF(".%u.\r\n", curtask);
}
void BF_encrypt(uint32_t *data, uint32_t *key){
        TASK_BOUNDARY(TASK_ENCRYPT);
        DINO_MANUAL_RESTORE_NONE();
	uint32_t l, r, p, s0_t, s1_t, s2_t, s3_t, tmp;
	r = data[0];
	l = data[1];
	for (unsigned index = 0; index < 17; ++index){
		p = key[index];

		if (index == 0) {
			r ^= p;
			++index;
		}
		p = key[index];
		l^=p; 
		s0_t = s0[(r>>24L)];
		s1_t = s1[((r>>16L)&0xff)];
		s2_t = s2[((r>> 8L)&0xff)];
		s3_t = s3[((r     )&0xff)];
		l^=(((	s0_t + 
			s1_t)^ 
			s2_t)+ 
			s3_t)&0xffffffff;

		tmp = r;
		r = l;
		l = tmp;
	}
	p = key[17];
	l ^= p;
	data[1] = r;
	data[0] = l;
	//CHECKPOINT
        TASK_BOUNDARY(TASK_ENCRYPT_END);
        DINO_MANUAL_RESTORE_NONE();
}
void BF_set_key(unsigned char *data, uint32_t *key){
	unsigned i;
	uint32_t ri, ri2;
	unsigned d = 0;
        TASK_BOUNDARY(TASK_INIT_S);
        DINO_MANUAL_RESTORE_NONE();
	for (i=0; i<18; ++i){
		ri= data[d++];

		d = (d >=8)? 0 : d;

		ri<<=8;
		ri2 = data[d++];
		ri |= ri2;
		d = (d >=8)? 0 : d;

		ri<<=8;
		ri2 = data[d++];
		ri |= ri2;
		d = (d >=8)? 0 : d;

		ri<<=8;
		ri2 = data[d++];
		ri |= ri2;
		d = (d >=8)? 0 : d;

		key[i]^=ri;
	}
        TASK_BOUNDARY(TASK_SET_KEY);
        DINO_MANUAL_RESTORE_NONE();
	uint32_t in[2]={0L,0L};
	BF_encrypt(in, key);
	uint32_t li;
	for (li=2; li< 256*4+20; li+=2){
		if(li < 20){
			key[li-2] = in[0];
			key[li-1] = in[1];
#if VERBOSE > 0
			LOG("key[%u]=", li-2);
			print_long(in[0]);
			LOG("key[%u]=", li-1);
			print_long(in[1]);	
#endif
			BF_encrypt(in, key);
		}
		else if(li < 256+20){
			s0[li-20] = in[0];
			s0[li-19] = in[1];
#if VERBOSE > 0
			if (li == 20 || li == 254 + 20) {
				LOG("s0[%u]=", li-20);
				print_long(in[0]);
				LOG("s0[%u]=", li-19);
				print_long(in[1]);
			}
#endif
			BF_encrypt(in, key);
		}
		else if(li < 512+20){
			s1[li-(256+20)] = in[0];
			s1[li-(256+19)] = in[1];
#if VERBOSE > 0
			if (li == 256 + 20 || li == (256*2 - 2) + 20) {
				LOG("s1[%u]=", li-(256 + 20));
				print_long(in[0]);
				LOG("s1[%u]=", li-(256 + 19));
				print_long(in[1]);
			}
#endif
			BF_encrypt(in, key);
		}
		else if(li < 256*3+20){
			s2[li-(256*2+20)] = in[0];
			s2[li-(256*2+19)] = in[1];
#if VERBOSE > 0
			if (li == 256*2 + 20 || li == (256*3 - 2) + 20) {
				LOG("s2[%u]=", li-(256*2 + 20));
				print_long(in[0]);
				LOG("s2[%u]=", li-(256*2 + 19));
				print_long(in[1]);
			}
#endif
			BF_encrypt(in, key);
		}
		else if(li < 256*4+20){
			s3[li-(256*3+20)] = in[0];
			s3[li-(256*3+19)] = in[1];
#if VERBOSE > 0
			if (li == 256*3 + 20 || li == (256*4 - 2) + 20) {
				LOG("s3[%u]=", li-(256*3 + 20));
				print_long(in[0]);
				LOG("s3[%u]=", li-(256*3 + 19));
				print_long(in[1]);
			}
#endif
			BF_encrypt(in, key);
		}
	}
}
void BF_cfb64_encrypt(unsigned char* out, unsigned char* iv, uint32_t *key){
	uint32_t ti[2];
	unsigned char c;
	unsigned n = 0;
	for (unsigned i=0; i< LENGTH; ++i){
		TASK_BOUNDARY(TASK_START_ENCRYPT);
		DINO_MANUAL_RESTORE_NONE();
		if (n == 0){
			for (unsigned j=0; j<8; ++j){
				LOG("before: iv[%u]=%u\r\n",j,iv[j]);
			}	
			ti[0] =((unsigned long)((iv[0])))<<24L;
			ti[0]|=((unsigned long)(iv[1]))<<16L;
			ti[0]|=((unsigned long)(iv[2]))<< 8L;
			ti[0]|=((unsigned long)(iv[3]));
			ti[1] =((unsigned long)(iv[4]))<<24L;
			ti[1]|=((unsigned long)(iv[5]))<<16L;
			ti[1]|=((unsigned long)(iv[6]))<< 8L;
			ti[1]|=((unsigned long)(iv[7]));
			test11 = 1;
			BF_encrypt(ti, key);
			test11 = 0;

			iv[0] = (unsigned char)(((ti[0])>>24L)&0xff);
			iv[1] = (unsigned char)(((ti[0])>>16L)&0xff);
			iv[2] = (unsigned char)(((ti[0])>> 8L)&0xff);
			iv[3] = (unsigned char)(((ti[0])     )&0xff);
			iv[4] = (unsigned char)(((ti[1])>>24L)&0xff);
			iv[5] = (unsigned char)(((ti[1])>>16L)&0xff);
			iv[6] = (unsigned char)(((ti[1])>> 8L)&0xff);
			iv[7] = (unsigned char)(((ti[1])     )&0xff);
#if VERBOSE > 0
			for (unsigned j=0; j<8; ++j){
				LOG("iv[%u]=%u\r\n",j,iv[j]);
			}	
#endif
		}
		TASK_BOUNDARY(TASK_START_ENCRYPT2);
		DINO_MANUAL_RESTORE_NONE();
		c= indata[i]^iv[n];
		out[i]=c;
		PRINTF("result: %x\r\n", c);
		iv[n]=c;
		n=(n+1)&0x07;
	}
}


int main()
{
	uint32_t key[18];
	init();

	DINO_RESTORE_CHECK();

	unsigned char ukey[16];
	unsigned char indata[40], outdata[40], ivec[8];

	unsigned i = 0, by = 0;	

	for (i = 0; i < 8; ++i){
		ivec[i] = 0;
	}
	i = 0;
	//CHECKPOINT
        TASK_BOUNDARY(TASK_INIT);
        DINO_MANUAL_RESTORE_NONE();
	while (i < 32) {
		if(cp[i] >= '0' && cp[i] <= '9')
			by = (by << 4) + cp[i] - '0';
		else if(cp[i] >= 'A' && cp[i] <= 'F') //currently, key should be 0-9 or A-F
			by = (by << 4) + cp[i] - 'A' + 10;
		else
			PRINTF("Key must be hexadecimal!!\r\n");
		if ((i++) & 1) {
			ukey[i/2-1] = by & 0xff;
			LOG("ukey[%u]: %u\r\n",i/2-1,by & 0xff);
		}

	}
        TASK_BOUNDARY(TASK_SET_UKEY);
        DINO_MANUAL_RESTORE_NONE();
	for (i = 0; i < 18; ++i)
		key[i] = init_key[i];
	
	for (i = 0; i < 1024; ++i) {
		if (i == 0 || i == 256 || i == 256*2 || i == 256*3){
			TASK_BOUNDARY(TASK_INIT_KEY);
			DINO_MANUAL_RESTORE_NONE();
		}
		if (i < 256) 
			s0[i] = init_s0[i];
		else if (i < 256*2)
			s1[i-256] = init_s1[i-256];
		else if (i < 256*3)
			s2[i-256*2] = init_s2[i-256*2];
		else 
			s3[i-256*3] = init_s3[i-256*3];
	}
	BF_set_key(ukey, key);
	BF_cfb64_encrypt(outdata, ivec, key);
	PRINTF("TIME end is 65536*%u+%u\r\n",overflow,(unsigned)TBR);
}


