#include "textflag.h"

// One L2 distance with the same four-lane accumulation and reduction used by
// the ANN scalar fallback.
TEXT ·squaredEuclideanDistance32SSE(SB), NOSPLIT, $0-52
	MOVQ a_base+0(FP), AX
	MOVQ b_base+24(FP), BX
	MOVQ a_len+8(FP), CX
	XORQ DX, DX
	XORPS X0, X0

single_vector_loop:
	MOVQ CX, R8
	SUBQ DX, R8
	CMPQ R8, $4
	JL single_tail
	MOVUPS (AX)(DX*4), X4
	MOVUPS (BX)(DX*4), X5
	SUBPS X5, X4
	MULPS X4, X4
	ADDPS X4, X0
	ADDQ $4, DX
	JMP single_vector_loop

single_tail:
	CMPQ DX, CX
	JGE single_reduce
	MOVSS (AX)(DX*4), X4
	MOVSS (BX)(DX*4), X5
	SUBSS X5, X4
	MULSS X4, X4
	ADDSS X4, X0
	INCQ DX
	JMP single_tail

single_reduce:
	MOVAPS X0, X4
	MOVAPS X0, X5
	SHUFPS $0x55, X5, X5
	ADDSS X5, X4
	MOVAPS X0, X5
	MOVAPS X0, X6
	SHUFPS $0xaa, X5, X5
	SHUFPS $0xff, X6, X6
	ADDSS X6, X5
	ADDSS X5, X4
	MOVSS X4, sum+48(FP)
	RET

// Four L2 distances with the same four-lane float32 accumulation and
// (lane0+lane1)+(lane2+lane3) reduction used by the Go ANN kernel.
TEXT ·squaredEuclideanDistance32x4SSE(SB), NOSPLIT, $0-136
	MOVQ a0_base+0(FP), AX
	MOVQ a1_base+24(FP), BX
	MOVQ a2_base+48(FP), R9
	MOVQ a3_base+72(FP), R10
	MOVQ b_base+96(FP), R11
	MOVQ a0_len+8(FP), CX
	XORQ DX, DX
	XORPS X0, X0
	XORPS X1, X1
	XORPS X2, X2
	XORPS X3, X3

vector_loop:
	MOVQ CX, R12
	SUBQ DX, R12
	CMPQ R12, $4
	JL tail
	MOVUPS (R11)(DX*4), X8
	MOVUPS (AX)(DX*4), X4
	SUBPS X8, X4
	MULPS X4, X4
	ADDPS X4, X0
	MOVUPS (BX)(DX*4), X5
	SUBPS X8, X5
	MULPS X5, X5
	ADDPS X5, X1
	MOVUPS (R9)(DX*4), X6
	SUBPS X8, X6
	MULPS X6, X6
	ADDPS X6, X2
	MOVUPS (R10)(DX*4), X7
	SUBPS X8, X7
	MULPS X7, X7
	ADDPS X7, X3
	ADDQ $4, DX
	JMP vector_loop

tail:
	CMPQ DX, CX
	JGE reduce
	MOVSS (R11)(DX*4), X8
	MOVSS (AX)(DX*4), X4
	SUBSS X8, X4
	MULSS X4, X4
	ADDSS X4, X0
	MOVSS (BX)(DX*4), X5
	SUBSS X8, X5
	MULSS X5, X5
	ADDSS X5, X1
	MOVSS (R9)(DX*4), X6
	SUBSS X8, X6
	MULSS X6, X6
	ADDSS X6, X2
	MOVSS (R10)(DX*4), X7
	SUBSS X8, X7
	MULSS X7, X7
	ADDSS X7, X3
	INCQ DX
	JMP tail

reduce:
	MOVAPS X0, X4
	MOVAPS X0, X5
	SHUFPS $0x55, X5, X5
	ADDSS X5, X4
	MOVAPS X0, X5
	MOVAPS X0, X6
	SHUFPS $0xaa, X5, X5
	SHUFPS $0xff, X6, X6
	ADDSS X6, X5
	ADDSS X5, X4
	MOVSS X4, sum0+120(FP)

	MOVAPS X1, X4
	MOVAPS X1, X5
	SHUFPS $0x55, X5, X5
	ADDSS X5, X4
	MOVAPS X1, X5
	MOVAPS X1, X6
	SHUFPS $0xaa, X5, X5
	SHUFPS $0xff, X6, X6
	ADDSS X6, X5
	ADDSS X5, X4
	MOVSS X4, sum1+124(FP)

	MOVAPS X2, X4
	MOVAPS X2, X5
	SHUFPS $0x55, X5, X5
	ADDSS X5, X4
	MOVAPS X2, X5
	MOVAPS X2, X6
	SHUFPS $0xaa, X5, X5
	SHUFPS $0xff, X6, X6
	ADDSS X6, X5
	ADDSS X5, X4
	MOVSS X4, sum2+128(FP)

	MOVAPS X3, X4
	MOVAPS X3, X5
	SHUFPS $0x55, X5, X5
	ADDSS X5, X4
	MOVAPS X3, X5
	MOVAPS X3, X6
	SHUFPS $0xaa, X5, X5
	SHUFPS $0xff, X6, X6
	ADDSS X6, X5
	ADDSS X5, X4
	MOVSS X4, sum3+132(FP)
	RET

// Arena variant of the kernel above. The four candidate pointers are derived
// once from a shared base and element offsets, avoiding four slice headers in
// the HNSW expansion loop.
TEXT ·squaredEuclideanDistance32x4ArenaSSE(SB), NOSPLIT, $0-104
	MOVQ arena_base+0(FP), R8
	MOVQ offset0+24(FP), AX
	LEAQ (R8)(AX*4), AX
	MOVQ offset1+32(FP), BX
	LEAQ (R8)(BX*4), BX
	MOVQ offset2+40(FP), R9
	LEAQ (R8)(R9*4), R9
	MOVQ offset3+48(FP), R10
	LEAQ (R8)(R10*4), R10
	MOVQ dim+56(FP), CX
	MOVQ b_base+64(FP), R11
	XORQ DX, DX
	XORPS X0, X0
	XORPS X1, X1
	XORPS X2, X2
	XORPS X3, X3

arena_vector_loop:
	MOVQ CX, R12
	SUBQ DX, R12
	CMPQ R12, $4
	JL arena_tail
	MOVUPS (R11)(DX*4), X8
	MOVUPS (AX)(DX*4), X4
	SUBPS X8, X4
	MULPS X4, X4
	ADDPS X4, X0
	MOVUPS (BX)(DX*4), X5
	SUBPS X8, X5
	MULPS X5, X5
	ADDPS X5, X1
	MOVUPS (R9)(DX*4), X6
	SUBPS X8, X6
	MULPS X6, X6
	ADDPS X6, X2
	MOVUPS (R10)(DX*4), X7
	SUBPS X8, X7
	MULPS X7, X7
	ADDPS X7, X3
	ADDQ $4, DX
	JMP arena_vector_loop

arena_tail:
	CMPQ DX, CX
	JGE arena_reduce
	MOVSS (R11)(DX*4), X8
	MOVSS (AX)(DX*4), X4
	SUBSS X8, X4
	MULSS X4, X4
	ADDSS X4, X0
	MOVSS (BX)(DX*4), X5
	SUBSS X8, X5
	MULSS X5, X5
	ADDSS X5, X1
	MOVSS (R9)(DX*4), X6
	SUBSS X8, X6
	MULSS X6, X6
	ADDSS X6, X2
	MOVSS (R10)(DX*4), X7
	SUBSS X8, X7
	MULSS X7, X7
	ADDSS X7, X3
	INCQ DX
	JMP arena_tail

arena_reduce:
	MOVAPS X0, X4
	MOVAPS X0, X5
	SHUFPS $0x55, X5, X5
	ADDSS X5, X4
	MOVAPS X0, X5
	MOVAPS X0, X6
	SHUFPS $0xaa, X5, X5
	SHUFPS $0xff, X6, X6
	ADDSS X6, X5
	ADDSS X5, X4
	MOVSS X4, sum0+88(FP)

	MOVAPS X1, X4
	MOVAPS X1, X5
	SHUFPS $0x55, X5, X5
	ADDSS X5, X4
	MOVAPS X1, X5
	MOVAPS X1, X6
	SHUFPS $0xaa, X5, X5
	SHUFPS $0xff, X6, X6
	ADDSS X6, X5
	ADDSS X5, X4
	MOVSS X4, sum1+92(FP)

	MOVAPS X2, X4
	MOVAPS X2, X5
	SHUFPS $0x55, X5, X5
	ADDSS X5, X4
	MOVAPS X2, X5
	MOVAPS X2, X6
	SHUFPS $0xaa, X5, X5
	SHUFPS $0xff, X6, X6
	ADDSS X6, X5
	ADDSS X5, X4
	MOVSS X4, sum2+96(FP)

	MOVAPS X3, X4
	MOVAPS X3, X5
	SHUFPS $0x55, X5, X5
	ADDSS X5, X4
	MOVAPS X3, X5
	MOVAPS X3, X6
	SHUFPS $0xaa, X5, X5
	SHUFPS $0xff, X6, X6
	ADDSS X6, X5
	ADDSS X5, X4
	MOVSS X4, sum3+100(FP)
	RET
