"""Dense float32 GEMM with warp tiles (own module so its expensive compile is cached separately)."""

import warp as wp

TILE_M = 64
TILE_N = 64
TILE_K = 32
BLOCK_DIM = 128


@wp.kernel(enable_backward=False)
def gemm_tiled(A: wp.array2d(dtype=wp.float32), B: wp.array2d(dtype=wp.float32), C: wp.array2d(dtype=wp.float32)):
	# C = A @ B; all shapes must be multiples of the tile sizes (callers pad with zeros)
	i, j = wp.tid()
	acc = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=wp.float32)
	count = A.shape[1] // TILE_K
	for k in range(count):
		a = wp.tile_load(A, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
		b = wp.tile_load(B, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))
		wp.tile_matmul(a, b, acc)
	wp.tile_store(C, acc, offset=(i * TILE_M, j * TILE_N))


def gemm(A, B, C, device):
	M, K = A.shape
	N = B.shape[1]
	assert M % TILE_M == 0 and N % TILE_N == 0 and K % TILE_K == 0 and B.shape[0] == K
	wp.launch_tiled(gemm_tiled, dim=[M // TILE_M, N // TILE_N], inputs=[A, B, C], block_dim=BLOCK_DIM,
		device=device)
