/**
 * \file tensor_parallel_nccl.cu
 * \brief NCCL-backed collective communication for tensor parallelism.
 *
 * Provides ring all-reduce and all-gather using NCCL when available
 * (compiled with -DHAVE_NCCL). Falls back to no-op stubs otherwise so
 * the rest of the build is unaffected on machines without NCCL.
 *
 * \par Single-GPU testing strategy
 * On a single-GPU machine, use ncclCommInitAll with devs = {0, 0}.
 * NCCL recognises the duplicate device index and selects SHM transport,
 * which exercises the full NCCL API surface without requiring PCIe/NVLink.
 * ncclGroupStart / ncclGroupEnd bracket multi-rank calls within one process.
 *
 * \par Multi-GPU production use
 * Pass distinct device indices (e.g. {0, 1, 2, 3}) to ncclCommInitAll.
 * NCCL selects the optimal transport (NVLink, PCIe, or SHM) automatically.
 *
 * \par API
 * - nccl_comm_init_all(comms, num_comms, devs)  -- init R communicators
 * - nccl_comm_destroy(comm)                      -- teardown one comm
 * - nccl_group_start() / nccl_group_end()        -- bracket multi-rank calls
 * - launch_ring_allreduce_nccl(comm, buf, count, stream)
 * - launch_allgather_nccl(comm, send, recv, sendcount, stream)
 * - nccl_comm_count(comm) -> int                 -- number of ranks in comm
 * - nccl_comm_user_rank(comm) -> int             -- rank of this comm
 */

#include <cuda_runtime.h>
#include <cstdio>

#ifdef HAVE_NCCL
#include <nccl.h>

// ---------------------------------------------------------------------------
// Internal error-checking macro
// ---------------------------------------------------------------------------

#define NCCL_CHECK(expr) do {                                                  \
    ncclResult_t _r = (expr);                                                  \
    if (_r != ncclSuccess) {                                                   \
        fprintf(stderr, "[NCCL] error at %s:%d: %s\n",                        \
                __FILE__, __LINE__, ncclGetErrorString(_r));                   \
    }                                                                          \
} while (0)

// ---------------------------------------------------------------------------
// Communicator lifecycle
// ---------------------------------------------------------------------------

/**
 * \brief Initialise \p num_comms NCCL communicators covering the listed devices.
 *
 * \param[out] comms      Array of \p num_comms void* to be filled.
 * \param[in]  num_comms  Number of ranks (may equal 1 for single-rank use).
 * \param[in]  devs       Array of CUDA device indices (may repeat for SHM test).
 */
extern "C" void nccl_comm_init_all(void** comms, int num_comms, const int* devs)
{
    NCCL_CHECK(ncclCommInitAll(reinterpret_cast<ncclComm_t*>(comms),
                               num_comms, devs));
}

/**
 * \brief Destroy a single NCCL communicator.
 *
 * \param[in] comm  Communicator returned by nccl_comm_init_all.
 */
extern "C" void nccl_comm_destroy(void* comm)
{
    if (comm) NCCL_CHECK(ncclCommDestroy(static_cast<ncclComm_t>(comm)));
}

/**
 * \brief Start a NCCL group (allows multi-rank calls from one process/thread).
 */
extern "C" void nccl_group_start(void)
{
    NCCL_CHECK(ncclGroupStart());
}

/**
 * \brief End a NCCL group and flush pending collectives.
 */
extern "C" void nccl_group_end(void)
{
    NCCL_CHECK(ncclGroupEnd());
}

// ---------------------------------------------------------------------------
// Collective operations
// ---------------------------------------------------------------------------

/**
 * \brief NCCL-backed in-place all-reduce (sum).
 *
 * Must be called from each rank's thread inside a ncclGroupStart/End bracket
 * when multiple ranks share one process.
 *
 * \param[in]     comm    Rank's communicator.
 * \param[in,out] d_buf   Device buffer to reduce in-place [count].
 * \param[in]     count   Number of float elements.
 * \param[in]     stream  CUDA stream for the operation.
 */
extern "C" void launch_ring_allreduce_nccl(
    void* comm, float* d_buf, int count, cudaStream_t stream)
{
    NCCL_CHECK(ncclAllReduce(
        d_buf, d_buf, (size_t)count,
        ncclFloat, ncclSum,
        static_cast<ncclComm_t>(comm), stream));
}

/**
 * \brief NCCL-backed all-gather.
 *
 * Each rank contributes \p sendcount floats from \p d_sendbuf.
 * \p d_recvbuf must have space for num_ranks * sendcount floats.
 * The output is laid out as [rank0_chunk | rank1_chunk | ...].
 *
 * \param[in]  comm       Rank's communicator.
 * \param[in]  d_sendbuf  This rank's input chunk [sendcount].
 * \param[out] d_recvbuf  Concatenated output [num_ranks * sendcount].
 * \param[in]  sendcount  Elements contributed by this rank.
 * \param[in]  stream     CUDA stream.
 */
extern "C" void launch_allgather_nccl(
    void* comm, const float* d_sendbuf, float* d_recvbuf,
    int sendcount, cudaStream_t stream)
{
    NCCL_CHECK(ncclAllGather(
        d_sendbuf, d_recvbuf, (size_t)sendcount,
        ncclFloat,
        static_cast<ncclComm_t>(comm), stream));
}

// ---------------------------------------------------------------------------
// Introspection helpers (useful for tests and Python bindings)
// ---------------------------------------------------------------------------

/**
 * \brief Return the number of ranks in the communicator.
 *
 * \param[in] comm  Communicator handle.
 * \return Number of ranks, or 0 on error.
 */
extern "C" int nccl_comm_count(void* comm)
{
    int count = 0;
    NCCL_CHECK(ncclCommCount(static_cast<ncclComm_t>(comm), &count));
    return count;
}

/**
 * \brief Return the rank of this communicator within the group.
 *
 * \param[in] comm  Communicator handle.
 * \return Rank index (0-based), or -1 on error.
 */
extern "C" int nccl_comm_user_rank(void* comm)
{
    int rank = -1;
    NCCL_CHECK(ncclCommUserRank(static_cast<ncclComm_t>(comm), &rank));
    return rank;
}

#else  // -----------------------------------------------------------------------
// Stub implementations when NCCL is not available
// ---------------------------------------------------------------------------

extern "C" void nccl_comm_init_all(void** comms, int num_comms, const int* devs)
{
    (void)comms; (void)num_comms; (void)devs;
    fprintf(stderr, "[NCCL stub] NCCL not available — recompile with -DHAVE_NCCL\n");
}

extern "C" void nccl_comm_destroy(void* comm)
{
    (void)comm;
}

extern "C" void nccl_group_start(void) {}
extern "C" void nccl_group_end(void) {}

extern "C" void launch_ring_allreduce_nccl(
    void* comm, float* d_buf, int count, cudaStream_t stream)
{
    (void)comm; (void)d_buf; (void)count; (void)stream;
    fprintf(stderr, "[NCCL stub] NCCL not available\n");
}

extern "C" void launch_allgather_nccl(
    void* comm, const float* d_sendbuf, float* d_recvbuf,
    int sendcount, cudaStream_t stream)
{
    (void)comm; (void)d_sendbuf; (void)d_recvbuf; (void)sendcount; (void)stream;
    fprintf(stderr, "[NCCL stub] NCCL not available\n");
}

extern "C" int nccl_comm_count(void* comm)  { (void)comm; return 0; }
extern "C" int nccl_comm_user_rank(void* comm) { (void)comm; return -1; }

#endif  // HAVE_NCCL
