/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/mma_base.h"

#include "cutlass/util/debug.h"
#include "cutlass/util/device_dump.h"

//#define DEBUG_INTERNAL_IMP
//#define DEBUG_TILE_INDICES_FILTER
//#define DEBUG_TILE_INDICES_ACTIVATION
//#define DEBUG_FRAGMENT
//#define DEBUG_SHRMEM

#define DEBUG_KGROUP

#ifdef DEBUG_INTERNAL_IMP
  #define PREQ
#endif

#ifdef DEBUG_FRAGMENT
  #define PREQ
#endif

#ifdef DEBUG_TILE_INDICES_FILTER
  #define PREQ
#endif

#ifdef DEBUG_TILE_INDICES_ACTIVATION
  #define PREQ
#endif

#ifdef DEBUG_SHRMEM
  #define PREQ
#endif

#ifdef DEBUG_KGROUP
  #define PREQ
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  /// Iterates over tiles of A operand in global memory 
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorA_,
  /// Iterates over tiles of A operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorA_,
  /// Iterates over tiles of B operand in global memory
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorB_,
  /// Iterates over tiles of B operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorB_,
  /// Data type of accumulator matrix
  typename ElementC_,
  /// Data type of accumulator matrix
  typename LayoutC_,
  /// Policy describing tuning details (concept: MmaPolicy)
  typename Policy_,
  /// Transformation applied to A operand
  typename TransformA_ = NumericArrayConverter<
    typename SmemIteratorA_::Element, 
    typename IteratorA_::Element, 
    IteratorA_::Fragment::kElements>,
  ///
  /// Transformation applied to A operand
  typename TransformB_ = NumericArrayConverter<
    typename SmemIteratorB_::Element, 
    typename IteratorB_::Element, 
    IteratorB_::Fragment::kElements>,
  /// Used for partial specialization
  typename Enable = bool
>
class ImplicitGemmPipelined : public gemm::threadblock::MmaBase<Shape_, Policy_, 2> {
public:

  ///< Base class
  using Base = gemm::threadblock::MmaBase<Shape_, Policy_, 2>;

  using Shape = Shape_;             ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using IteratorA = IteratorA_;     ///< Iterates over tiles of A operand in global memory
  using IteratorB = IteratorB_;     ///< Iterates over tiles of B operand in global memory
  using ElementC = ElementC_;       ///< Data type of accumulator matrix
  using LayoutC = LayoutC_;         ///< Layout of accumulator matrix
  using Policy = Policy_;           ///< Policy describing tuning details

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  using TransformA = TransformA_;
  using TransformB = TransformB_;

  //
  // Dependent types
  //

  /// Fragment of operand A loaded from global memory
  using FragmentA = typename IteratorA::Fragment;

  /// Fragment of operand B loaded from global memory
  using FragmentB = typename IteratorB::Fragment;

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Obtain the arch tag from the warp-level operator
  using ArchTag = typename Policy::Operator::ArchTag;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  using SharedStorage = typename Base::SharedStorage;
  // staticaly assert kStages for MmaPipelined is two (Double-buffered pipeline)
  static_assert((Base::kStages==2), "MmaPipelined requires kStages set to value 2");

private:

  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;


protected:
  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  SharedStorage shared_storage_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  ImplicitGemmPipelined(
    typename Base::SharedStorage &shared_storage,       ///< Shared storage needed for internal use by threadblock-scoped GEMM
    int thread_idx,                                     ///< ID within the threadblock
    int warp_idx,                                       ///< ID of warp
    int lane_idx                                        ///< ID of each thread within a warp
  ):
	shared_storage_(shared_storage),
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
    smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx) {

    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,                            ///< number of iterations of the mainloop
    FragmentC &accum,                                 ///< destination accumulator tile
    IteratorA iterator_A,                             ///< iterator over A operand in global memory
    IteratorB iterator_B,                             ///< iterator over B operand in global memory
    FragmentC const &src_accum,                       ///< source accumulator tile
    int gemm_k_iterations_per_channel = 0,             ///< number of iterations per channel
    TransformA transform_A = TransformA(),            ///< transformation applied to A fragment
    TransformB transform_B = TransformB()) {          ///< transformation applied to B fragment

    //
    // Prologue
    //

    // Perform accumulation in the 'd' output operand
    #ifdef PREQ
      int thread_id = (threadIdx.z * (blockDim.x * blockDim.y)) +
                  (threadIdx.y * blockDim.x) + threadIdx.x;
      int block_id = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    #endif

    accum = src_accum;

    FragmentA tb_frag_A;
    FragmentB tb_frag_B;

    tb_frag_A.clear();
    tb_frag_B.clear();

	// The last kblock is loaded in the prolog
	
	#ifdef DEBUG_INTERNAL_IMP

    if (thread_id == 0 && block_id == 0)
      printf("\n*******************Load for Activation here*******************\n\n");
	#endif

    iterator_A.load(tb_frag_A);
	
	#ifdef DEBUG_INTERNAL_IMP
	__syncthreads();

    if (thread_id == 0 && block_id == 0)
      printf("\n*******************Load for Filter here*******************\n\n");
	#endif

    iterator_B.load(tb_frag_B);


    #ifdef DEBUG_FRAGMENT
    if (thread_id == 0 && block_id == 0)
      printf("\n*******************Prolog starts here*******************\n\n");

    debug::dump_fragment(tb_frag_B, 0);

    if (thread_id == 0 && block_id == 0)
      printf("\n*******************Prolog ends here*******************\n\n");
    #endif

    #ifdef DEBUG_TILE_INDICES_FILTER
    if (thread_id == 0 && block_id == 0)
      printf("\n*******************Prolog starts here*******************\n\n");
	
	__syncthreads();

    debug::dump_tile_indices(iterator_B, 0, 0);

	__syncthreads();
    
	if (thread_id == 0 && block_id == 0)
      printf("\n*******************Prolog ends here*******************\n\n");
    #endif


    #ifdef DEBUG_TILE_INDICES_ACTIVATION

    if (thread_id == 0 && block_id == 0)
      printf("\n*******************Prolog starts here*******************\n\n");
	
	__syncthreads();

    debug::dump_tile_indices(iterator_A, 0, 0);

	__syncthreads();
    
	if (thread_id == 0 && block_id == 0)
      printf("\n*******************Prolog ends here*******************\n\n");
    #endif

    ++iterator_A;
    ++iterator_B;

    this->smem_iterator_A_.store(transform_A(tb_frag_A));
    this->smem_iterator_B_.store(transform_B(tb_frag_B));

    #ifdef DEBUG_SHRMEM

    if (thread_id == 0 && block_id == 0)
      printf("\n*******************Prolog starts here*******************\n\n");
	
	__syncthreads();

    debug::dump_shmem(shared_storage_.operand_A_ref().data(), 128 * 128, 1);

	__syncthreads();
    
	if (thread_id == 0 && block_id == 0)
      printf("\n*******************Prolog ends here*******************\n\n");
    #endif

    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math instructions
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);
	
    #ifdef DEBUG_KGROUP

    if (thread_id == 0 && block_id == 0)
      printf("\n*******************Prolog starts here*******************\n\n");
	
	__syncthreads();

    debug::dump_mma_internal_state(this->warp_tile_iterator_B_, 0, 0, -1);

	__syncthreads();
    
	if (thread_id == 0 && block_id == 0)
      printf("\n*******************Prolog ends here*******************\n\n");
    #endif

    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    Operator warp_mma;

    int smem_write_stage_idx = 1;

    // Issue loads during the first warp-level matrix multiply-add *AFTER* issuing 
    // shared memory loads (which have the tightest latency requirement).

    //
    // Mainloop
    //

    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
        // as the case may be.

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {

          // Write fragments to shared memory
          this->smem_iterator_A_.store(transform_A(tb_frag_A));

          this->smem_iterator_B_.store(transform_B(tb_frag_B));

          __syncthreads();
          
          ++this->smem_iterator_A_;
          ++this->smem_iterator_B_;

          // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
          if (smem_write_stage_idx == 1) {
            this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
          }
          else {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations,
                 0});
          }

          smem_write_stage_idx ^= 1;
        }

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);

		#ifdef DEBUG_KGROUP

		if (thread_id == 0 && block_id == 0)
		  printf("\n*******************Kgroup for iteration %d begins*******************\n\n", gemm_k_iterations);
		
		__syncthreads();

		debug::dump_mma_internal_state(this->warp_tile_iterator_B_, 0, 0, -1);

		__syncthreads();
		
		if (thread_id == 0 && block_id == 0)
		  printf("\n*******************Kgroup for iteration %d ends*******************\n\n", gemm_k_iterations);
		#endif
        
        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0) {

		#ifdef DEBUG_INTERNAL_IMP
		 if (thread_id == 0 && block_id == 0)
			  printf("\n*******************Load for Activation here*******************\n\n");
		
		#endif

          iterator_A.load(tb_frag_A);
		
		#ifdef DEBUG_INTERNAL_IMP
		__syncthreads();		

    	if (thread_id == 0 && block_id == 0)
     		 printf("\n*******************Load for Filter here*******************\n\n");
		#endif

          iterator_B.load(tb_frag_B);


          #ifdef DEBUG_FRAGMENT
          if (thread_id == 0 && block_id == 0)
            printf("\n******************* Gemm Iteration %d begins here *******************\n", gemm_k_iterations);
          debug::dump_fragment(tb_frag_B, 0);
          if (thread_id == 0 && block_id == 0)
            printf("\n******************* Gemm Iteration %d ends here *******************\n", gemm_k_iterations);
          #endif

          #ifdef DEBUG_TILE_INDICES_FILTER
		  __syncthreads();
          if (threadIdx.x == 0 && blockIdx.x == 0)
            printf("\n******************* Gemm Iteration %d begins here *******************\n", gemm_k_iterations);
              
          debug::dump_tile_indices(iterator_B, 0, 0);

	 	  __syncthreads();
          if (threadIdx.x == 0 && blockIdx.x == 0)
            printf("\n******************* Gemm Iteration %d ends here *******************\n", gemm_k_iterations);

          #endif

		#ifdef DEBUG_TILE_INDICES_ACTIVATION
		__syncthreads();
		if (thread_id == 0 && block_id == 0)
		  printf("\n*******************Gemm Iteration %d begins here*******************\n", gemm_k_iterations);
		
		debug::dump_tile_indices(iterator_A, 0, 0);

		__syncthreads();
		if (thread_id == 0 && block_id == 0)
            printf("\n******************* Gemm Iteration %d ends here *******************\n", gemm_k_iterations);
		#endif
    
          ++iterator_A;
          ++iterator_B;
        }

        warp_mma(accum, warp_frag_A[warp_mma_k % 2],
                 warp_frag_B[warp_mma_k % 2], accum);
      }
    }

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
