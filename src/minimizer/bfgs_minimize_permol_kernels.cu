#include "bfgs_minimize_permol_kernels.h"
#include "mmff_kernels.h"
#include "mmff_kernels_device.cuh"
#include "device_vector.h"

#include <cub/cub.cuh>

namespace nvMolKit {

namespace {
constexpr int BLOCK_SIZE = 128;
constexpr int MAX_LINESEARCH_ITERS = 1000;
constexpr double FUNCTOL = 1e-4;
constexpr double MOVETOL = 1e-7;
constexpr double TOLX = 4. * 3e-8;

__device__ void setMaxStep(const double* pos, const int numTerms, double* maxStepOutSquared,
                           typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  double sumSquaredPos = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    double dx2 = pos[i] * pos[i];
    sumSquaredPos += dx2;
  }
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;

  const double squaredSum = BlockReduce(tempStorage).Sum(sumSquaredPos);
  if (threadIdx.x == 0) {
    constexpr double maxStepFactorSquared = 100.0 * 100.0;
    *maxStepOutSquared = maxStepFactorSquared * max(squaredSum, static_cast<double>(numTerms) * static_cast<double>(numTerms));
  }
}

__device__ void lineSearchSetup(const int numTerms, const double* posStart, const double* gradStart, const double maxStepSquared, double* dirStart, double& slope,  double& lambdaMin,
                                typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {

  const int idxInSys = threadIdx.x;
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
  __shared__ double dirSumSquared;

  // ---------------------------------
  //  Scale direction vector if needed
  // ---------------------------------
  double sumSquaredLocal = 0.0;
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    double dx2 = dirStart[i] * dirStart[i];
    sumSquaredLocal += dx2;
  }
  double blockSum = BlockReduce(tempStorage).Sum(sumSquaredLocal);
  if (idxInSys == 0) {
      dirSumSquared = blockSum;
  }
  __syncthreads();
  if (dirSumSquared < maxStepSquared) {
    double scale = sqrt(maxStepSquared) * rsqrt(dirSumSquared);
    for (int i = idxInSys; i < numTerms; i += blockDim.x) {
      dirStart[i] *= scale;
    }
  }
  __syncthreads();

  // -------------------------
  // Set slope, check validity
  // -------------------------
  double localSum = 0.0;
  double localGradSum = 0.0;
  double localDirSum = 0.0;
  // Each thread computes its partial sum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    localSum += dirStart[i] * gradStart[i];
    localGradSum += gradStart[i] * gradStart[i];
    localDirSum += dirStart[i] * dirStart[i];
  }

  // Perform block-wide reduction to compute the total sum
  blockSum = BlockReduce(tempStorage).Sum(localSum);
  
  // The first thread in the block writes the result
  if (idxInSys == 0) {
    slope = blockSum;
  }
  __syncthreads();

  // ----------------------
  // Compute initial lambda
  // ----------------------
  double localMax_numerator = 0.0;
  double localMax_denominator = 1.0;
  // Each thread computes its local maximum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    double temp_numerator = fabs(dirStart[i]);
    double temp_denominator = fmax(fabs(posStart[i]), 1.0);
    // temp_numerator / temp_denominator > localMax_numerator / localMax_denominator
    // <=>
    // temp_numerator * localMax_denominator > localMax_numerator * temp_denominator
    if (temp_numerator * localMax_denominator > localMax_numerator * temp_denominator) {
        localMax_numerator = temp_numerator;
        localMax_denominator = temp_denominator;
    }

  }
  
  double localMax = localMax_numerator / localMax_denominator;
  // Perform block-wide reduction to find the maximum
  double blockMax = BlockReduce(tempStorage).Reduce(localMax, cub::Max());

  // The first thread in the block writes the result
  if (threadIdx.x == 0) {
    lambdaMin = MOVETOL / blockMax;
  }
}

__device__ void lineSearchPerturb(const int numTerms, 
                                  const double* refPos,
                                  const double* dirStart,
                                  const double lambda,
                                  double* scratchPos) {
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    scratchPos[i] = refPos[i] + lambda * dirStart[i];
  }
  __syncthreads();
}

__device__ bool lineSearchPostEnergy(const bool isFirstIter,
                                     const double prevE,
                                     const double newE,
                                     const double slope,
                                     const double lambda,
                                     const double lambdaMin,
                                     double& lambda2,
                                     double& eScratch,
                                     double& lambdaOut) {
  bool converged = false;
  
  if (threadIdx.x == 0) {
    double eDiff = newE - prevE;
    double threshold = FUNCTOL * lambda * slope;
    if (lambda < lambdaMin) {
      converged = true;
    } else if (eDiff <= threshold) {
      converged = true;
    } else {
      // Need to backtrack
      double tmpLambda;
      if (isFirstIter) {
        tmpLambda = -slope / (2.0 * (newE - prevE - slope));
      } else {
        double rhs1 = newE - prevE - lambda * slope;
        double rhs2 = eScratch - prevE - lambda2 * slope;
        double rLambdaSquared = 1.0 / (lambda * lambda);
        double rLambda2Squared = 1.0 / (lambda2 * lambda2);
        double unscaled_a = rhs1 * rLambdaSquared - rhs2 * rLambda2Squared;
        double unscaled_b = -lambda2 * rhs1 * rLambdaSquared + lambda * rhs2 * rLambda2Squared;
        double scale = lambda - lambda2;
        double unscaled_slope = slope * scale;
        if (unscaled_a == 0.0) {
          tmpLambda = -unscaled_slope / (2.0 * unscaled_b);
        } else {
          double unscaled_disc = unscaled_b * unscaled_b - 3 * unscaled_a * unscaled_slope;
          if (unscaled_disc < 0.0) {
            tmpLambda = 0.5 * lambda;
          } else if ((unscaled_b == 0.0) || ((unscaled_b > 0.0) != (scale > 0.0))) {
            tmpLambda = (-unscaled_b + sqrt(unscaled_disc)) / (3.0 * unscaled_a);
          } else {
            tmpLambda = -unscaled_slope / (unscaled_b + sqrt(unscaled_disc));
          }
        }
        if (tmpLambda > 0.5 * lambda) {
          tmpLambda = 0.5 * lambda;
        }
      }
      lambda2 = lambda;
      eScratch = newE;
      lambdaOut = max(tmpLambda, 0.1 * lambda);
    }
  }
  __syncthreads();
  return converged;
}

__device__ void setDirection(const int numTerms,
                             const double* posFromLineSearch,
                             const double* pos,
                             double* xi,
                             double* dGrad,
                             const double* grad,
                             bool& converged,
                             typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  double localMax_numerator = 0.0;
  double localMax_denominator = 1.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    xi[i] = posFromLineSearch[i] - pos[i];
    dGrad[i] = grad[i];
    
    double temp_numerator = fabs(xi[i]);
    double temp_denominator = fmax(fabs(posFromLineSearch[i]), 1.0);
    // temp_numerator / temp_denominator > localMax_numerator / localMax_denominator
    // <=>
    // temp_numerator * localMax_denominator > localMax_numerator * temp_denominator
    if (temp_numerator * localMax_denominator > localMax_numerator * temp_denominator) {
      localMax_numerator = temp_numerator;
      localMax_denominator = temp_denominator;
    }
  }
  
  double localMax = localMax_numerator / localMax_denominator;
  double blockMax = cub::BlockReduce<double, BLOCK_SIZE>(tempStorage).Reduce(localMax, cub::Max());
  
  if (threadIdx.x == 0 && blockMax < TOLX) {
    converged = true;
  }
  __syncthreads();
}

template <bool scaleGrads>
__device__ void scaleGrad(const int numTerms, double* grad, double& gradScale,
                          typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  gradScale = scaleGrads ? 0.1 : 1.0;
  
  double maxGrad = -1e8;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    if constexpr (scaleGrads) {
      grad[i] *= gradScale;
    }
    if (grad[i] > maxGrad) {
      maxGrad = grad[i];
    }
  }
  
  double blockMax = cub::BlockReduce<double, BLOCK_SIZE>(tempStorage).Reduce(maxGrad, cub::Max());
  
  __shared__ double distributedMax[1];
  if (threadIdx.x == 0) {
    distributedMax[0] = blockMax;
  }
  __syncthreads();
  
  maxGrad = distributedMax[0];
  
  if (scaleGrads && maxGrad > 10.0) {
    while (maxGrad * gradScale > 10.0) {
      gradScale *= 0.5;
    }
    for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
      grad[i] *= gradScale;
    }
  }
  __syncthreads();
}

__device__ void updateDGrad(const int numTerms,
                           const double gradTol,
                           const double energy,
                           const double gradScale,
                           const double* grad,
                           const double* pos,
                           double* dGrad,
                           bool& converged,
                           typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  double localMax = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    dGrad[i] = grad[i] - dGrad[i];
    double temp = fabs(grad[i]) * fmax(fabs(pos[i]), 1.0);
    if (temp > localMax) {
      localMax = temp;
    }
  }
  
  double blockMax = cub::BlockReduce<double, BLOCK_SIZE>(tempStorage).Reduce(localMax, cub::Max());
  
  if (threadIdx.x == 0) {
    const double term = max(energy * gradScale, 1.0);
    blockMax /= term;
    if (blockMax < gradTol) {
      converged = true;
    }
  }
  __syncthreads();
}

__device__ void updateInverseHessian(const int numTerms,
                                     double* invHessian,
                                     double* dGrad,
                                     double* xi,
                                     double* hessDGrad,
                                     double* grad,
                                     typename cub::BlockReduce<double, BLOCK_SIZE>::TempStorage& tempStorage) {
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
  
  // Compute hessDGrad = invHessian * dGrad
  for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
    double dotProduct = 0.0;
    for (int col = 0; col < numTerms; col++) {
      dotProduct += invHessian[row * numTerms + col] * dGrad[col];
    }
    hessDGrad[row] = dotProduct;
  }
  __syncthreads();
  
  // Compute BFGS sums
  __shared__ double fac, fae, fad, sumDGrad, sumXi;
  __shared__ bool needUpdate;
  
  double sumFac = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumFac += dGrad[i] * xi[i];
  }
  double facReduced = BlockReduce(tempStorage).Sum(sumFac);
  if (threadIdx.x == 0) fac = facReduced;
  __syncthreads();
  
  double sumFae = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumFae += dGrad[i] * hessDGrad[i];
  }
  double faeReduced = BlockReduce(tempStorage).Sum(sumFae);
  if (threadIdx.x == 0) fae = faeReduced;
  __syncthreads();
  
  double sumDGradSq = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumDGradSq += dGrad[i] * dGrad[i];
  }
  double sumDGradReduced = BlockReduce(tempStorage).Sum(sumDGradSq);
  if (threadIdx.x == 0) sumDGrad = sumDGradReduced;
  __syncthreads();
  
  double sumXiSq = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumXiSq += xi[i] * xi[i];
  }
  double sumXiReduced = BlockReduce(tempStorage).Sum(sumXiSq);
  if (threadIdx.x == 0) sumXi = sumXiReduced;
  __syncthreads();
  
  if (threadIdx.x == 0) {
    constexpr double EPS = 3e-8;
    needUpdate = fac > sqrt(EPS * sumDGrad * sumXi);
    
    if (needUpdate) {
      fac = 1.0 / fac;
      fad = 1.0 / fae;
    }
  }
  __syncthreads();
  
  if (needUpdate) {
    // Update dGrad for Hessian update
    for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
      dGrad[i] = fac * xi[i] - fad * hessDGrad[i];
    }
    __syncthreads();
    
    // Update inverse Hessian and compute new direction
    for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
      double pxi = fac * xi[row];
      double hdgi = fad * hessDGrad[row];
      double dgi = fae * dGrad[row];
      
      for (int col = 0; col < numTerms; col++) {
        double pxj = xi[col];
        double hdgj = hessDGrad[col];
        double dgj = dGrad[col];
        double update = pxi * pxj - hdgi * hdgj + dgi * dgj;
        invHessian[row * numTerms + col] += update;
      }
    }
    __syncthreads();
  }
  
  // Update xi = -invHessian * grad
  for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
    double dotProduct = 0.0;
    for (int col = 0; col < numTerms; col++) {
      dotProduct += invHessian[row * numTerms + col] * grad[col];
    }
    xi[row] = -dotProduct;
  }
  __syncthreads();
}


}  // namespace

template <int MaxAtoms, bool UseSharedMem>
__global__ __launch_bounds__(BLOCK_SIZE, 1) void bfgsMinimizeKernel(const int numIters,
                                   const double gradTol,
                                   const bool scaleGrads,
                                   const MMFF::EnergyForceContribsDevicePtr* terms,
                                   const MMFF::BatchedIndicesDevicePtr* systemIndices,
                                   const int* molIdList,
                                   const int* atomStarts,
                                   const int* hessianStarts,
                                   double* positions,
                                   double* grad,
                                   double* inverseHessian,
                                   double** scratchBuffers,
                                   double* energyOuts,
                                   const int DIM) {
  const int molIdx = molIdList[blockIdx.x];
  const int tid = threadIdx.x;
  const int stride = blockDim.x;
  
  const int atomStart = atomStarts[molIdx];
  const int atomEnd = atomStarts[molIdx + 1];
  const int numAtoms = atomEnd - atomStart;
  const int numTerms = DIM * numAtoms;
  
  constexpr int maxTerms = MaxAtoms * 3;
  
  // Pointers to working memory (either shared or global)
  double* localPos;
  double* localGrad;
  double* localDir;
  double* scratchPos;
  double* dGrad;
  double* oldPos;
  
  if constexpr (UseSharedMem) {
    // Shared memory for small molecules (≤64 atoms)
    // Note: oldPos moved to global memory to reduce shared memory pressure
    __shared__ double sharedLocalPos[maxTerms];
    __shared__ double sharedLocalGrad[maxTerms];
    __shared__ double sharedLocalDir[maxTerms];
    __shared__ double sharedScratchPos[maxTerms];
    __shared__ double sharedDGrad[maxTerms];
    
    const int termStart = atomStart * DIM;
    localPos = sharedLocalPos;
    localGrad = sharedLocalGrad;
    localDir = sharedLocalDir;
    scratchPos = sharedScratchPos;
    dGrad = sharedDGrad;
    // For small molecules, grad buffer is unused (using sharedLocalGrad), so reuse it for oldPos
    oldPos = scratchBuffers[0] + termStart;       // Reuse grad buffer for oldPos
  } else {
    // Global memory for large molecules (>64 atoms) - index into pre-allocated buffers
    const int termStart = atomStart * DIM;
    localPos = scratchBuffers[0] + termStart;    // Reuse grad buffer as scratch
    localGrad = grad + termStart;                 // Use main gradient buffer
    localDir = scratchBuffers[1] + termStart;     // lineSearchDir
    scratchPos = scratchBuffers[2] + termStart;   // scratchPositions
    dGrad = scratchBuffers[3] + termStart;        // hessDGrad
    oldPos = scratchBuffers[4] + termStart;       // scratchGrad (repurposed)
  }

  // Shared scalars
  __shared__ double maxStep;
  __shared__ double prevE;
  __shared__ double currE;
  __shared__ double slope;
  __shared__ double lambda;
  __shared__ double lambdaMin;
  __shared__ double lambda2;
  __shared__ double eScratch;
  __shared__ double gradScale;
  __shared__ bool converged;
  __shared__ bool lineSearchConverged;
  
  // Inverse Hessian in global memory (O(n^2), too large for shared)
  // Indexed by hessianStarts which stores cumulative (numTerms * numTerms) offsets
  double* invHessian = inverseHessian + hessianStarts[molIdx];
  
  // Initialize positions from global memory
  double* globalPos = positions + atomStart * DIM;
  for (int i = tid; i < numTerms; i += stride) {
    localPos[i] = globalPos[i];
  }
  __syncthreads();
  
  // Initialize inverse Hessian to identity
  const int hessianSize = numTerms * numTerms;
  for (int i = tid; i < hessianSize; i += stride) {
    const int row = i / numTerms;
    const int col = i % numTerms;
    invHessian[i] = (row == col) ? 1.0 : 0.0;
  }
  
  if (tid == 0) {
    converged = false;
  }
  __syncthreads();
  
  // Shared temp storage for all BlockReduce operations
  using BlockReduce = cub::BlockReduce<double, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  
  // Compute initial energy
  const double threadEnergy = MMFF::molEnergy(*terms, *systemIndices, positions, molIdx, tid, stride);
  const double blockEnergy = BlockReduce(tempStorage).Sum(threadEnergy);
  
  if (tid == 0) {
    prevE = blockEnergy;
    energyOuts[molIdx] = blockEnergy;
  }
  __syncthreads();
  
  // Compute initial gradient  
  for (int i = tid; i < numTerms; i += stride) {
    localGrad[i] = 0.0;
  }
  __syncthreads();
  
  MMFF::molGrad(*terms, *systemIndices, positions, localGrad, molIdx, tid, stride);
  __syncthreads();
  
  // Scale gradients
  if (scaleGrads) {
    scaleGrad<true>(numTerms, localGrad, gradScale, tempStorage);
  } else {
    scaleGrad<false>(numTerms, localGrad, gradScale, tempStorage);
  }
  
  // Set initial direction as negative gradient
  for (int i = tid; i < numTerms; i += stride) {
    localDir[i] = -localGrad[i];
  }
  __syncthreads();
  
  // Set max step
  setMaxStep(localPos, numTerms, &maxStep, tempStorage);
  __syncthreads();
  
  // Main BFGS loop
  __shared__ int currIter;
  if (tid == 0) {
    currIter = 0;
  }
  __syncthreads();
  
  while (!converged && currIter < numIters) {
    // Save current position before line search
    for (int i = tid; i < numTerms; i += stride) {
      oldPos[i] = localPos[i];
    }
    __syncthreads();
    
    // Line search setup
    if (tid == 0) {
      lineSearchConverged = false;
      lambda = 1.0;
    }
    __syncthreads();
    
    lineSearchSetup(numTerms, localPos, localGrad, maxStep, localDir, slope, lambdaMin, tempStorage);
    __syncthreads();
    
    // Line search loop
    __shared__ int lineSearchIter;
    if (tid == 0) {
      lineSearchIter = 0;
    }
    __syncthreads();
    
    while (!lineSearchConverged && lineSearchIter < MAX_LINESEARCH_ITERS) {
      // Perturb positions
      lineSearchPerturb(numTerms, localPos, localDir, lambda, scratchPos);
      
      // Copy to global for energy calculation
      for (int i = tid; i < numTerms; i += stride) {
        globalPos[i] = scratchPos[i];
      }
      __syncthreads();
      
      // Compute energy at perturbed position
      const double lsThreadEnergy = MMFF::molEnergy(*terms, *systemIndices, positions, molIdx, tid, stride);
      const double lsBlockEnergy = BlockReduce(tempStorage).Sum(lsThreadEnergy);
      
      if (tid == 0) {
        currE = lsBlockEnergy;
      }
      __syncthreads();
      
      // Check convergence and update lambda
      lineSearchConverged = lineSearchPostEnergy(lineSearchIter == 0, prevE, currE, slope, lambda, lambdaMin, lambda2, eScratch, lambda);
      __syncthreads();
      
      if (tid == 0) {
        lineSearchIter++;
      }
      __syncthreads();
    }
    
    // Update positions with final line search result and compute direction
    for (int i = tid; i < numTerms; i += stride) {
      localPos[i] = scratchPos[i];
      globalPos[i] = scratchPos[i];
    }
    __syncthreads();
    
    // Set direction (compute xi = new - old)
    setDirection(numTerms, scratchPos, oldPos, localDir, dGrad, localGrad, converged, tempStorage);
    if (converged) break;
    
    // Update stored energy for next iteration
    if (tid == 0) {
      prevE = currE;
    }
    __syncthreads();
    
    // Compute gradients at new position
    for (int i = tid; i < numTerms; i += stride) {
      localGrad[i] = 0.0;
    }
    __syncthreads();
    
    MMFF::molGrad(*terms, *systemIndices, positions, localGrad, molIdx, tid, stride);
    __syncthreads();
    
    // Scale gradients
    if (scaleGrads) {
      scaleGrad<true>(numTerms, localGrad, gradScale, tempStorage);
    } else {
      scaleGrad<false>(numTerms, localGrad, gradScale, tempStorage);
    }
    
    // Update dGrad and check convergence
    updateDGrad(numTerms, gradTol, currE, gradScale, localGrad, localPos, dGrad, converged, tempStorage);
    if (converged) break;
    
    // Update Hessian and compute new direction (reuses scratchPos as hessDGrad)
    updateInverseHessian(numTerms, invHessian, dGrad, localDir, scratchPos, localGrad, tempStorage);
    
    if (tid == 0) {
      currIter++;
    }
    __syncthreads();
  }
  
  // Write final energy
  if (tid == 0) {
    energyOuts[molIdx] = prevE;
  }
}

namespace {

template <int MaxAtoms, bool UseSharedMem>
cudaError_t launchBinnedKernel(int numMolsInBin,
                                const int* molIdList,
                                int numIters,
                                double gradTol,
                                bool scaleGrads,
                                const MMFF::EnergyForceContribsDevicePtr* devTerms,
                                const MMFF::BatchedIndicesDevicePtr* devSysIdx,
                                const int* atomStarts,
                                const int* hessianStarts,
                                double* positions,
                                double* grad,
                                double* inverseHessian,
                                double** scratchBuffers,
                                double* energyOuts,
                                int dataDim,
                                cudaStream_t stream) {
  if (numMolsInBin == 0) {
    return cudaSuccess;
  }
  
  bfgsMinimizeKernel<MaxAtoms, UseSharedMem><<<numMolsInBin, BLOCK_SIZE, 0, stream>>>(
    numIters,
    gradTol,
    scaleGrads,
    devTerms,
    devSysIdx,
    molIdList,
    atomStarts,
    hessianStarts,
    positions,
    grad,
    inverseHessian,
    scratchBuffers,
    energyOuts,
    dataDim);
  
  return cudaGetLastError();
}

}  // namespace

cudaError_t launchBfgsMinimizePerMolKernel(const int* binCounts,
                                           const int** binMolIds,
                                           const int* atomStarts,
                                           const int* hessianStarts,
                                           int numIters,
                                           double gradTol,
                                           bool scaleGrads,
                                           const MMFF::EnergyForceContribsDevicePtr& terms,
                                           const MMFF::BatchedIndicesDevicePtr& systemIndices,
                                           double* positions,
                                           double* grad,
                                           double* inverseHessian,
                                           double** scratchBuffers,
                                           double* energyOuts,
                                           int dataDim,
                                           cudaStream_t stream) {
  // Prepare device pointers for terms and indices
  const AsyncDevicePtr<MMFF::EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<MMFF::BatchedIndicesDevicePtr> devSysIdx(systemIndices, stream);
  
  cudaError_t err = cudaSuccess;
  // TODO: Run these concurrently, large to small?
  
  // Launch kernels for each size bin
  // Bin 0: 32 atoms, use shared memory
  err = launchBinnedKernel<32, true>(
    binCounts[0], binMolIds[0], numIters, gradTol, scaleGrads,
    devTerms.data(), devSysIdx.data(), atomStarts, hessianStarts,
    positions, grad, inverseHessian, scratchBuffers, energyOuts, dataDim, stream);
  if (err != cudaSuccess) return err;
  
  // Bin 1: 64 atoms, use shared memory
  err = launchBinnedKernel<64, true>(
    binCounts[1], binMolIds[1], numIters, gradTol, scaleGrads,
    devTerms.data(), devSysIdx.data(), atomStarts, hessianStarts,
    positions, grad, inverseHessian, scratchBuffers, energyOuts, dataDim, stream);
  if (err != cudaSuccess) return err;
  
  // Bin 2: 128 atoms, use global memory
  err = launchBinnedKernel<128, false>(
    binCounts[2], binMolIds[2], numIters, gradTol, scaleGrads,
    devTerms.data(), devSysIdx.data(), atomStarts, hessianStarts,
    positions, grad, inverseHessian, scratchBuffers, energyOuts, dataDim, stream);
  if (err != cudaSuccess) return err;
  
  // Bin 3: 256 atoms, use global memory
  err = launchBinnedKernel<256, false>(
    binCounts[3], binMolIds[3], numIters, gradTol, scaleGrads,
    devTerms.data(), devSysIdx.data(), atomStarts, hessianStarts,
    positions, grad, inverseHessian, scratchBuffers, energyOuts, dataDim, stream);
  if (err != cudaSuccess) return err;
  
  // Bin 4: 2048 atoms, use global memory
  err = launchBinnedKernel<2048, false>(
    binCounts[4], binMolIds[4], numIters, gradTol, scaleGrads,
    devTerms.data(), devSysIdx.data(), atomStarts, hessianStarts,
    positions, grad, inverseHessian, scratchBuffers, energyOuts, dataDim, stream);
  
  return err;
}

}  // namespace nvMolKit