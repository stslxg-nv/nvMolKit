// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "bfgs_distgeom.h"

#include "bfgs_minimize.h"
#include "bfgs_minimize_permol_kernels.h"
#include "dist_geom.h"

namespace nvMolKit::DistGeom {

void DistGeomMinimizeBFGS(BatchedMolecularSystemHost&    molSystemHost,
                          BatchedMolecularDeviceBuffers& molSystemDevice,
                          detail::ETKDGContext&          context,
                          double                         chiralWeight,
                          double                         fourthDimWeight,
                          const int                      maxIters,
                          const double                   gradTol,
                          const bool                     repeatUntilConverged,
                          cudaStream_t                   stream) {
  // Setup device buffers
  setupDeviceBuffers(molSystemHost,
                     molSystemDevice,
                     context.systemHost.positions,
                     static_cast<int>(context.systemHost.atomStarts.size() - 1));

  const size_t numAtoms = context.systemHost.atomStarts.back();
  const size_t numPos   = context.systemHost.positions.size();
  const int    dim      = (numPos == numAtoms * 3) ? 3 : 4;

  // Create energy and gradient functions
  auto eFunc = [&](const double* positions) {
    computeEnergy(molSystemDevice,
                  context.systemDevice.atomStarts,
                  context.systemDevice.positions,
                  context.activeThisStage.data(),
                  positions,
                  stream,
                  chiralWeight,
                  fourthDimWeight);
  };

  auto gFunc = [&]() {
    computeGradients(molSystemDevice,
                     context.systemDevice.atomStarts,
                     context.systemDevice.positions,
                     context.activeThisStage.data(),
                     stream,
                     chiralWeight,
                     fourthDimWeight);
  };

  // Create and configure BFGS minimizer
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(/*dataDim=*/dim, nvMolKit::DebugLevel::NONE, true, stream);

  // Run minimization
  bool needsMore = bfgsMinimizer.minimize(maxIters,
                                          gradTol,
                                          context.systemHost.atomStarts,
                                          context.systemDevice.atomStarts,
                                          context.systemDevice.positions,
                                          molSystemDevice.grad,
                                          molSystemDevice.energyOuts,
                                          molSystemDevice.energyBuffer,
                                          eFunc,
                                          gFunc,
                                          context.activeThisStage.data());
  while (needsMore && repeatUntilConverged) {
    needsMore = bfgsMinimizer.minimize(maxIters,
                                       gradTol,
                                       context.systemHost.atomStarts,
                                       context.systemDevice.atomStarts,
                                       context.systemDevice.positions,
                                       molSystemDevice.grad,
                                       molSystemDevice.energyOuts,
                                       molSystemDevice.energyBuffer,
                                       eFunc,
                                       gFunc,
                                       context.activeThisStage.data());
  }
}

void DistGeomMinimizeBFGSPerMol(BatchedMolecularSystemHost&    molSystemHost,
                                BatchedMolecularDeviceBuffers& molSystemDevice,
                                detail::ETKDGContext&          context,
                                double                         chiralWeight,
                                double                         fourthDimWeight,
                                const int                      maxIters,
                                const double                   gradTol,
                                const bool                     repeatUntilConverged,
                                cudaStream_t                   stream) {
  //printf("\nCall DistGeomMinimizeBFGSPerMol\n\n");
  // Setup device buffers
  setupDeviceBuffers(molSystemHost,
                     molSystemDevice,
                     context.systemHost.positions,
                     static_cast<int>(context.systemHost.atomStarts.size() - 1));

  const size_t numAtoms = context.systemHost.atomStarts.back();
  const size_t numPos   = context.systemHost.positions.size();
  const int    dim      = (numPos == numAtoms * 3) ? 3 : 4;

  // Create BFGS minimizer
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(dim, nvMolKit::DebugLevel::NONE, true, stream, 
                                               nvMolKit::BfgsBackend::PER_MOLECULE);

  // Allocate energy buffer (not used for DG but required by interface)
  nvMolKit::AsyncDeviceVector<double> energyBuffer(0, stream);

  // Run minimization (with optional repeat-until-converged)
  bool needsMore = bfgsMinimizer.minimizeWithDG(maxIters,
                                                 gradTol,
                                                 context.systemHost.atomStarts,
                                                 context.systemDevice.atomStarts,
                                                 context.systemDevice.positions,
                                                 molSystemDevice.grad,
                                                 molSystemDevice.energyOuts,
                                                 toEnergyForceContribsDevicePtr(molSystemDevice),
                                                 toBatchedIndicesDevicePtr(molSystemDevice, context.systemDevice.atomStarts.data()),
                                                 chiralWeight,
                                                 fourthDimWeight,
                                                 context.activeThisStage.data());
  //printf("Needs more ? %d\n", needsMore);

  while (needsMore && repeatUntilConverged) {
    //printf("Repeating DG minimization\n");
    needsMore = bfgsMinimizer.minimizeWithDG(maxIters,
                                              gradTol,
                                              context.systemHost.atomStarts,
                                              context.systemDevice.atomStarts,
                                              context.systemDevice.positions,
                                              molSystemDevice.grad,
                                              molSystemDevice.energyOuts,
                                              toEnergyForceContribsDevicePtr(molSystemDevice),
                                              toBatchedIndicesDevicePtr(molSystemDevice, context.systemDevice.atomStarts.data()),
                                              chiralWeight,
                                              fourthDimWeight,
                                              context.activeThisStage.data());
  }
}

void ETKMinimizeBFGSPerMol(BatchedMolecularSystem3DHost&    molSystemHost,
                           BatchedMolecular3DDeviceBuffers& molSystemDevice,
                           detail::ETKDGContext&            context,
                           const int                        maxIters,
                           const double                     gradTol,
                           const bool                       repeatUntilConverged,
                           cudaStream_t                     stream) {
  // Setup device buffers
  setupDeviceBuffers3D(molSystemHost,
                       molSystemDevice,
                       context.systemHost.positions,
                       static_cast<int>(context.systemHost.atomStarts.size() - 1));

  // Create BFGS minimizer (3D for ETK)
  nvMolKit::BfgsBatchMinimizer bfgsMinimizer(3, nvMolKit::DebugLevel::NONE, true, stream,
                                               nvMolKit::BfgsBackend::PER_MOLECULE);

  // Allocate energy buffer (not used for ETK but required by interface)
  nvMolKit::AsyncDeviceVector<double> energyBuffer(0, stream);

  // Run minimization (with optional repeat-until-converged)
  bool needsMore = bfgsMinimizer.minimizeWithETK(maxIters,
                                                  gradTol,
                                                  context.systemHost.atomStarts,
                                                  context.systemDevice.atomStarts,
                                                  context.systemDevice.positions,
                                                  molSystemDevice.grad,
                                                  molSystemDevice.energyOuts,
                                                  toEnergy3DForceContribsDevicePtr(molSystemDevice),
                                                  toBatchedIndices3DDevicePtr(molSystemDevice, context.systemDevice.atomStarts.data()),
                                                  context.activeThisStage.data());
  
  while (needsMore && repeatUntilConverged) {
    needsMore = bfgsMinimizer.minimizeWithETK(maxIters,
                                               gradTol,
                                               context.systemHost.atomStarts,
                                               context.systemDevice.atomStarts,
                                               context.systemDevice.positions,
                                               molSystemDevice.grad,
                                               molSystemDevice.energyOuts,
                                               toEnergy3DForceContribsDevicePtr(molSystemDevice),
                                               toBatchedIndices3DDevicePtr(molSystemDevice, context.systemDevice.atomStarts.data()),
                                               context.activeThisStage.data());
  }
}

}  // namespace nvMolKit::DistGeom
