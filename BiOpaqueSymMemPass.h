#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

struct BiOpaqueSymMemPass : llvm::PassInfoMixin<BiOpaqueSymMemPass>
{
    llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager&);
};

bool RunBiOpaqueSymMem(llvm::Module& m);

extern llvm::cl::opt<bool> BiOpaqueSymMemEnable;