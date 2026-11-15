#include "BiOpaqueSymMemPass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace llvm;

llvm::cl::opt<bool> BiOpaqueSymMemEnable("obfbosmopq", llvm::cl::init(true), llvm::cl::desc("BiOpaque with symbolic memory indirection."));

static constexpr unsigned PoolBitSetCount = 16u;
static constexpr unsigned PoolMaskedEqCount = 16u;
static constexpr unsigned PoolCompoundPairCount = 8u;

static constexpr unsigned TableSizes[5] = { 128u, 192u, 256u, 384u, 512u };

static constexpr unsigned Density = 100u;
static constexpr unsigned PerFunctionCap = 20u;
static constexpr unsigned MinBBInsts = 2u;

static constexpr bool SkipLoops = false;
static constexpr bool SkipEntry = false;

static constexpr unsigned HarvestMin = 2u;
static constexpr unsigned HarvestMax = 3u;
static constexpr unsigned HarvestScanCap = 128u;

static constexpr unsigned WeightClassic = 30u;
static constexpr unsigned WeightDiamond = 30u;
static constexpr unsigned WeightCompound = 20u;
static constexpr unsigned WeightLoop = 20u;
static_assert(WeightClassic + WeightDiamond + WeightCompound + WeightLoop == 100u, "mode weights must sum to 100");

static constexpr unsigned DiamondMinSuffix = 2u;

static constexpr unsigned LoopMinN = 7u;
static constexpr unsigned LoopMaxN = 11u;

static constexpr unsigned DecoySinkPoolSize = 16u;

static constexpr unsigned ClassicDecoyOpsMin = 5u;
static constexpr unsigned ClassicDecoyOpsRange = 8u;
static constexpr unsigned DiamondDecoyOpsMin = 2u;
static constexpr unsigned DiamondDecoyOpsRange = 4u;
static constexpr unsigned IntermediateStoreMax = 3u;
static constexpr unsigned IntermediateStorePctLoad = 25u;
static constexpr unsigned ConstPerturbPct = 50u;
static constexpr unsigned ClassicDiamondCloneCap = 8u;
static constexpr unsigned StealCeiling = 12u;
static constexpr unsigned StealFloor = 3u;
static constexpr unsigned RealBBMinResidue = 5u;

static constexpr uint64_t MultiplierPool[16] = {
    0x9E3779B97F4A7C15ULL, 0xBF58476D1CE4E5B9ULL, 0x94D049BB133111EBULL,
    0xC2B2AE3D27D4EB4FULL, 0x165667B19E3779F9ULL, 0x85EBCA77C2B2AE63ULL,
    0xDEADBEEFCAFEBABDULL, 0xA4093822299F31D1ULL, 0x8CB92BA72F3D8DD7ULL,
    0xB492B66FBE98F273ULL, 0xD9B4BEF9FB2F3B2DULL, 0xFF51AFD7ED558CCDULL,
    0xC6A4A7935BD1E995ULL, 0x9F79D799CA51B91BULL, 0x7A1CBE68A1B5C9D3ULL,
    0xE7037ED1A0B428DBULL
};

static bool HasAnnotationTag(Function& f, const char* tag)
{
    Module* m = f.getParent();
    GlobalVariable* ga = m->getGlobalVariable("llvm.global.annotations");
    if (!ga || !ga->hasInitializer()) return false;
    auto* ca = dyn_cast<ConstantArray>(ga->getOperand(0));
    if (!ca) return false;
    for (unsigned i = 0; i < ca->getNumOperands(); ++i)
    {
        auto* cs = dyn_cast<ConstantStruct>(ca->getOperand(i));
        if (!cs || cs->getNumOperands() < 2) continue;
        if (cs->getOperand(0)->stripPointerCasts() != &f) continue;
        auto* annogv = dyn_cast<GlobalVariable>(cs->getOperand(1)->stripPointerCasts());
        if (!annogv || !annogv->hasInitializer()) continue;
        auto* cds = dyn_cast<ConstantDataSequential>(annogv->getInitializer());
        if (!cds || !cds->isString()) continue;
        if (strcmp(cds->getAsString().str().c_str(), tag) == 0) return true;
    }
    return false;
}

static bool HasObfuscateAnnotation(Function& f) { return HasAnnotationTag(f, "obfus"); }
static bool HasNoObfuscateAnnotation(Function& f) { return HasAnnotationTag(f, "noobfus"); }
static bool HasNoBosmopqAnnotation(Function& f) { return HasAnnotationTag(f, "nobosmopq"); }
static bool HasNoOpaqueAnnotation(Function& f) { return HasAnnotationTag(f, "noopaque"); }

enum class InvKind : uint8_t
{
    BitSet,
    MaskedEq,
    CompoundPair,
};

struct SymMemTable
{
    GlobalVariable* GV;
    InvKind Kind;
    uint32_t Mask;
    uint32_t Const;
    unsigned Size;
    uint8_t XorKey;
};

struct SymMemPair
{
    SymMemTable T1;
    SymMemTable T2;
    uint32_t Mask;
    uint32_t InvXorConst;
    unsigned Size;
};

static std::vector<SymMemTable> Tables;
static std::vector<SymMemPair> Pairs;
static Module* TablesModule = nullptr;
static std::mt19937_64 Rng;

static GlobalVariable* NoiseSlot = nullptr;
static GlobalVariable* DecoySinks[DecoySinkPoolSize] = { nullptr };

static Function* TableDecryptor = nullptr;

static Module* LoopLimitModule = nullptr;
static unsigned LoopLimitNext = 0;

static GlobalVariable* EnsureNoiseSlot(Module& m)
{
    if (NoiseSlot && NoiseSlot->getParent() == &m) return NoiseSlot;
    LLVMContext& c = m.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);
    NoiseSlot = new GlobalVariable(m, i64, false, GlobalValue::InternalLinkage, ConstantInt::get(i64, 0), ".BosmopqNoise");
    NoiseSlot->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    NoiseSlot->setAlignment(Align(8));
    return NoiseSlot;
}

static GlobalVariable* EnsureDecoySink(Module& m, unsigned idx)
{
    const unsigned i = idx % DecoySinkPoolSize;
    if (DecoySinks[i] && DecoySinks[i]->getParent() == &m) return DecoySinks[i];
    LLVMContext& c = m.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);
    const std::string name = ".BosmopqSink." + std::to_string(i);
    DecoySinks[i] = new GlobalVariable(m, i64, false, GlobalValue::InternalLinkage, ConstantInt::get(i64, 0), name);
    DecoySinks[i]->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    DecoySinks[i]->setAlignment(Align(8));
    return DecoySinks[i];
}

static GlobalVariable* CreateLoopLimit(Module& m)
{
    if (LoopLimitModule != &m)
    {
        LoopLimitModule = &m;
        LoopLimitNext = 0;
    }
    LLVMContext& c = m.getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    const uint32_t limit = LoopMinN + (uint32_t)(Rng() % (LoopMaxN - LoopMinN + 1u));
    const std::string name = ".BosmopqLL." + std::to_string(LoopLimitNext++);
    auto* gv = new GlobalVariable(m, i32, false, GlobalValue::InternalLinkage, ConstantInt::get(i32, limit), name);
    gv->setUnnamedAddr(GlobalValue::UnnamedAddr::None);
    gv->setAlignment(Align(4));
    return gv;
}

// tables are XOR encrypted on disk with a per table 8 bit key, placed in .data (not .rdata), and marked none constant so the module ctor can decrypt in place at startup; invariant holds on PLAIN bytes, not disk bytes, so BitSet/MaskedEq invariant checks MUST see post ctor bytes. A site that reads a table before the ctor runs sees encrypted data and the compare routes into pollute
static SymMemTable BuildBitSetTable(Module& m, unsigned poolidx, unsigned size)
{
    LLVMContext& c = m.getContext();
    IntegerType* i8 = Type::getInt8Ty(c);
    ArrayType* arrty = ArrayType::get(i8, size);

    const unsigned bitpos = (unsigned)(Rng() % 8u);
    const uint32_t mask = 1u << bitpos;
    const uint32_t cst = mask;
    const uint8_t key = (uint8_t)(Rng() & 0xFFu);

    SmallVector<Constant*, 512> bytes;
    bytes.reserve(size);
    for (unsigned i = 0; i < size; ++i)
    {
        const uint8_t plain = (uint8_t)(Rng() & 0xFFu) | (uint8_t)mask;
        const uint8_t encrypted = plain ^ key;
        bytes.push_back(ConstantInt::get(i8, encrypted));
    }
    Constant* init = ConstantArray::get(arrty, bytes);
    const std::string name = ".BosmopqT." + std::to_string(poolidx);
    auto* gv = new GlobalVariable(m, arrty, false, GlobalValue::PrivateLinkage, init, name);
    gv->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    gv->setAlignment(Align(1));
    gv->setSection(".data");
    return { gv, InvKind::BitSet, mask, cst, size, key };
}

static SymMemTable BuildMaskedEqTable(Module& m, unsigned poolidx, unsigned size)
{
    LLVMContext& c = m.getContext();
    IntegerType* i8 = Type::getInt8Ty(c);
    ArrayType* arrty = ArrayType::get(i8, size);

    const unsigned popcount = 2u + (unsigned)(Rng() % 3u);
    uint32_t mask = 0;
    unsigned have = 0;
    while (have < popcount)
    {
        const unsigned bit = (unsigned)(Rng() % 8u);
        if ((mask & (1u << bit)) == 0) { mask |= (1u << bit); ++have; }
    }
    uint32_t cst = 0;
    for (unsigned b = 0; b < 8; ++b)
    {
        if ((mask & (1u << b)) && (Rng() & 1u)) cst |= (1u << b);
    }
    const uint8_t key = (uint8_t)(Rng() & 0xFFu);

    SmallVector<Constant*, 512> bytes;
    bytes.reserve(size);
    for (unsigned i = 0; i < size; ++i)
    {
        const uint8_t plain = (uint8_t)((Rng() & ~(uint64_t)mask) | (uint64_t)cst) & 0xFFu;
        const uint8_t encrypted = plain ^ key;
        bytes.push_back(ConstantInt::get(i8, encrypted));
    }
    Constant* init = ConstantArray::get(arrty, bytes);
    const std::string name = ".BosmopqT." + std::to_string(poolidx);
    auto* gv = new GlobalVariable(m, arrty, false, GlobalValue::PrivateLinkage, init, name);
    gv->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
    gv->setAlignment(Align(1));
    gv->setSection(".data");
    return { gv, InvKind::MaskedEq, mask, cst, size, key };
}

// C1 and C2 MUST differ so the invariant C1^C2 is none trivially none zero; equal constants would give the compound compare the trivial RHS 0 and let an SMT solver recognise the "always compares to zero" shape as a red flag, so if the random draws collide twice force one bit to differ
static SymMemPair BuildCompoundPair(Module& m, unsigned& nextpoolidx, unsigned size)
{
    LLVMContext& c = m.getContext();
    IntegerType* i8 = Type::getInt8Ty(c);
    ArrayType* arrty = ArrayType::get(i8, size);

    const unsigned popcount = 2u + (unsigned)(Rng() % 3u);
    uint32_t mask = 0;
    unsigned have = 0;
    while (have < popcount)
    {
        const unsigned bit = (unsigned)(Rng() % 8u);
        if ((mask & (1u << bit)) == 0) { mask |= (1u << bit); ++have; }
    }

    auto picksubset = [&]() -> uint32_t
    {
        uint32_t cv = 0;
        for (unsigned b = 0; b < 8; ++b)
        {
            if ((mask & (1u << b)) && (Rng() & 1u)) cv |= (1u << b);
        }
        return cv;
    };
    uint32_t c1 = picksubset();
    uint32_t c2 = picksubset();
    for (int tries = 0; tries < 8 && c1 == c2; ++tries) c2 = picksubset();
    if (c1 == c2)
    {
        for (unsigned b = 0; b < 8; ++b)
        {
            if (mask & (1u << b)) { c2 ^= (1u << b); break; }
        }
    }

    auto buildone = [&](uint32_t cconst) -> SymMemTable
    {
        const uint8_t key = (uint8_t)(Rng() & 0xFFu);
        SmallVector<Constant*, 512> bytes;
        bytes.reserve(size);
        for (unsigned i = 0; i < size; ++i)
        {
            const uint8_t plain = (uint8_t)((Rng() & ~(uint64_t)mask) | (uint64_t)cconst) & 0xFFu;
            const uint8_t encrypted = plain ^ key;
            bytes.push_back(ConstantInt::get(i8, encrypted));
        }
        Constant* init = ConstantArray::get(arrty, bytes);
        const std::string name = ".BosmopqT." + std::to_string(nextpoolidx++);
        auto* gv = new GlobalVariable(m, arrty, false, GlobalValue::PrivateLinkage, init, name);
        gv->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
        gv->setAlignment(Align(1));
        gv->setSection(".data");
        return { gv, InvKind::CompoundPair, mask, cconst, size, key };
    };

    SymMemTable t1 = buildone(c1);
    SymMemTable t2 = buildone(c2);
    SymMemPair p;
    p.T1 = t1;
    p.T2 = t2;
    p.Mask = mask;
    p.InvXorConst = c1 ^ c2;
    p.Size = size;
    return p;
}

static void EmitTableDecryptLoop(IRBuilder<>& b, const SymMemTable& t)
{
    LLVMContext& c = b.getContext();
    IntegerType* i8 = Type::getInt8Ty(c);
    IntegerType* i32 = Type::getInt32Ty(c);
    Function* parent = b.GetInsertBlock()->getParent();

    BasicBlock* prebb = b.GetInsertBlock();
    BasicBlock* loopbb = BasicBlock::Create(c, "", parent);
    BasicBlock* exitbb = BasicBlock::Create(c, "", parent);

    b.CreateBr(loopbb);

    IRBuilder<> lb(loopbb);
    PHINode* iphi = lb.CreatePHI(i32, 2);
    iphi->addIncoming(ConstantInt::get(i32, 0), prebb);

    Value* p = lb.CreateInBoundsGEP(t.GV->getValueType(), t.GV, { ConstantInt::get(i32, 0), iphi });
    LoadInst* ld = lb.CreateLoad(i8, p);
    ld->setAlignment(Align(1));
    Value* decrypted = lb.CreateXor(ld, ConstantInt::get(i8, t.XorKey));
    StoreInst* st = lb.CreateStore(decrypted, p);
    st->setAlignment(Align(1));

    Value* inext = lb.CreateAdd(iphi, ConstantInt::get(i32, 1));
    Value* done = lb.CreateICmpUGE(inext, ConstantInt::get(i32, t.Size));
    lb.CreateCondBr(done, exitbb, loopbb);
    iphi->addIncoming(inext, loopbb);

    b.SetInsertPoint(exitbb);
}

// ctor priority 103 sits after other low priority ctors but doesn't fight with the C runtime (which uses priority ~65535 for user ctors); every protected function MUST wait for this ctor so site code sees plain bytes rather than encrypted ones
static void EmitTableDecryptor(Module& m)
{
    if (Tables.empty() && Pairs.empty()) return;

    LLVMContext& c = m.getContext();
    FunctionType* ft = FunctionType::get(Type::getVoidTy(c), false);
    TableDecryptor = Function::Create(ft, GlobalValue::InternalLinkage, "BosmopqTableDecrypt", &m);
    TableDecryptor->addFnAttr(Attribute::NoInline);

    BasicBlock* entry = BasicBlock::Create(c, "Entry", TableDecryptor);
    IRBuilder<> b(entry);

    for (const SymMemTable& t : Tables) EmitTableDecryptLoop(b, t);
    for (const SymMemPair& p : Pairs)
    {
        EmitTableDecryptLoop(b, p.T1);
        EmitTableDecryptLoop(b, p.T2);
    }

    b.CreateRetVoid();
    appendToGlobalCtors(m, TableDecryptor, 103);
}

static void EnsureTablePool(Module& m)
{
    if (TablesModule == &m && !Tables.empty()) return;
    Tables.clear();
    Pairs.clear();
    TablesModule = &m;

    unsigned idx = 0;
    for (unsigned i = 0; i < PoolBitSetCount; ++i)
    {
        const unsigned sz = TableSizes[Rng() % 5u];
        Tables.push_back(BuildBitSetTable(m, idx++, sz));
    }
    for (unsigned i = 0; i < PoolMaskedEqCount; ++i)
    {
        const unsigned sz = TableSizes[Rng() % 5u];
        Tables.push_back(BuildMaskedEqTable(m, idx++, sz));
    }
    for (unsigned i = 0; i < PoolCompoundPairCount; ++i)
    {
        const unsigned sz = TableSizes[Rng() % 5u];
        Pairs.push_back(BuildCompoundPair(m, idx, sz));
    }

    EmitTableDecryptor(m);
}

static bool BBAlreadyProcessed(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return true;
    return term->getMetadata("bosmopq.done") != nullptr;
}

static void MarkBBProcessed(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return;
    term->setMetadata("bosmopq.done", MDNode::get(bb->getContext(), {}));
}

static bool IsDecoyBlock(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return false;
    return term->getMetadata("bosmopq.decoy") != nullptr;
}

static void MarkDecoyBlock(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return;
    term->setMetadata("bosmopq.decoy", MDNode::get(bb->getContext(), {}));
}

// real_bb MUST be tagged on its existing terminator after each site emit so the chain into real worklist can locate it via FindRealBBAfterEmit regardless of mode. RealBBFromSplit style walks of bb's conditional-br successors break for loop sites where bb's terminator is uncond_br to loopHdr; the tag disappears naturally when real_bb is chained into (its terminator gets replaced), which is correct
static void MarkRealBB(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return;
    term->setMetadata("bosmopq.real", MDNode::get(bb->getContext(), {}));
}

static bool IsRealBB(BasicBlock* bb)
{
    Instruction* term = bb->getTerminator();
    if (!term) return false;
    return term->getMetadata("bosmopq.real") != nullptr;
}

static bool IsEHOrFuncletInterior(BasicBlock* bb)
{
    if (bb->isEHPad()) return true;
    Instruction* term = bb->getTerminator();
    if (!term) return true;
    if (isa<CatchReturnInst>(term) || isa<CleanupReturnInst>(term) || isa<ResumeInst>(term)) return true;
    for (Instruction& i : *bb)
    {
        if (auto* cb = dyn_cast<CallBase>(&i))
        {
            if (cb->countOperandBundlesOfType(LLVMContext::OB_funclet) > 0) return true;
        }
    }
    return false;
}

static Instruction* PickInsertionPoint(BasicBlock* bb)
{
    Instruction* first = bb->getFirstNonPHIOrDbgOrLifetime();
    if (!first) return nullptr;
    if ((unsigned)bb->size() < MinBBInsts) return nullptr;

    // splitting the entry block before an alloca relocates it into realbb and demotes it from static to dynamic, which breaks AllocaInst::isStaticAlloca downstream; clamp `first` past the last entry alloca
    if (bb == &bb->getParent()->getEntryBlock())
    {
        Instruction* lastalloca = nullptr;
        for (Instruction& i : *bb)
        {
            if (isa<AllocaInst>(&i)) lastalloca = &i;
        }
        if (lastalloca)
        {
            Instruction* after = lastalloca->getNextNode();
            if (!after) return nullptr;
            BasicBlock::iterator seek(first);
            while (seek != bb->end() && &*seek != after) ++seek;
            if (seek != bb->end()) first = &*seek;
            else first = after;
        }
    }

    unsigned splittable = 0;
    for (auto it = BasicBlock::iterator(first); it != bb->end(); ++it)
    {
        if (&*it == bb->getTerminator()) break;
        ++splittable;
    }
    if (splittable == 0) return nullptr;

    const unsigned pick = (unsigned)(Rng() % splittable);
    BasicBlock::iterator it(first);
    for (unsigned i = 0; i < pick; ++i) ++it;

    if (it == bb->end()) return nullptr;
    if (&*it == bb->getTerminator()) return nullptr;
    if (isa<PHINode>(&*it)) return nullptr;
    return &*it;
}

struct HarvestSlot
{
    Value* V;
    bool Is64;
};

static void CollectHarvestCandidates(Function& f, DominatorTree& dt, Instruction* insertbefore, SmallVectorImpl<HarvestSlot>& out)
{
    for (Argument& a : f.args())
    {
        Type* t = a.getType();
        if (t->isIntegerTy(32)) out.push_back({ &a, false });
        else if (t->isIntegerTy(64)) out.push_back({ &a, true });
    }
    unsigned scanned = 0;
    for (BasicBlock& bb : f)
    {
        for (Instruction& i : bb)
        {
            if (scanned++ > HarvestScanCap) break;
            if (&i == insertbefore) continue;
            if (i.getType()->isVoidTy()) continue;
            if (!dt.dominates(&i, insertbefore)) continue;
            Type* t = i.getType();
            if (t->isIntegerTy(32)) out.push_back({ &i, false });
            else if (t->isIntegerTy(64)) out.push_back({ &i, true });
        }
        if (scanned > HarvestScanCap) break;
    }
}

// candidate shuffle is required so successive BuildTableIndex calls in compound mode pick different harvest values; without it both index chains correlate perfectly and the attacker can factor the compound site back to a single index shape
static Value* BuildTableIndex(IRBuilder<>& b, Function& f, DominatorTree& dt, Instruction* insertbefore, Module& m, unsigned tablesize)
{
    LLVMContext& c = b.getContext();
    IntegerType* i32 = Type::getInt32Ty(c);
    IntegerType* i64 = Type::getInt64Ty(c);

    SmallVector<HarvestSlot, 16> candidates;
    CollectHarvestCandidates(f, dt, insertbefore, candidates);

    for (size_t i = candidates.size(); i > 1; --i)
    {
        const size_t j = (size_t)(Rng() % i);
        std::swap(candidates[i - 1], candidates[j]);
    }

    const unsigned want = HarvestMin + (unsigned)(Rng() % (HarvestMax - HarvestMin + 1u));
    SmallVector<Value*, 4> picked;
    for (unsigned i = 0; i < want && !candidates.empty(); ++i)
    {
        const unsigned idx = (unsigned)(Rng() % candidates.size());
        HarvestSlot& h = candidates[idx];
        Value* v = h.V;
        if (!h.Is64) v = b.CreateZExt(v, i64, "BosmopqHz");
        picked.push_back(v);
    }

    Value* mix = nullptr;
    if (picked.empty())
    {
        GlobalVariable* noise = EnsureNoiseSlot(m);
        LoadInst* ld = b.CreateLoad(i64, noise, true, "BosmopqNoise");
        ld->setAlignment(Align(8));
        mix = ld;
    }
    else
    {
        mix = picked[0];
        for (size_t i = 1; i < picked.size(); ++i) mix = b.CreateXor(mix, picked[i], "BosmopqXm");
    }

    const uint64_t k1 = Rng() | 1ull;
    const uint64_t k2 = Rng() | 1ull;
    Value* t1 = b.CreateLShr(mix, ConstantInt::get(i64, 30), "BosmopqSm1");
    mix = b.CreateXor(mix, t1, "BosmopqSm2");
    mix = b.CreateMul(mix, ConstantInt::get(i64, k1), "BosmopqSm3");
    Value* t2 = b.CreateLShr(mix, ConstantInt::get(i64, 27), "BosmopqSm4");
    mix = b.CreateXor(mix, t2, "BosmopqSm5");
    mix = b.CreateMul(mix, ConstantInt::get(i64, k2), "BosmopqSm6");
    Value* t3 = b.CreateLShr(mix, ConstantInt::get(i64, 31), "BosmopqSm7");
    mix = b.CreateXor(mix, t3, "BosmopqSm8");

    // Lemire multiply high; `((trunc_i32(scaled)) * tablesize) >> 32`; works for ANY table size (pow 2 OR not) and gives a value uniformly distributed in [0, tablesize); using the simpler `shr by log2(size)` path only works for pow 2 sizes and would exclude the 192/384-byte tables from the pool
    const uint64_t mult = MultiplierPool[Rng() % 16u];
    Value* scaled = b.CreateMul(mix, ConstantInt::get(i64, mult), "BosmopqRm");
    Value* strunc = b.CreateTrunc(scaled, i32, "BosmopqRt");
    Value* szext = b.CreateZExt(strunc, i64, "BosmopqRz");
    Value* smulsize = b.CreateMul(szext, ConstantInt::get(i64, (uint64_t)tablesize), "BosmopqRms");
    Value* reduced = b.CreateLShr(smulsize, ConstantInt::get(i64, 32), "BosmopqRr");
    return b.CreateTrunc(reduced, i32, "BosmopqId");
}

static bool IsSafeToClone(const Instruction& i)
{
    if (isa<PHINode>(&i)) return false;
    if (i.isTerminator()) return false;
    if (isa<AllocaInst>(&i)) return false;
    if (isa<LandingPadInst>(&i)) return false;
    if (isa<CatchPadInst>(&i)) return false;
    if (isa<CleanupPadInst>(&i)) return false;
    if (isa<CallBase>(&i)) return false;
    if (isa<FenceInst>(&i)) return false;
    if (isa<AtomicRMWInst>(&i)) return false;
    if (isa<AtomicCmpXchgInst>(&i)) return false;
    if (auto* li = dyn_cast<LoadInst>(&i)) { if (li->isVolatile() || li->isAtomic()) return false; }
    if (auto* si = dyn_cast<StoreInst>(&i)) { if (si->isVolatile() || si->isAtomic()) return false; }
    return true;
}

static bool FindMovableSuffix(BasicBlock& bb, SmallVectorImpl<Instruction*>& out)
{
    out.clear();
    Instruction* term = bb.getTerminator();
    if (!term) return false;
    DenseSet<Instruction*> inwin;
    for (Instruction* curr = term->getPrevNode(); curr; curr = curr->getPrevNode())
    {
        if (!IsSafeToClone(*curr)) break;
        bool allinwin = true;
        for (User* u : curr->users())
        {
            Instruction* ui = dyn_cast<Instruction>(u);
            if (!ui) continue;
            if (!inwin.contains(ui)) { allinwin = false; break; }
        }
        if (!allinwin) break;
        out.push_back(curr);
        inwin.insert(curr);
    }
    std::reverse(out.begin(), out.end());
    return out.size() >= DiamondMinSuffix;
}

struct SlotGroup
{
    Type* AllocatedTy;
    SmallVector<AllocaInst*, 4> Allocas;
};

static void CollectStaticAllocas(Function& f, SmallVectorImpl<SlotGroup>& out)
{
    out.clear();
    DenseMap<Type*, SmallVector<AllocaInst*, 4>> bytype;
    for (Instruction& i : f.getEntryBlock())
    {
        auto* a = dyn_cast<AllocaInst>(&i);
        if (!a || !a->isStaticAlloca()) continue;
        bytype[a->getAllocatedType()].push_back(a);
    }
    for (auto& p : bytype)
    {
        if (p.second.size() < 2) continue;
        SlotGroup g;
        g.AllocatedTy = p.first;
        g.Allocas = p.second;
        out.push_back(g);
    }
}

static void BuildStackSlotRemap(ArrayRef<Instruction*> suffix, ArrayRef<SlotGroup> groups, DenseMap<AllocaInst*, AllocaInst*>& outremap, std::mt19937_64& rng)
{
    DenseSet<AllocaInst*> referenced;
    for (Instruction* i : suffix)
    {
        for (Use& u : i->operands())
        {
            if (auto* a = dyn_cast<AllocaInst>(u.get())) referenced.insert(a);
        }
    }

    for (AllocaInst* src : referenced)
    {
        const SlotGroup* g = nullptr;
        for (const SlotGroup& candidate : groups)
        {
            if (candidate.AllocatedTy == src->getAllocatedType()) { g = &candidate; break; }
        }
        if (!g) { outremap[src] = src; continue; }

        AllocaInst* pick = src;
        for (int tries = 0; tries < 8; ++tries)
        {
            AllocaInst* cand = g->Allocas[rng() % g->Allocas.size()];
            if (cand != src) { pick = cand; break; }
        }
        outremap[src] = pick;
    }
}

static void CloneSuffixIntoDecoy(IRBuilder<>& b, ArrayRef<Instruction*> suffix, const DenseMap<AllocaInst*, AllocaInst*>& remap, std::mt19937_64& rng)
{
    DenseMap<Value*, Value*> vmap;
    const uint64_t mutnonce = rng();

    for (Instruction* orig : suffix)
    {
        Instruction* cl = orig->clone();
        cl->setName("");

        SmallVector<std::pair<unsigned, MDNode*>, 4> allmd;
        cl->getAllMetadata(allmd);
        for (auto& p : allmd) cl->setMetadata(p.first, nullptr);

        for (unsigned op = 0; op < cl->getNumOperands(); ++op)
        {
            Value* v = cl->getOperand(op);
            if (!v) continue;

            auto vmit = vmap.find(v);
            if (vmit != vmap.end()) { cl->setOperand(op, vmit->second); continue; }

            if (auto* a = dyn_cast<AllocaInst>(v))
            {
                auto rmit = remap.find(a);
                if (rmit != remap.end() && rmit->second != a) cl->setOperand(op, rmit->second);
                continue;
            }

            if (auto* ci = dyn_cast<ConstantInt>(v))
            {
                if (!ci->getType()->isIntegerTy()) continue;
                // shift amount operand (position 1 of Shl/LShr/AShr) MUST stay in [1, w-1] because perturbation would produce an out of range shift and LLVM's verifier rejects or the backend emits poison
                const unsigned opc = cl->getOpcode();
                if ((opc == Instruction::Shl || opc == Instruction::LShr || opc == Instruction::AShr) && op == 1u) continue;
                if (isa<GetElementPtrInst>(cl)) continue;
                if ((rng() % 100u) < ConstPerturbPct)
                {
                    cl->setOperand(op, ConstantInt::get(ci->getType(), ci->getZExtValue() ^ mutnonce));
                }
                continue;
            }
        }

        b.Insert(cl);
        vmap[orig] = cl;
    }
}

static void FillClassicDecoy(Function& f, BasicBlock* decoybb, BasicBlock* realbb, std::mt19937_64& rng)
{
    Module& m = *f.getParent();
    LLVMContext& c = f.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);
    IRBuilder<> b(decoybb);

    const unsigned diamondcap = std::min<unsigned>((unsigned)(realbb->size() > 1 ? realbb->size() - 1 : 0), ClassicDiamondCloneCap);
    ValueToValueMapTy vmap;
    unsigned cloned = 0;
    const uint64_t mutnonce = rng();
    for (Instruction& i : *realbb)
    {
        if (cloned >= diamondcap) break;
        if (i.isTerminator()) break;
        if (!IsSafeToClone(i)) continue;

        bool ok = true;
        for (Use& u : i.operands())
        {
            Value* op = u.get();
            auto* oi = dyn_cast<Instruction>(op);
            if (!oi) continue;
            if (vmap.count(op)) continue;
            if (oi->getParent() != realbb && oi->getParent() != decoybb) continue;
            ok = false;
            break;
        }
        if (!ok) continue;

        Instruction* cl = i.clone();
        cl->setName("");
        cl->insertInto(decoybb, decoybb->end());
        RemapInstruction(cl, vmap, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);

        if (auto* bo = dyn_cast<BinaryOperator>(cl))
        {
            for (unsigned op = 0; op < bo->getNumOperands(); ++op)
            {
                if (auto* ci = dyn_cast<ConstantInt>(bo->getOperand(op)))
                {
                    if ((rng() % 100u) < ConstPerturbPct)
                    {
                        bo->setOperand(op, ConstantInt::get(ci->getType(), ci->getZExtValue() ^ mutnonce));
                    }
                }
            }
        }
        vmap[&i] = cl;
        ++cloned;
    }

    b.SetInsertPoint(decoybb);

    const unsigned ops = ClassicDecoyOpsMin + (unsigned)(rng() % ClassicDecoyOpsRange);

    // 1-3 intermediate volatile stores interleaved into the arithmetic chain anchor a prefix of the computation so ida displays decoy as multiple statements instead of collapsing the whole chain into one expression tree
    const unsigned numintermediate = 1u + (unsigned)(rng() % IntermediateStoreMax);
    SmallVector<unsigned, 4> storepositions;
    for (unsigned k = 0; k < numintermediate; ++k) storepositions.push_back((unsigned)(rng() % ops));
    std::sort(storepositions.begin(), storepositions.end());

    GlobalVariable* src0 = EnsureDecoySink(m, (unsigned)rng());
    LoadInst* ld = b.CreateLoad(i64, src0, true, "BosmopqDl");
    ld->setAlignment(Align(8));
    Value* acc = ld;

    size_t nextstoreidx = 0;
    for (unsigned n = 0; n < ops; ++n)
    {
        const uint64_t k = rng() | 1ull;
        Value* kv = ConstantInt::get(i64, k);
        switch (rng() % 8u)
        {
            case 0: acc = b.CreateXor(acc, kv, "BosmopqDx"); break;
            case 1: acc = b.CreateAdd(acc, kv, "BosmopqDa"); break;
            case 2: acc = b.CreateSub(acc, kv, "BosmopqDs"); break;
            case 3: acc = b.CreateMul(acc, kv, "BosmopqDm"); break;
            case 4: acc = b.CreateOr(acc, kv, "BosmopqDo"); break;
            case 5: acc = b.CreateAnd(acc, kv, "BosmopqDn"); break;
            case 6:
            {
                const unsigned shamt = 1u + (unsigned)(rng() % 31u);
                acc = b.CreateShl(acc, ConstantInt::get(i64, shamt), "BosmopqDsl");
                break;
            }
            default:
            {
                const unsigned shamt = 1u + (unsigned)(rng() % 31u);
                acc = b.CreateLShr(acc, ConstantInt::get(i64, shamt), "BosmopqDsr");
                break;
            }
        }

        if ((rng() % 100u) < IntermediateStorePctLoad)
        {
            GlobalVariable* sx = EnsureDecoySink(m, (unsigned)rng());
            LoadInst* lx = b.CreateLoad(i64, sx, true, "BosmopqDlx");
            lx->setAlignment(Align(8));
            switch (rng() % 4u)
            {
                case 0: acc = b.CreateXor(acc, lx, "BosmopqDxx"); break;
                case 1: acc = b.CreateAdd(acc, lx, "BosmopqDax"); break;
                case 2: acc = b.CreateAnd(acc, lx, "BosmopqDnx"); break;
                default: acc = b.CreateOr(acc, lx, "BosmopqDox"); break;
            }
        }

        while (nextstoreidx < storepositions.size() && storepositions[nextstoreidx] == n)
        {
            GlobalVariable* middst = EnsureDecoySink(m, (unsigned)rng());
            StoreInst* midst = b.CreateStore(acc, middst);
            midst->setVolatile(true);
            midst->setAlignment(Align(8));
            ++nextstoreidx;
        }
    }

    GlobalVariable* dst = EnsureDecoySink(m, (unsigned)rng());
    StoreInst* st = b.CreateStore(acc, dst);
    st->setVolatile(true);
    st->setAlignment(Align(8));

    b.CreateBr(realbb);
}

static void FillDiamondDecoy(Function& f, BasicBlock* decoybb, ArrayRef<Instruction*> suffix, ArrayRef<SlotGroup> groups, std::mt19937_64& rng, BasicBlock* realbb)
{
    Module& m = *f.getParent();
    LLVMContext& c = f.getContext();
    IntegerType* i64 = Type::getInt64Ty(c);
    IRBuilder<> b(decoybb);

    DenseMap<AllocaInst*, AllocaInst*> remap;
    BuildStackSlotRemap(suffix, groups, remap, rng);
    CloneSuffixIntoDecoy(b, suffix, remap, rng);

    // short sink pool volatile chain MUST follow the suffix clone so the decoy always has an observable side effect even when the cloned suffix was pure computation that DCE could eliminate; without it a diamond site whose suffix only computed values gets its decoy stripped and the predicate collapses at opt time
    b.SetInsertPoint(decoybb);
    const unsigned ops = DiamondDecoyOpsMin + (unsigned)(rng() % DiamondDecoyOpsRange);
    GlobalVariable* src0 = EnsureDecoySink(m, (unsigned)rng());
    LoadInst* ld = b.CreateLoad(i64, src0, true, "BosmopqDla");
    ld->setAlignment(Align(8));
    Value* acc = ld;
    for (unsigned n = 0; n < ops; ++n)
    {
        const uint64_t k = rng() | 1ull;
        Value* kv = ConstantInt::get(i64, k);
        switch (rng() % 4u)
        {
            case 0: acc = b.CreateXor(acc, kv, "BosmopqDax"); break;
            case 1: acc = b.CreateAdd(acc, kv, "BosmopqDaa"); break;
            case 2: acc = b.CreateMul(acc, kv, "BosmopqDam"); break;
            default: acc = b.CreateOr(acc, kv, "BosmopqDao"); break;
        }
    }
    GlobalVariable* dst = EnsureDecoySink(m, (unsigned)rng());
    StoreInst* st = b.CreateStore(acc, dst);
    st->setVolatile(true);
    st->setAlignment(Align(8));

    b.CreateBr(realbb);
}

static unsigned StealIntoHead(BasicBlock* headbb, BasicBlock* realbb, unsigned maxsteal)
{
    unsigned stolen = 0;
    while (stolen < maxsteal)
    {
        Instruction* first = realbb->getFirstNonPHIOrDbgOrLifetime();
        if (!first || first->isTerminator()) break;
        if (isa<CallBase>(first) || isa<AllocaInst>(first) || isa<LandingPadInst>(first) || isa<CatchPadInst>(first) || isa<CleanupPadInst>(first)) break;
        if (first->isAtomic() || isa<FenceInst>(first)) break;
        if (first->mayHaveSideEffects()) break;

        bool opsok = true;
        for (Use& op : first->operands())
        {
            if (auto* oi = dyn_cast<Instruction>(op.get()))
            {
                if (oi->getParent() == realbb) { opsok = false; break; }
            }
        }
        if (!opsok) break;

        first->removeFromParent();
        first->insertInto(headbb, headbb->end());
        ++stolen;
    }
    return stolen;
}

struct CompareResult
{
    Value* Cond;
    bool RealOnTrue;
};

static CompareResult EmitSingleTableCompare(IRBuilder<>& builder, const SymMemTable& t, Value* idx, IntegerType* i32, IntegerType* i8)
{
    Value* p = builder.CreateInBoundsGEP(t.GV->getValueType(), t.GV, { ConstantInt::get(i32, 0), idx }, "BosmopqP");
    LoadInst* ld = builder.CreateLoad(i8, p, false, "BosmopqB");
    ld->setAlignment(Align(1));
    Value* bz = builder.CreateZExt(ld, i32, "BosmopqBz");

    const uint32_t nonce = (uint32_t)Rng();
    Value* xb = builder.CreateXor(bz, ConstantInt::get(i32, nonce), "BosmopqXb");
    Value* ma = builder.CreateAnd(xb, ConstantInt::get(i32, t.Mask), "BosmopqMa");
    const uint32_t expected = (t.Const ^ (nonce & t.Mask)) & t.Mask;

    const bool usene = (Rng() & 1u) != 0;
    Value* cond = usene ? builder.CreateICmpNE(ma, ConstantInt::get(i32, expected), "BosmopqCn") : builder.CreateICmpEQ(ma, ConstantInt::get(i32, expected), "BosmopqCe");
    return { cond, !usene };
}

static CompareResult EmitCompoundCompare(IRBuilder<>& builder, const SymMemPair& p, Value* idx1, Value* idx2, IntegerType* i32, IntegerType* i8)
{
    Value* p1 = builder.CreateInBoundsGEP(p.T1.GV->getValueType(), p.T1.GV, { ConstantInt::get(i32, 0), idx1 }, "BosmopqP1");
    LoadInst* ld1 = builder.CreateLoad(i8, p1, false, "BosmopqB1");
    ld1->setAlignment(Align(1));
    Value* bz1 = builder.CreateZExt(ld1, i32, "BosmopqBz1");

    Value* p2 = builder.CreateInBoundsGEP(p.T2.GV->getValueType(), p.T2.GV, { ConstantInt::get(i32, 0), idx2 }, "BosmopqP2");
    LoadInst* ld2 = builder.CreateLoad(i8, p2, false, "BosmopqB2");
    ld2->setAlignment(Align(1));
    Value* bz2 = builder.CreateZExt(ld2, i32, "BosmopqBz2");

    Value* xored = builder.CreateXor(bz1, bz2, "BosmopqCx");
    const uint32_t nonce = (uint32_t)Rng();
    Value* xb = builder.CreateXor(xored, ConstantInt::get(i32, nonce), "BosmopqCxn");
    Value* ma = builder.CreateAnd(xb, ConstantInt::get(i32, p.Mask), "BosmopqCma");
    const uint32_t expected = (p.InvXorConst ^ (nonce & p.Mask)) & p.Mask;

    const bool usene = (Rng() & 1u) != 0;
    Value* cond = usene ? builder.CreateICmpNE(ma, ConstantInt::get(i32, expected), "BosmopqCcn") : builder.CreateICmpEQ(ma, ConstantInt::get(i32, expected), "BosmopqCce");
    return { cond, !usene };
}

static bool TryEmitClassic(Function& f, BasicBlock* bb, DominatorTree& dt)
{
    if (bb->size() < MinBBInsts) return false;
    Instruction* splitat = PickInsertionPoint(bb);
    if (!splitat) return false;
    if (Tables.empty()) return false;

    const SymMemTable& t = Tables[Rng() % Tables.size()];
    Module& m = *f.getParent();
    LLVMContext& c = f.getContext();
    IntegerType* i8 = Type::getInt8Ty(c);
    IntegerType* i32 = Type::getInt32Ty(c);

    IRBuilder<> builder(splitat);
    Value* idx = BuildTableIndex(builder, f, dt, splitat, m, t.Size);
    CompareResult cr = EmitSingleTableCompare(builder, t, idx, i32, i8);

    BasicBlock* realbb = bb->splitBasicBlock(splitat, "Cont");
    BasicBlock* decoybb = BasicBlock::Create(c, "Cont", &f, realbb);
    FillClassicDecoy(f, decoybb, realbb, Rng);
    MarkDecoyBlock(decoybb);

    Instruction* splitbr = bb->getTerminator();
    splitbr->eraseFromParent();

    unsigned stealcap = 0;
    const unsigned rbsize = (unsigned)realbb->size();
    if (rbsize > RealBBMinResidue)
    {
        const unsigned headroom = rbsize - RealBBMinResidue;
        const unsigned ceiling = std::min(StealCeiling, headroom);
        stealcap = 1u + (unsigned)(Rng() % ceiling);
        const unsigned floor = std::min(StealFloor, ceiling);
        if (stealcap < floor) stealcap = floor;
    }
    if (stealcap > 0) (void)StealIntoHead(bb, realbb, stealcap);

    IRBuilder<> termbuilder(bb);
    if (cr.RealOnTrue) termbuilder.CreateCondBr(cr.Cond, realbb, decoybb);
    else termbuilder.CreateCondBr(cr.Cond, decoybb, realbb);

    MarkRealBB(realbb);
    MarkBBProcessed(bb);
    return true;
}

// diamond mode intentionally skips head steal because the whole suffix IS the real work and splitting part of it back into head would make the real control flow depend on a partial suffix, breaking semantic equivalence with the pre pass CFG
static bool TryEmitDiamond(Function& f, BasicBlock* bb, DominatorTree& dt)
{
    if (bb->size() < MinBBInsts) return false;

    SmallVector<Instruction*, 16> suffix;
    if (!FindMovableSuffix(*bb, suffix)) return false;
    if (suffix.size() < DiamondMinSuffix) return false;

    Instruction* splitat = suffix.front();

    // entry block alloca safety; reject the site when ANY alloca sits after splitat in bb, because splitting there would demote that alloca from static to dynamic; classic fallback via PickInsertionPoint advances past the last entry alloca and is the correct path here
    if (bb == &f.getEntryBlock())
    {
        for (Instruction& i : *bb)
        {
            if (&i == splitat) break;
            if (isa<AllocaInst>(&i))
            {
                bool bad = false;
                bool pastsplit = false;
                for (Instruction& j : *bb)
                {
                    if (&j == splitat) pastsplit = true;
                    if (pastsplit && isa<AllocaInst>(&j)) { bad = true; break; }
                }
                if (bad) return false;
                break;
            }
        }
    }

    SmallVector<SlotGroup, 8> groups;
    CollectStaticAllocas(f, groups);

    if (Tables.empty()) return false;
    const SymMemTable& t = Tables[Rng() % Tables.size()];

    Module& m = *f.getParent();
    LLVMContext& c = f.getContext();
    IntegerType* i8 = Type::getInt8Ty(c);
    IntegerType* i32 = Type::getInt32Ty(c);

    IRBuilder<> builder(splitat);
    Value* idx = BuildTableIndex(builder, f, dt, splitat, m, t.Size);
    CompareResult cr = EmitSingleTableCompare(builder, t, idx, i32, i8);

    BasicBlock* realbb = bb->splitBasicBlock(splitat, "Cont");
    BasicBlock* decoybb = BasicBlock::Create(c, "Cont", &f, realbb);

    SmallVector<Instruction*, 16> movedsuffix;
    for (Instruction& i : *realbb)
    {
        if (i.isTerminator()) break;
        movedsuffix.push_back(&i);
    }

    FillDiamondDecoy(f, decoybb, movedsuffix, groups, Rng, realbb);
    MarkDecoyBlock(decoybb);

    Instruction* splitbr = bb->getTerminator();
    splitbr->eraseFromParent();

    IRBuilder<> termbuilder(bb);
    if (cr.RealOnTrue) termbuilder.CreateCondBr(cr.Cond, realbb, decoybb);
    else termbuilder.CreateCondBr(cr.Cond, decoybb, realbb);

    MarkRealBB(realbb);
    MarkBBProcessed(bb);
    return true;
}

static bool TryEmitCompound(Function& f, BasicBlock* bb, DominatorTree& dt)
{
    if (bb->size() < MinBBInsts) return false;
    if (Pairs.empty()) return false;

    Instruction* splitat = PickInsertionPoint(bb);
    if (!splitat) return false;

    const SymMemPair& p = Pairs[Rng() % Pairs.size()];
    Module& m = *f.getParent();
    LLVMContext& c = f.getContext();
    IntegerType* i8 = Type::getInt8Ty(c);
    IntegerType* i32 = Type::getInt32Ty(c);

    IRBuilder<> builder(splitat);
    Value* idx1 = BuildTableIndex(builder, f, dt, splitat, m, p.Size);
    Value* idx2 = BuildTableIndex(builder, f, dt, splitat, m, p.Size);
    CompareResult cr = EmitCompoundCompare(builder, p, idx1, idx2, i32, i8);

    BasicBlock* realbb = bb->splitBasicBlock(splitat, "Cont");
    BasicBlock* decoybb = BasicBlock::Create(c, "Cont", &f, realbb);
    FillClassicDecoy(f, decoybb, realbb, Rng);
    MarkDecoyBlock(decoybb);

    Instruction* splitbr = bb->getTerminator();
    splitbr->eraseFromParent();

    unsigned stealcap = 0;
    const unsigned rbsize = (unsigned)realbb->size();
    if (rbsize > RealBBMinResidue)
    {
        const unsigned headroom = rbsize - RealBBMinResidue;
        const unsigned ceiling = std::min(StealCeiling, headroom);
        stealcap = 1u + (unsigned)(Rng() % ceiling);
        const unsigned floor = std::min(StealFloor, ceiling);
        if (stealcap < floor) stealcap = floor;
    }
    if (stealcap > 0) (void)StealIntoHead(bb, realbb, stealcap);

    IRBuilder<> termbuilder(bb);
    if (cr.RealOnTrue) termbuilder.CreateCondBr(cr.Cond, realbb, decoybb);
    else termbuilder.CreateCondBr(cr.Cond, decoybb, realbb);

    MarkRealBB(realbb);
    MarkBBProcessed(bb);
    return true;
}

static bool TryEmitLoop(Function& f, BasicBlock* bb, DominatorTree& dt)
{
    if (bb->size() < MinBBInsts) return false;
    Instruction* splitat = PickInsertionPoint(bb);
    if (!splitat) return false;
    if (Tables.empty()) return false;

    const SymMemTable& t = Tables[Rng() % Tables.size()];
    Module& m = *f.getParent();
    LLVMContext& c = f.getContext();
    IntegerType* i8 = Type::getInt8Ty(c);
    IntegerType* i32 = Type::getInt32Ty(c);
    IntegerType* i64 = Type::getInt64Ty(c);

    IRBuilder<> headb(splitat);

    SmallVector<HarvestSlot, 16> candidates;
    CollectHarvestCandidates(f, dt, splitat, candidates);
    for (size_t i = candidates.size(); i > 1; --i)
    {
        const size_t j = (size_t)(Rng() % i);
        std::swap(candidates[i - 1], candidates[j]);
    }

    const unsigned want = HarvestMin + (unsigned)(Rng() % (HarvestMax - HarvestMin + 1u));
    SmallVector<Value*, 4> picked;
    for (unsigned i = 0; i < want && !candidates.empty(); ++i)
    {
        const unsigned idx = (unsigned)(Rng() % candidates.size());
        HarvestSlot& h = candidates[idx];
        Value* v = h.V;
        if (!h.Is64) v = headb.CreateZExt(v, i64, "BosmopqLhz");
        picked.push_back(v);
    }

    Value* harvestmix = nullptr;
    if (picked.empty())
    {
        GlobalVariable* noise = EnsureNoiseSlot(m);
        LoadInst* ld = headb.CreateLoad(i64, noise, true, "BosmopqLns");
        ld->setAlignment(Align(8));
        harvestmix = ld;
    }
    else
    {
        harvestmix = picked[0];
        for (size_t i = 1; i < picked.size(); ++i) harvestmix = headb.CreateXor(harvestmix, picked[i], "BosmopqLhxm");
    }

    GlobalVariable* limitgv = CreateLoopLimit(m);
    LoadInst* limitld = headb.CreateLoad(i32, limitgv, true, "BosmopqLl");
    limitld->setAlignment(Align(4));
    Value* limit = limitld;

    const uint64_t loopk1 = Rng() | 1ull;
    const uint64_t loopk2 = Rng() | 1ull;
    const uint64_t loopmult = MultiplierPool[Rng() % 16u];
    const uint64_t accupdk = Rng() | 1ull;
    const uint32_t nonce = (uint32_t)Rng();

    BasicBlock* realbb = bb->splitBasicBlock(splitat, "Cont");

    BasicBlock* loophdr = BasicBlock::Create(c, "LoopHdr", &f, realbb);
    BasicBlock* loopbody = BasicBlock::Create(c, "LoopBody", &f, realbb);
    BasicBlock* decoybb = BasicBlock::Create(c, "Cont", &f, realbb);

    Instruction* splitbr = bb->getTerminator();
    splitbr->eraseFromParent();
    IRBuilder<> bbb(bb);
    bbb.CreateBr(loophdr);

    IRBuilder<> hb(loophdr);
    PHINode* counterphi = hb.CreatePHI(i64, 2, "BosmopqCtr");
    counterphi->addIncoming(ConstantInt::get(i64, 0), bb);
    PHINode* accumphi = hb.CreatePHI(i64, 2, "BosmopqAcc");
    accumphi->addIncoming(harvestmix, bb);

    Value* mix = hb.CreateXor(counterphi, accumphi, "BosmopqLmix");

    Value* sm1 = hb.CreateLShr(mix, ConstantInt::get(i64, 30), "BosmopqLsm1");
    mix = hb.CreateXor(mix, sm1, "BosmopqLsm2");
    mix = hb.CreateMul(mix, ConstantInt::get(i64, loopk1), "BosmopqLsm3");
    Value* sm2 = hb.CreateLShr(mix, ConstantInt::get(i64, 27), "BosmopqLsm4");
    mix = hb.CreateXor(mix, sm2, "BosmopqLsm5");
    mix = hb.CreateMul(mix, ConstantInt::get(i64, loopk2), "BosmopqLsm6");
    Value* sm3 = hb.CreateLShr(mix, ConstantInt::get(i64, 31), "BosmopqLsm7");
    mix = hb.CreateXor(mix, sm3, "BosmopqLsm8");

    Value* scaled = hb.CreateMul(mix, ConstantInt::get(i64, loopmult), "BosmopqLrm");
    Value* strunc = hb.CreateTrunc(scaled, i32, "BosmopqLrt");
    Value* szext = hb.CreateZExt(strunc, i64, "BosmopqLrz");
    Value* smulsize = hb.CreateMul(szext, ConstantInt::get(i64, (uint64_t)t.Size), "BosmopqLrms");
    Value* reduced = hb.CreateLShr(smulsize, ConstantInt::get(i64, 32), "BosmopqLrr");
    Value* idx = hb.CreateTrunc(reduced, i32, "BosmopqLid");

    Value* p = hb.CreateInBoundsGEP(t.GV->getValueType(), t.GV, { ConstantInt::get(i32, 0), idx }, "BosmopqLp");
    LoadInst* ld = hb.CreateLoad(i8, p, false, "BosmopqLb");
    ld->setAlignment(Align(1));
    Value* bz = hb.CreateZExt(ld, i32, "BosmopqLbz");
    Value* xb = hb.CreateXor(bz, ConstantInt::get(i32, nonce), "BosmopqLxb");
    Value* ma = hb.CreateAnd(xb, ConstantInt::get(i32, t.Mask), "BosmopqLma");
    const uint32_t expected = (t.Const ^ (nonce & t.Mask)) & t.Mask;

    // loop mode uses EQ only; the predicate MUST evaluate TRUE each iteration for the loop to progress to loopbody; NE flip would require inverting every successor on the loopbody/decoybb edge, which is a layout change with no security benefit because the attacker sees the same cmp either way
    Value* okcond = hb.CreateICmpEQ(ma, ConstantInt::get(i32, expected), "BosmopqLok");
    hb.CreateCondBr(okcond, loopbody, decoybb);

    IRBuilder<> lb(loopbody);
    Value* ctrnext = lb.CreateAdd(counterphi, ConstantInt::get(i64, 1), "BosmopqLctn");
    // per iteration accum update XORs with (counter * K) to give accum a per iteration none linear shift so successive iterations don't produce correlatable mix inputs; a linear accum update would let an SMT solver see "mix cycles trivially" and collapse the N iteration state product to a single state
    Value* accmul = lb.CreateMul(counterphi, ConstantInt::get(i64, accupdk), "BosmopqLacmul");
    Value* accnext = lb.CreateXor(accumphi, accmul, "BosmopqLacn");
    Value* limit64 = lb.CreateZExt(limit, i64, "BosmopqLl64");
    Value* exitcond = lb.CreateICmpULT(ctrnext, limit64, "BosmopqLec");
    BranchInst* latchbr = lb.CreateCondBr(exitcond, loophdr, realbb);

    // loop metadata requires the MDNode's first operand to reference the node itself (replaceOperandWith(0, loopmd)); without the self reference LLVM's loop pass manager asserts on malformed loop MD and the unroll disable hint is lost
    MDNode* unrolldisable = MDNode::get(c, { MDString::get(c, "llvm.loop.unroll.disable") });
    MDNode* loopmd = MDNode::get(c, { nullptr, unrolldisable });
    loopmd->replaceOperandWith(0, loopmd);
    latchbr->setMetadata(LLVMContext::MD_loop, loopmd);

    counterphi->addIncoming(ctrnext, loopbody);
    accumphi->addIncoming(accnext, loopbody);

    FillClassicDecoy(f, decoybb, realbb, Rng);
    MarkDecoyBlock(decoybb);

    MarkRealBB(realbb);
    MarkBBProcessed(bb);
    return true;
}

static bool TryEmitAtBB(Function& f, BasicBlock* bb, DominatorTree& dt)
{
    if (BBAlreadyProcessed(bb)) return false;
    if (SkipEntry && bb == &f.getEntryBlock()) return false;
    if (IsEHOrFuncletInterior(bb)) return false;

    const unsigned roll = (unsigned)(Rng() % 100u);
    unsigned cum = 0;

    cum += WeightDiamond;
    if (roll < cum)
    {
        if (TryEmitDiamond(f, bb, dt)) return true;
        return TryEmitClassic(f, bb, dt);
    }

    cum += WeightCompound;
    if (roll < cum)
    {
        if (TryEmitCompound(f, bb, dt)) return true;
        return TryEmitClassic(f, bb, dt);
    }

    cum += WeightLoop;
    if (roll < cum)
    {
        if (TryEmitLoop(f, bb, dt)) return true;
        return TryEmitClassic(f, bb, dt);
    }

    return TryEmitClassic(f, bb, dt);
}

// BFS up to depth 3 because loop sites hide real_bb behind bb → loopHdr → loopBody → real_bb, while classic/diamond/compound sit at depth 1; a narrower walk misses loop sites and the worklist stops chaining through them, and decoy tagged BBs MUST be skipped because their successors loop back to real_bb and would create false positive hits
static BasicBlock* FindRealBBAfterEmit(BasicBlock* bb)
{
    SmallVector<BasicBlock*, 8> frontier;
    SmallPtrSet<BasicBlock*, 16> seen;
    frontier.push_back(bb);
    seen.insert(bb);
    for (unsigned depth = 0; depth < 3u && !frontier.empty(); ++depth)
    {
        SmallVector<BasicBlock*, 8> next;
        for (BasicBlock* b : frontier)
        {
            Instruction* term = b->getTerminator();
            if (!term) continue;
            for (unsigned i = 0; i < term->getNumSuccessors(); ++i)
            {
                BasicBlock* s = term->getSuccessor(i);
                if (!seen.insert(s).second) continue;
                if (IsDecoyBlock(s)) continue;
                if (IsRealBB(s)) return s;
                next.push_back(s);
            }
        }
        frontier = next;
    }
    return nullptr;
}

static bool ProcessFunction(Function& f)
{
    if (f.isDeclaration()) return false;
    if (HasNoObfuscateAnnotation(f) || HasNoBosmopqAnnotation(f) || HasNoOpaqueAnnotation(f)) return false;
    if (!HasObfuscateAnnotation(f)) return false;

    SmallVector<BasicBlock*, 64> worklist;
    SmallPtrSet<BasicBlock*, 32> seen;
    for (BasicBlock& bb : f)
    {
        worklist.push_back(&bb);
        seen.insert(&bb);
    }

    unsigned emitted = 0;
    bool mutated = false;

    while (!worklist.empty() && emitted < PerFunctionCap)
    {
        BasicBlock* bb = worklist.pop_back_val();
        if (IsDecoyBlock(bb)) continue;

        DominatorTree dt;
        dt.recalculate(f);
        LoopInfo li;
        li.analyze(dt);
        if (SkipLoops && li.getLoopFor(bb) != nullptr) continue;

        if (!TryEmitAtBB(f, bb, dt)) continue;

        ++emitted;
        mutated = true;

        if (BasicBlock* realbb = FindRealBBAfterEmit(bb))
        {
            if (seen.insert(realbb).second) worklist.push_back(realbb);
        }
    }

    return mutated;
}

bool RunBiOpaqueSymMem(Module& m)
{
    if (!BiOpaqueSymMemEnable) return false;

    std::random_device rd;
    uint64_t seed = (uint64_t)rd();
    seed ^= (uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    seed ^= ((uint64_t)rd()) << 32;
    seed ^= 0x424F534D4F50515FULL;
    Rng.seed(seed);

    // every module scoped pointer MUST be reset per invocation; carrying a stale pointer from a previous module would reach into reclaimed storage when the lazy create check skips, and THIS module needs its own fresh globals / decryptor / loop limit counters on first demand
    NoiseSlot = nullptr;
    TableDecryptor = nullptr;
    LoopLimitModule = nullptr;
    LoopLimitNext = 0;
    for (unsigned i = 0; i < DecoySinkPoolSize; ++i) DecoySinks[i] = nullptr;
    Tables.clear();
    Pairs.clear();
    TablesModule = nullptr;

    EnsureTablePool(m);
    if (Tables.empty()) return false;

    bool mutated = false;
    for (Function& f : m) mutated |= ProcessFunction(f);

    Tables.clear();
    Pairs.clear();
    TablesModule = nullptr;
    return mutated;
}

PreservedAnalyses BiOpaqueSymMemPass::run(Module& m, ModuleAnalysisManager&)
{
    bool mutated = RunBiOpaqueSymMem(m);
    return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
}