#include<string>
#include<vector>
#include<iostream>
#include<fstream>
#include "llvm/Pass.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Constant.h"
#include "llvm/Analysis/LoopInfo.h"
#include<unordered_map>
#include "Graph.h"

using namespace llvm;
using namespace graph;
//This pass is to insert timing function before & after each function and also analysis 
//data dependencies among functions.

//Construct graph for each function

namespace{

    struct Profiling : public ModulePass{
        static char ID;

        Profiling();
        ~Profiling();

        bool runOnModule(Module & M) override;
        std::string get_father_dir(std::string);
        void get_tool_library_func(Module & M);
        void getAnalysisUsage(AnalysisUsage &AU) const override;

        //data
        std::string profiling_data_file_path;
        std::unordered_map<Function*,bool> tool_func_map;
        std::unordered_map<CallInst*,Node*> callinst_node_map;

    };

    Profiling::Profiling() : ModulePass(ID){

    }

    Profiling::~Profiling(){

    }

    void Profiling::getAnalysisUsage(AnalysisUsage &AU) const{
        AU.addRequired<LoopInfoWrapperPass>();
    }

    bool Profiling::runOnModule(Module & M){
        std::string father_path;
        std::string ModuleName = M.getModuleIdentifier();
        father_path = get_father_dir(ModuleName);
        std::string profiling_data_file_path = father_path+"profiling.txt";
        std::string data_dependency_graph_file_path = father_path+"DDG.txt";

        //This is to profile
        //TO.DO.: need an extra function to output the timing into file(TOOL LIBRARY)
        auto hipEvent_t = StructType::create(M.getContext(), "struct.ihipEvent_t");
        auto hipEvent_PtrT = PointerType::get(hipEvent_t,0);
        GlobalVariable * gpu_start_event_ptr = new GlobalVariable(M,hipEvent_PtrT,false,GlobalValue::CommonLinkage,
                                                                    0,"mzw_gpu_start");
        GlobalVariable * gpu_end_event_ptr = new GlobalVariable(M,hipEvent_PtrT,false,GlobalValue::CommonLinkage,
                                                                    0,"mzw_gpu_end");

        gpu_start_event_ptr->setAlignment(MaybeAlign(8));
        gpu_end_event_ptr->setAlignment(MaybeAlign(8));
        ConstantPointerNull * NULLPTR = ConstantPointerNull::get(hipEvent_PtrT);
        if(NULLPTR == nullptr)
        {
            errs()<<"cannot gen nullptr for hipevent\n";
            exit(1);
        }
        gpu_start_event_ptr->setInitializer(NULLPTR);
        gpu_end_event_ptr->setInitializer(NULLPTR);

        //In order to construct the ddg of each function(CPU function, GPU kernel and Memcpy)
        //First we need to identify them
        //For Kernel, we will use hipLaunchKernel to call kernel, so use it to recognize
        //For Memcpy, we will use hipMemcpy, so use it to recognize
        //For regular CPU function, we need user to add prefix of "cpu_" of each CPU function
        //Other functions in HIP and library functions will be treated as CPU functions.

        get_tool_library_func(M);


        //In fact, we need to handle all branch and loop first, just make them as a sub-graph
        //then hash them into a set, in the second walk(handle the sequential instruction) ingnore
        //those hashed inst.
        //Also, in the first walk, should we identify input and output of each function?
        for(Module::FunctionListType::iterator func = M.getFunctionList().begin(),
            func_end = M.getFunctionList().end(); func != func_end; func++)
        {
            Function * cur_func = dyn_cast<Function>(func);
            if(tool_func_map.find(cur_func) != tool_func_map.end()) continue;
            else if(cur_func->size() == 0) continue;

            LoopInfo & LI = getAnalysis<LoopInfoWrapperPass>(*cur_func).getLoopInfo();
            size_t loop_counter = 0;
            for(Loop * L : LI)        //this is for every outmost(top-level) loop?
            {
                errs()<<"Loop "<<++loop_counter<<"\n";
                //11-19 consider one bb, not nested loop
                for(BasicBlock * bb : L->getBlocks())
                {
                    for(BasicBlock::iterator inst = bb->begin(), inst_end = bb->end();
                        inst != inst_end; inst++)
                    {
                        
                    }
                }
            }
        }

        //Second walk.
        for(Module::FunctionListType::iterator func = M.getFunctionList().begin(),
            func_end = M.getFunctionList().end(); func != func_end; func++)
        {
            Function * caller_func = dyn_cast<Function>(func);
            if(caller_func == nullptr)
            {
                errs()<<"Fail to get the caller func\n";
                exit(1);
            }
            //We only care about the control flow in a function, we dont care the whole CFG
            //So we have a general garph in a function, but a sub-graph representing a loop / branch
            //is also a part of the general graph.
            GRAPH G;
            
            if(tool_func_map.find(caller_func) != tool_func_map.end()) continue;
            
            size_t inst_counter = 0;
            std::string func_name = func->getName().str();
            for(Function::iterator bb = func->begin(), bb_end = func->end(); bb != bb_end; bb++)
            {
                for(BasicBlock::iterator inst = bb->begin(), inst_end = bb->end(); inst != inst_end; inst++)
                {
                    inst_counter++;
                    if(isa<CallInst>(inst))
                    {
                        Function * called_func = dyn_cast<CallInst>(inst)->getCalledFunction();
                        if(called_func == nullptr)
                        {
                            continue;
                        }
                        CallInst * call_inst = dyn_cast<CallInst>(inst);
                        std::string called_func_name = called_func->getName().str();
                        if(called_func_name == "hipLaunchKernel")
                        {
                            //get the first argument as target kernel function
                            //errs()<<"We get one kernel call inst\n";
                            Value * v = dyn_cast<Value>(call_inst->getArgOperand(0));
                            //errs()<<*v<<"\n";
                            Function * kernel_func = dyn_cast<Function>(v->stripPointerCasts());
                            //errs()<<kernel_func->getName();
                            //errs()<<"\n";

                            //GPUFuncNode * node = GPUFuncNode(call_inst,kernel_func,true);
                            //G.Insert()
                        }
                        else if(called_func_name.find("cpu_") == 0)
                        {
                            //This is a cpu function
                            Function * cpu_func = dyn_cast<Function>(call_inst->getFunction());
                            
                        }
                        else if(called_func_name == "hipMemcpy")
                        {
                            //This is memcpy
                        }
                        else
                        {
                            //These are considered to be cpu function(how about hipblasGemmEx etc.?)
                        }
                    }
                }
            }
        }


        return true;
    }


    std::string Profiling::get_father_dir(std::string file_path)
    {
        std::string res = file_path;
        int n = file_path.length();
        int i = n - 1;
        for(; i >= 0; i--)
        {
            if(file_path[i] == '/') break;
        }
        int file_name_length = n - 1 - i;
        while(file_name_length >0)
        {
            file_name_length--;
            res.pop_back();
        }
        std::cout<<"Father path is "<<res<<std::endl;
        return res;
    }

    void Profiling::get_tool_library_func(Module & M)
    {
        //TO.DO.: 
        
    }

}

char Profiling::ID = 0;

static RegisterPass<Profiling> X("profiling","A Pass to Profile and run data analysis",
                                true, false);

static RegisterStandardPasses Y(
    PassManagerBuilder::EP_EarlyAsPossible,
    [] (const PassManagerBuilder & Builder, 
    legacy::PassManagerBase & PM) {PM.add(new Profiling());});













