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
        //TO.DO.:  Handle the Loop Graph
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

        //We use this map to collect graph of each function, and walk through main() to build the whole CFG
        std::unordered_map<Function *, Seq_Graph*> Func_SG_map;

        //NOTE: When we have a function, the first argument(int32) n means the following n arguments
        //are output values(also input values)

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
            if(caller_func->size() == 0)
            {
                errs()<<"Func "<<caller_func->getName()<<" size is 0\n";
                continue;
            }
            //We only care about the control flow in a function, we dont care the whole CFG
            //So we have a general garph in a function, but a sub-graph representing a loop / branch
            //is also a part of the general graph.
            Seq_Graph * SG;
            if(Func_SG_map.find(caller_func) != Func_SG_map.end()) SG = Func_SG_map[caller_func];
            else
            {
                SG = new Seq_Graph();
                Func_SG_map[caller_func] = SG;
            } 

            if(tool_func_map.find(caller_func) != tool_func_map.end()) continue;
            
            size_t inst_counter = 0;
            //std::string func_name = func->getName().str();
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

                            GPUFuncNode * node = new GPUFuncNode(call_inst,kernel_func,true,true);
                            Node * prev_node = SG->get_last_Node();
                            SG->Insert(node,prev_node);

                            //Register Input/Output Value
                            Value * output_value_n = call_inst->getArgOperand(0);
                            if(ConstantInt * CI = dyn_cast<ConstantInt>(output_value_n))
                            {
                                size_t output_n = CI->getZExtValue();
                                size_t n = call_inst->getNumArgOperands();
                                for(size_t i = 1; i < n; i++)
                                {
                                    //TO.DO.: How to locate the right input argument?
                                    //if(i < 1 + output_n) node->add_output_value()
                                }
                            }
                            else
                            {
                                errs()<<"1st argument of Function "<<called_func_name<<" is not constant value\n";
                                exit(1);
                            }

                        }
                        else if(called_func_name.find("cpu_") != -1)
                        {
                            //This is a cpu function
                            Function * cpu_func = dyn_cast<Function>(call_inst->getFunction());
                            
                            CPUFuncNode * node = new CPUFuncNode(call_inst,cpu_func,false,true);
                            Node * prev_node = SG->get_last_Node();
                            SG->Insert(node,prev_node);
                        }
                        else if(called_func_name == "hipMemcpy")
                        {
                            //This is memcpy
                            Function * memcpy_func = dyn_cast<Function>(call_inst->getFunction());
                            MemcpyNode * node = new MemcpyNode(call_inst,memcpy_func,false,false);             //QUES.: Should the gpu_flag be true?
                            Node * prev_node = SG->get_last_Node();
                            SG->Insert(node,prev_node);
                        }
                        else
                        {
                            //These are considered to be cpu function(how about hipblasGemmEx etc.?)
                            //So far, we dont consider about hipblas stuff.
                            //TO.DO.: But hipMalloc should be specified to handle.
                            Function * cpu_library_func = dyn_cast<Function>(call_inst->getCalledFunction());
                            //errs()<<"we meet "<<cpu_library_func->getName()<<"\n";
                            CPUFuncNode * node = new CPUFuncNode(call_inst,cpu_library_func,false,false);
                            Node * prev_node = SG->get_last_Node();
                            SG->Insert(node,prev_node);
                        }
                    }
                    else
                    {
                        //These instructions will be collected by data-flow analysis of each call_inst
                        //For those instructions might not be collected after all these, maybe it's time to kick off
                        //collect them as InstNode and bundle related ones into one SuperNode
                        //So now we dont have to handle them here.
                    }
                }
            }

            //TO.DO.: Print out the function Seq-Graph                              DONE
            //errs()<<"This is Seq_Graph for Function "<<caller_func->getName()<<"\n";
            //SG->Print_allNode();

        }

        //TO.DO.: Analysis data input & output of each cpu/gpu function
        //Only passed pointer argument can be modified and affect other functions

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













