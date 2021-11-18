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

using namespace llvm;

//This pass is to insert timing function before & after each function and also analysis 
//data dependencies among functions.

//Construct graph for each function

namespace{
    struct Node{
        Function * func;
        std::vector<Node*> succ;
        size_t undone_prec;
    };

    struct SuperNode{
        Node * node_list;
    };

    //A function has a graph
    //A branch has a graph(if-else), including loop
    struct Graph{
        Node * Entry_Node;

    };


    struct Profiling : public ModulePass{
        static char ID;

        Profiling();
        ~Profiling();

        bool runOnModule(Module & M) override;
        std::string get_father_dir(std::string);
        void get_tool_library_func(Module & M);

        //data
        std::string profiling_data_file_path;
        std::unordered_map<Function*,bool> tool_func_map;


    };

    Profiling::Profiling() : ModulePass(ID){

    }

    Profiling::~Profiling(){

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


        //In order to construct the ddg of each function(CPU function, GPU kernel and Memcpy)
        //First we need to identify them
        //For Kernel, we will use hipLaunchKernel to call kernel, so use it to recognize
        //For Memcpy, we will use hipMemcpy, so use it to recognize
        //For regular CPU function, we need user to add prefix of "cpu_" of each CPU function
        //Other functions in HIP and library functions will be treated as CPU functions.

        get_tool_library_func(M);

        for(Module::FunctionListType::iterator func = M.getFunctionList().begin(),
            func_end = M.getFunctionList().end(); func != func_end; func++)
        {
            if(tool_func_map.find(func) != tool_func_map.end()) continue;
            
            size_t inst_counter = 0;
            std::string func_name = func->getName().str();
            for(Function::iterator bb = func->begin(), bb_end = func->end(); bb != bb_end; bb++)
            {
                for(BasicBlock::iterator inst = bb->begin, inst_end = bb->end; inst != inst_end; inst++)
                {
                    inst_counter++;
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













