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
#include "llvm/IR/GlobalValue.h"
#include<unordered_map>
#include <limits>
using namespace llvm;

namespace{
    struct Gen : public ModulePass{
        static char ID;

        Gen();
        ~Gen();
        
        bool runOnModule(Module & M) override;
        void read_all_func_args_file();
        void read_runned_funcID_file();
        std::string get_father_path(std::string);
        void get_biggest_dimension();
        void get_tool_library_func(Module &, std::unordered_map<Function*,int> &);
        void make_runned_func_ID_set();
        template<typename T>
        bool is_in_list(std::vector<T>, T);

        //data
        std::string single_pass_data_file_path;
        std::string dimension_file_path;
        std::string runned_funcID_file_path;
        std::unordered_map<int,std::vector<int>> gemm_arg_m;        //read the sigle_pass_err.txt to init
        std::vector<int> runned_funcID_list;
        int pool_m;
        int pool_n;
        int pool_k;

    };

    Gen::Gen():ModulePass(ID){  pool_m = pool_n = pool_k = 0;}
    Gen::~Gen(){};

    bool Gen::runOnModule(Module & M){
        std::string ModuleName = M.getModuleIdentifier();
        std::string father_path = get_father_path(ModuleName);
        single_pass_data_file_path = father_path+"single_pass_err.txt";
        dimension_file_path = father_path+"dimension.txt";
        runned_funcID_file_path = father_path+"calledFuncID.txt";
        
        read_runned_funcID_file();
        get_biggest_dimension();
        read_all_func_args_file();

        std::unordered_map<Function*,int> tool_func_map;
        get_tool_library_func(M,tool_func_map);

        Function * fastermul_func_ptr = M.getFunction("_Z14mzw_faster_mulPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tS_iii");
        if(fastermul_func_ptr == nullptr)
        {
            errs()<<"Cannot get the fastermul func ptr\n";
            exit(1);
        }
        FunctionType * fastermul_func_type = fastermul_func_ptr->getFunctionType();

        size_t pool_size = 
        sizeof(float)*(
            2*pool_m*pool_k+
            2*pool_k*pool_n+
            2*pool_m*pool_n
        )+sizeof(short)*(
            pool_m*pool_k*2+
            pool_k*pool_n*2+
            pool_m*pool_n*2
        );
        std::cout<<"Pool size: "<<pool_size<<std::endl;

        PointerType * int8PtrType = PointerType::getUnqual(Type::getInt8Ty(M.getContext()));
        if(int8PtrType == nullptr)
        {
            errs()<<"Cannot generate i8* ptr type\n";
            exit(1);
        }
        GlobalVariable * gvar_ptr_mempool = new GlobalVariable(M,int8PtrType,false,GlobalValue::CommonLinkage,
                                                                0,"mzw_mempool");
        gvar_ptr_mempool->setAlignment(MaybeAlign(4));
        ConstantPointerNull * NULLPTR = ConstantPointerNull::get(int8PtrType);
        if(NULLPTR == nullptr)
        {
            errs()<<"Cannot generate nullptr value\n";
            exit(1);
        }
        gvar_ptr_mempool->setInitializer(NULLPTR);

        //TO.DO.: Loop over all inst to replace the gemmex inst
        int gemm_counter = 0;
        std::vector<Instruction*> inst_del;
        for(Module::FunctionListType::iterator func = M.getFunctionList().begin(),
            func_end = M.getFunctionList().end(); func != func_end; func++)
        {
            Function * caller_func = dyn_cast<Function>(func);

            //TO.DO.: Insert hipmalloc() at the begin of main and hipFree at every end of main()        DONE
            if(caller_func->getName() == "main")
            {
                int inst_counter = 0;
                Instruction * head_inst;
                std::vector<Instruction*> end_inst_list;        //store all ret instruction in main()
                for(Function::iterator bb = func->begin(), bb_end = func->end(); bb != bb_end; bb++)
                {
                    for(BasicBlock::iterator inst = bb->begin(), inst_end = bb->end(); inst != inst_end; inst++)
                    {
                        if(!inst_counter) head_inst = dyn_cast<Instruction>(inst);
                        inst_counter++;
                        //If this is a ret inst, store it into end_inst_list
                        if(isa<ReturnInst>(inst))
                        {
                            Instruction * ret_inst = dyn_cast<Instruction>(inst);
                            end_inst_list.push_back(ret_inst);
                        }
                    }
                }

                //For hipMalloc() at the head of main

                IRBuilder<> builder(head_inst);                                                         //QUES.: Should we need to insert before head_inst->getNextNode()
                Function * hipMalloc_func_ptr = M.getFunction("hipMalloc");
                if(hipMalloc_func_ptr == nullptr)
                {
                    errs()<<"Cannot get the hipMalloc func ptr\n";
                    exit(1);
                }
                FunctionType * hipMalloc_func_type = hipMalloc_func_ptr->getFunctionType();
                Function * hipFree_func_ptr = M.getFunction("hipFree");
                if(hipFree_func_ptr == nullptr)
                {
                    errs()<<"Cannot get the hipFree func ptr\n";
                    exit(1);
                }
                FunctionType * hipFree_func_type = hipFree_func_ptr->getFunctionType();

                //Here we just use the gvar_ptr_mempool as the first input arg
                Value * pool_ptr_ptr = dynamic_cast<Value*>(gvar_ptr_mempool);

                ConstantInt * MEMPOOLSIZE = builder.getInt64(pool_size);                    //NOTE:@hipMalloc(i8** nonnull %207, i64 1073741824) 不能是int32
                Value * argv_poolsize = dynamic_cast<Value*>(MEMPOOLSIZE);
                if(argv_poolsize == nullptr)
                {
                    errs()<<"Fail to create arg for poolsize\n";
                    exit(1);
                }
                
                std::vector<Value*> args = {pool_ptr_ptr,argv_poolsize};
                if(!builder.CreateCall(hipMalloc_func_type,hipMalloc_func_ptr,makeArrayRef(args)))
                {
                    errs()<<"Cannot create the call inst of hipMalloc\n";
                    exit(1);
                }
                
                while(args.size()) args.pop_back();

                //For hipFree() at the every ret of main
                for(Instruction * ret_inst : end_inst_list)
                {
                    builder.SetInsertPoint(ret_inst);
                    //get the ptr in the gvar_ptr_mempool
                    Value * pool_ptr = builder.CreateLoad(gvar_ptr_mempool,"mzw_pool_ptr");
                    if(pool_ptr == nullptr)
                    {
                        errs()<<"Cannot create load inst for pool_ptr\n";
                        exit(1);
                    }

                    args.push_back(pool_ptr);
                    if(!builder.CreateCall(hipFree_func_type,hipFree_func_ptr,makeArrayRef(args)))
                    {
                        errs()<<"Cannot create the call inst of hipFree\n";
                        exit(1);
                    }
                    args.pop_back();
                }

            }

            if(tool_func_map.find(caller_func) != tool_func_map.end()) continue;
            else
            {
                for(Function::iterator bb = func->begin(), bb_end = func->end(); bb != bb_end; bb++)
                {
                    for(BasicBlock::iterator inst = bb->begin(), inst_end = bb->end(); inst != inst_end; inst++)
                    {
                        if(isa<CallInst>(inst))
                        {
                            Function * called_func = dyn_cast<CallInst>(inst)->getCalledFunction();
                            if(called_func && called_func->getName() == "hipblasGemmEx")
                            {
                                gemm_counter++;
                                if(is_in_list<int>(runned_funcID_list,gemm_counter))
                                {
                                    std::cout<<"The "<<gemm_counter<<"th GemmEx is replaced by faster_mul"<<std::endl;
                                    CallInst * call_inst = dyn_cast<CallInst>(inst);
                                    std::vector<Value*> args;
                                    unsigned int args_num = call_inst->arg_size();
                                    for(auto i = 0; i < args_num; i++)
                                    {
                                        Value * argv = call_inst->getArgOperand(i);
                                        args.push_back(argv);
                                    }

                                    IRBuilder<> builder(dyn_cast<Instruction>(inst));
                                    Value * argv_poolptr = builder.CreateLoad(gvar_ptr_mempool,"mzw_pool_ptr");
                                    if(argv_poolptr == nullptr)
                                    {
                                        errs()<<"Cannot get pool ptr for faster_mul func\n";
                                        exit(1);
                                    }
                                    args.push_back(argv_poolptr);

                                    for(int i = 0; i < 3; i++)
                                    {
                                        ConstantInt * constantInt_pi = builder.getInt32(gemm_arg_m[gemm_counter][i]);
                                        if(constantInt_pi == nullptr)
                                        {
                                            errs()<<"Cannot get the constant pi as arg\n";
                                            exit(1);
                                        }
                                        Value * argv_pi = dynamic_cast<Value*>(constantInt_pi);
                                        args.push_back(argv_pi);
                                    }

                                    Value * faster_mul_ret_v = builder.CreateCall(fastermul_func_type,fastermul_func_ptr,makeArrayRef(args));
                                    if(!faster_mul_ret_v)
                                    {
                                        errs()<<"Cannot create the call inst of faster_mul func\n";
                                        exit(1);
                                    }
                                    call_inst->replaceAllUsesWith(faster_mul_ret_v);

                                    inst_del.push_back(call_inst);
                                }
                            }
                        }
                    }
                }
            }
        }

        for(auto inst : inst_del) inst->eraseFromParent();

        return true;


    }

    void Gen::get_tool_library_func(Module& M, std::unordered_map<Function*,int> &tool_func_map)
    {
        //get all function pointer in tool-library, these function should not be scanned and optimized
        Function * tuning_gemm_func_ptr = M.getFunction("_Z17mzw_tuning_GemmExPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tiPcS3_S_");
        if(tuning_gemm_func_ptr == nullptr)
        {
            errs()<<"Cannot get the tunning fuc\n";
            exit(1);
        }
        tool_func_map[tuning_gemm_func_ptr] = 1;

        //following func ptr are only used to check, not to replace
        Function * output_dimension_func_ptr = M.getFunction("_Z25mzw_output_dimension_fileiiiPc");
        if(output_dimension_func_ptr == nullptr)
        {
            errs()<<"Cannot get the od func\n";
            exit(1);
        }
        tool_func_map[output_dimension_func_ptr] = 1;
        Function * output_FuncID_func_ptr = M.getFunction("_Z22mzw_output_FuncID_fileiPc");
        if(output_FuncID_func_ptr == nullptr)
        {
            errs()<<"Cannot get the ofid func\n";
            exit(1);
        }
        tool_func_map[output_FuncID_func_ptr] = 1;
        Function * output_ErrP_func_ptr = M.getFunction("_Z31mzw_output_Err_Performance_fileiiiifffPc");
        if(output_ErrP_func_ptr == nullptr)
        {
            errs()<<"Cannot get the oep func\n";
            exit(1);
        }
        tool_func_map[output_ErrP_func_ptr] = 1;
        Function * checkerr_func_ptr = M.getFunction("_Z23mzw_check_hipblas_error15hipblasStatus_t");
        if(checkerr_func_ptr == nullptr)
        {
            errs()<<"Cannot get the che func\n";
            exit(1);
        }
        tool_func_map[checkerr_func_ptr] = 1;
        Function * wrapper_gemm_func_ptr = M.getFunction("_Z18mzw_wrapper_GemmExPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tiPcS3_S3_");
        if(wrapper_gemm_func_ptr == nullptr)
        {
            errs()<<"Cannot get the wrapper_gemm func\n";
            exit(1);
        }
        tool_func_map[wrapper_gemm_func_ptr] = 1;
        Function * rcrp_func_ptr = M.getFunction("_Z35mzw_Result_Check_Record_PerformancePvS_iiiiiifPci");
        if(rcrp_func_ptr == nullptr)
        {
            errs()<<"Cannot get the rcrp func\n";
            exit(1);
        }
        tool_func_map[rcrp_func_ptr] = 1;
        Function * fastermul_func_ptr = M.getFunction("_Z14mzw_faster_mulPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tS_iii");
        if(fastermul_func_ptr == nullptr)
        {
            errs()<<"Cannot get the faster_mul func\n";
            exit(1);
        }
        tool_func_map[fastermul_func_ptr] = 1;
        Function * readfile_func_ptr = M.getFunction("_Z12mzw_ReadFileRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERmPvm");
        if(readfile_func_ptr == nullptr)
        {
            errs()<<"Cannot get the readfile func\n";
            exit(1);
        }
        tool_func_map[readfile_func_ptr] = 1;
        Function * writefile_func_ptr = M.getFunction("_Z13mzw_WriteFileRKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEPKvm");
        if(writefile_func_ptr == nullptr)
        {
            errs()<<"Cannot get the writefile func\n";
            exit(1);
        }
        tool_func_map[writefile_func_ptr] = 1;
        Function * devicecast_func_ptr = M.getFunction("_Z33__device_stub__mzw_fp32_cast_fp16PfP6__halfii");
        if(devicecast_func_ptr == nullptr)
        {
            errs()<<"Cannot get the device_cast func\n";
            exit(1);
        }
        tool_func_map[devicecast_func_ptr] = 1;
        devicecast_func_ptr = M.getFunction("_Z33__device_stub__mzw_fp16_cast_fp32P6__halfPfii");
        if(devicecast_func_ptr == nullptr)
        {
            errs()<<"Cannot get the device_cast func\n";
            exit(1);
        }
        tool_func_map[devicecast_func_ptr] = 1;
        Function * quan_func_ptr = M.getFunction("_Z8mzw_quanPvS_iiPf17hipblasDatatype_tb");
        if(quan_func_ptr == nullptr)
        {
            errs()<<"Cannot get quan func\n";
            exit(1);
        }
        tool_func_map[quan_func_ptr] = 1;
        Function * dequan_func_ptr = M.getFunction("_Z10mzw_dequanPvS_iiPf17hipblasDatatype_t");
        if(dequan_func_ptr == nullptr)
        {
            errs()<<"Cannot get dequan func\n";
            exit(1);
        }
        tool_func_map[dequan_func_ptr] = 1;  
        Function * checker_faster_mul_func_ptr = M.getFunction("_Z22mzw_checker_faster_mulPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tS_iiiiPcS3_");
        if(checker_faster_mul_func_ptr == nullptr)
        {
            errs()<<"Cannot get checker_faster_mul func\n";
            exit(1);
        }
        tool_func_map[checker_faster_mul_func_ptr] = 1;  
        Function * checker_fp32_gemm_func_ptr = M.getFunction("_Z18mzw_checker_GemmExPv18hipblasOperation_tS0_iiiS_S_17hipblasDatatype_tiS_S1_iS_S_S1_iS1_17hipblasGemmAlgo_tiiiiPcS3_");
        if(checker_fp32_gemm_func_ptr == nullptr)
        {
            errs()<<"Cannot get checker_fp32_gemm func\n";
            exit(1);
        }
        tool_func_map[checker_fp32_gemm_func_ptr] = 1;  
    }

    template<typename T>
    bool Gen::is_in_list(std::vector<T> l, T target)
    {
        for(T e: l)
        {
            if(target == e)
            {
                return true;
            }
        }
        return false;
    }

    void Gen::get_biggest_dimension()
    {
        //TO.DO.: read the dimension_file and get the biggest pool size             //Done
        std::fstream fs(dimension_file_path,std::ios::in);
        if(!fs)
        {
            errs()<<"Cannot open the dimension file\n";
            exit(1);
        }
        int m,n,k;
        //TO.DO.: Every variable about pool_size should be size_t type
        size_t max_size = 0;
        //update the pool_m/n/k
        while(fs>>m>>n>>k)
        {
            size_t cur_pool_size = sizeof(float)*(
                                2*m*k +                                    
                                2*k*n +                                    
                                m*n*2) +                                    
                                sizeof(short)*(
                                m*k*2 +                                    
                                k*n*2 +                                    
                                m*n*2);
            if(cur_pool_size > max_size)
            {
                pool_m = m;
                pool_n = n;
                pool_k = k;
                max_size = cur_pool_size;
            }
        }
        fs.close();
    }

    std::string Gen::get_father_path(std::string file_path)
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

    void Gen::read_all_func_args_file()
    {
        std::fstream fs(single_pass_data_file_path, std::ios::in);
        if(!fs)
        {
            std::cout<<"Cannot open the single_pass_data_file"<<std::endl;
            exit(1);
        }

        int id;
        int p[3];
        float running_time, avg_err, max_err;
        while(fs>>id>>p[0]>>p[1]>>p[2]>>running_time>>avg_err>>max_err)
        {
            if(gemm_arg_m.find(id) != gemm_arg_m.end()) continue;
            for(int i = 0; i < 3; i++)
            {
                gemm_arg_m[id][i] = p[i];
            }
        }
        fs.close();
    }

    void Gen::read_runned_funcID_file()
    {
        std::fstream fs(runned_funcID_file_path,std::ios::in);
        if(!fs)
        {
            std::cout<<"Cannot open the runned_funcID_file"<<std::endl;
            exit(1);
        }
        else
        {
            int func_id;
            std::unordered_map<int,bool> m;
            bool reorg_flag = false;
            while(fs>>func_id)
            {
                if(m.find(func_id) != m.end())
                {
                    reorg_flag = true;
                    continue;
                }
                else
                {
                    runned_funcID_list.push_back(func_id);
                    m[func_id] = true;
                }
            }
            fs.close();
            if(reorg_flag) return make_runned_func_ID_set();
        }
    }
}

char Gen::ID = 0;

static RegisterPass<Gen> X("Gen","A Pass to generate final optimized code",
                                true, false);

static RegisterStandardPasses Y(
    PassManagerBuilder::EP_EarlyAsPossible,
    [] (const PassManagerBuilder & Builder, 
    legacy::PassManagerBase & PM) {PM.add(new Gen());});
