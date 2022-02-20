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

//This Pass is to recognize every GemmEx for fp32 and output the M,N,K into one file as well as
//Replace them with wrapper function with an additional ID
//then output IDs of gemm wrapped functions which are called into ont file

/*
Module *ParseBitcodeFile(MemoryBuffer *Buffer, LLVMContext& Context,
                         std::string *ErrMsg = 0);
*/

//TO.DO.: Replace the function name in tool-library!!!!!!

namespace {
    struct MDFID : public ModulePass {
        static char ID;

        MDFID();
        ~MDFID() {};

        bool runOnModule (Module & M)override;
        bool set_dimensionfile_path(std::string module_id);
        bool set_funcIDfile_path(std::string module_id);
        bool set_MemAddrfile_path(std::string module_id);
        std::string get_father_path(std::string);
        int CountAllGemmEx(Module & M);

        //self data
        int Func_Counter;
        std::string dimension_file_path;
        std::string FuncID_run_file_path; 
        std::string Gemm_Mem_addr_file_path;
    };

    MDFID::MDFID() : ModulePass(ID)
    {
        Func_Counter = 0;
    }

    bool MDFID::set_dimensionfile_path(std::string id)          //Module ID would be the full abs path of input ll file 
    {
        dimension_file_path = id+"dimension.txt";
        errs()<<"The target dimension file path is "<<dimension_file_path<<"\n";        //TO.DO.: This path is not right.
        std::fstream df;
        df.open(dimension_file_path, std::ios::out);
        if(!df)
        {
            errs()<<"Dimension file not exist, create one\n";
            df<<"";
            df.close();
            return true;
        }
        else
        {
            //TO.DO.: Need a check to ensure it can reflush the file with original context
            errs()<<"Dimension file exists, but need to reflush it\n";
            df<<"";
            df.close();
            return true;
        }
    }

    bool MDFID::set_funcIDfile_path(std::string id)
    {
        FuncID_run_file_path = id+"calledFuncID.txt";      //TO.DO.: the id appears to be <stdin> in result. fix it 
        errs()<<"The target CalledFuncID file path is "<<FuncID_run_file_path<<"\n";
        std::fstream df;
        df.open(FuncID_run_file_path, std::ios::out);
        if(!df)
        {
            errs()<<"CalledFuncID file not exist, create one\n";
            df<<"";
            df.close();
            return true;
        }
        else
        {
            //TO.DO.: Need a check to ensure it can reflush the file with original context
            errs()<<"CalledFuncID file exists, but need to reflush it\n";
            df<<"";
            df.close();
            return true;
        }
    }

    std::string MDFID::get_father_path(std::string file_path)
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

    int MDFID::CountAllGemmEx(Module & M)
    {
        int res = 0;

        //called wrapper function to replace _Z6GemmExiiiiii function
        Function * wrapper_GemmEx_func_ptr = M.getFunction("_Z18mzw_wrapper_GemmExP15_rocblas_handle18rocblas_operation_S1_iiiPvS2_17rocblas_datatype_iS2_S3_iS2_S2_S3_iS2_S3_iS3_18rocblas_gemm_algo_ijiiPcS5_S5_");
        if(wrapper_GemmEx_func_ptr == nullptr)
        {
            errs()<<"Cannot get the wGEf func\n";
            exit(1);
        }

        Function * faster_Gemm_func_ptr = M.getFunction("_Z14mzw_faster_mulP15_rocblas_handle18rocblas_operation_S1_iiiPvS2_17rocblas_datatype_iS2_S3_iS2_S2_S3_iS2_S3_iS3_18rocblas_gemm_algo_ijS2_iii");
        if(faster_Gemm_func_ptr == nullptr)
        {
            errs()<<"Cannot get the fGfp func\n";
            exit(1);
        }
        Function * tuning_GemmEx_func_ptr = M.getFunction("_Z17mzw_tuning_GemmExP15_rocblas_handle18rocblas_operation_S1_iiiPvS2_17rocblas_datatype_iS2_S3_iS2_S2_S3_iS2_S3_iS3_18rocblas_gemm_algo_ijiiPcS5_S2_");
        if(tuning_GemmEx_func_ptr == nullptr)
        {
            errs()<<"Cannot get the tuning gemm ptr\n";
            exit(1);
        }
        Function * checker_GemmEx_func_ptr = M.getFunction("_Z18mzw_checker_GemmExP15_rocblas_handle18rocblas_operation_S1_iiiPvS2_17rocblas_datatype_iS2_S3_iS2_S2_S3_iS2_S3_iS3_18rocblas_gemm_algo_ijiiiiiPcS5_");
        if(checker_GemmEx_func_ptr == nullptr)
        {
            errs()<<"Cannot get the tuning gemm ptr\n";
            exit(1);
        }


        for(Module::FunctionListType::iterator func = M.getFunctionList().begin(),
            end_Func = M.getFunctionList().end(); func != end_Func; func++)
        {
            //Function::iterator == BasicBlockListType::iterator
            //errs()<<"Now we are facing declare of function "<<func->getName()<<"\n";
            for(Function::iterator bb = func->begin(); bb != func->end(); bb++)
            {
                for(BasicBlock::iterator inst = bb->begin(); inst != bb->end(); inst++)
                {
                    if(isa<CallInst>(inst))
                    {
                        //QUES.: What about InvokeInst which means indirect call?
                        Function * called_func = dyn_cast<CallInst>(inst)->getCalledFunction();
                        if(called_func && called_func->getName() == "rocblas_gemm_ex" && dyn_cast<Instruction>(inst)->getParent()->getParent() != faster_Gemm_func_ptr
                            && dyn_cast<Instruction>(inst)->getParent()->getParent() != wrapper_GemmEx_func_ptr
                            && dyn_cast<Instruction>(inst)->getParent()->getParent() != tuning_GemmEx_func_ptr
                            && dyn_cast<Instruction>(inst)->getParent()->getParent() != checker_GemmEx_func_ptr)   //TO.DO.: cmp to all func in tool_library containing GemmEx!                                                                     //TO.DO.: Need to ensure the GemmEx in tool_library will not be modified
                                                                                                                                                                                                                    //ANS.: Done by comparing its caller function to the faster_mul in tool-library
                        {
                            res++;
                        }
                    }
                }
            }
        }
        
        return res;
    }

    bool MDFID::runOnModule(Module & M)
    {
        //Set up the output file
        std::string ModuleName = M.getModuleIdentifier();
        //std::cout<<"Module name is "<<ModuleName<<std::endl;                //TO.DO.: The modulename is <stdin>. why?       ANS.: Dont use < in command line in opt. just pass the filename and the Module name will be the abs path of input file module
        //TO.DO.: Make the file path name configurable for user, because this file name is used in pass2/3,
        //If they all use file name based on ModuleName, it cannot work.

        //TMP. FIX.: extract the father direc of modulename, and add filename behind it.
        std::string father_path = get_father_path(ModuleName);

        if(!set_dimensionfile_path(father_path))
        {
            errs()<<"Wrong when setting dimension file path \n";
            exit(1);
        }
        if(!set_funcIDfile_path(father_path))
        {
            errs()<<"Wrong when setting funcID file path \n";
            exit(1);
        }

        //TO.DO.: Create the last_pass_seperate_error_file 

        //Loop over all instructions, if one is call GemmEx, add one IR right behind it to output its dimension
        //QUES.: Can I use getFunction to locate all GemmEx functions?
        //NOTE: We cannot use the result of M.getFunctionList() outside the current line
        //const Module::FunctionListType & func_list = M.getFunctionList();

        //The tool to generate IR
        //IRBuilder<> builder(M.getContext());

        //get the inserted function ptr
        //TO.DO.: Maybe we can use Module::getOrInsertFunction() to get function ptr;
        //Function * output_dimension_func_ptr = M.getFunction("output_dimension_file");            //This cannot get the function
        /*
        std::string mangled_output_dimension_func_name;
        raw_string_ostream mangled_output_dimension_func_stream(mangled_output_dimension_func_name);
        Mangler::getNameWithPrefix(mangled_output_dimension_func_stream,"output_dimension_file",M.getDataLayout());
        std::string new_name = mangled_output_dimension_func_stream.str();
        */

        //TO.DO.: Find a way to locate function without using the mangled name.（All these function are from tool_library)

        Function * output_dimension_func_ptr = M.getFunction(StringRef("_Z25mzw_output_dimension_fileiiiPc"));
        if(output_dimension_func_ptr == nullptr)
        {
            errs()<<"Cannot get the odf func\n";
            exit(1);
        }
        Function * output_CalledFuncID_func_ptr = M.getFunction("_Z22mzw_output_FuncID_fileiPc");
        if(output_CalledFuncID_func_ptr == nullptr)
        {
            errs()<<"Cannot get the ocfidf func\n";
            exit(1);
        }
        //called wrapper function to replace _Z6GemmExiiiiii function
        Function * wrapper_GemmEx_func_ptr = M.getFunction("_Z18mzw_wrapper_GemmExP15_rocblas_handle18rocblas_operation_S1_iiiPvS2_17rocblas_datatype_iS2_S3_iS2_S2_S3_iS2_S3_iS3_18rocblas_gemm_algo_ijiiPcS5_S5_");
        if(wrapper_GemmEx_func_ptr == nullptr)
        {
            errs()<<"Cannot get the wGEf func\n";
            exit(1);
        }

        Function * faster_Gemm_func_ptr = M.getFunction("_Z14mzw_faster_mulP15_rocblas_handle18rocblas_operation_S1_iiiPvS2_17rocblas_datatype_iS2_S3_iS2_S2_S3_iS2_S3_iS3_18rocblas_gemm_algo_ijS2_iii");
        if(faster_Gemm_func_ptr == nullptr)
        {
            errs()<<"Cannot get the fGfp func\n";
            exit(1);
        }
        Function * tuning_GemmEx_func_ptr = M.getFunction("_Z17mzw_tuning_GemmExP15_rocblas_handle18rocblas_operation_S1_iiiPvS2_17rocblas_datatype_iS2_S3_iS2_S2_S3_iS2_S3_iS3_18rocblas_gemm_algo_ijiiPcS5_S2_");
        if(tuning_GemmEx_func_ptr == nullptr)
        {
            errs()<<"Cannot get the tuning gemm ptr\n";
            exit(1);
        }
        Function * checker_GemmEx_func_ptr = M.getFunction("_Z18mzw_checker_GemmExP15_rocblas_handle18rocblas_operation_S1_iiiPvS2_17rocblas_datatype_iS2_S3_iS2_S2_S3_iS2_S3_iS3_18rocblas_gemm_algo_ijiiiiiPcS5_");
        if(checker_GemmEx_func_ptr == nullptr)
        {
            errs()<<"Cannot get the tuning gemm ptr\n";
            exit(1);
        }

        //Value * arg_CalledFuncID_file_path = builder.CreateGlobalStringPtr(StringRef(FuncID_run_file_path));
        //FunctionType * output_func_type = FunctionType::get(FunctionType::getVoidTy(M.getContext()),false);     //getVoidTy is a method in Class Type         NOTE.: Dont fucking use this to get the void type. But why??
        FunctionType * wrapper_GemmEx_func_type = wrapper_GemmEx_func_ptr->getFunctionType();

        
        //delcare global run counter for each GEMM
        //first: get number of all gemmex
        int gemm_num = CountAllGemmEx(M);

        Type * Int32Type = Type::getInt32Ty(M.getContext());
        std::vector<GlobalVariable *> run_counters;
        run_counters.push_back(nullptr);
        for(int i = 1; i <= gemm_num; i++)
        {
            GlobalVariable * run_counter = new GlobalVariable(M,Int32Type,false,GlobalValue::CommonLinkage,0,"mzw_rc"+std::to_string(i));
            Constant * Zero_Value = Constant::getNullValue(Int32Type);
            run_counter->setInitializer(Zero_Value);
            run_counters.push_back(run_counter);
        }

        std::vector<Instruction*> inst_del;

        int gemm_id = 0;

        //SymbolTableList<Function> == FunctionListType
        for(Module::FunctionListType::iterator func = M.getFunctionList().begin(),
            end_Func = M.getFunctionList().end(); func != end_Func; func++)
        {
            //Function::iterator == BasicBlockListType::iterator
            errs()<<"Now we are facing declare of function "<<func->getName()<<"\n";
            for(Function::iterator bb = func->begin(); bb != func->end(); bb++)
            {
                for(BasicBlock::iterator inst = bb->begin(); inst != bb->end(); inst++)
                {
                    if(isa<CallInst>(inst))
                    {
                        //QUES.: What about InvokeInst which means indirect call?
                        Function * called_func = dyn_cast<CallInst>(inst)->getCalledFunction();
                        if(called_func && called_func->getName() == "rocblas_gemm_ex" && dyn_cast<Instruction>(inst)->getParent()->getParent() != faster_Gemm_func_ptr
                            && dyn_cast<Instruction>(inst)->getParent()->getParent() != wrapper_GemmEx_func_ptr
                            && dyn_cast<Instruction>(inst)->getParent()->getParent() != tuning_GemmEx_func_ptr
                            && dyn_cast<Instruction>(inst)->getParent()->getParent() != checker_GemmEx_func_ptr)   //TO.DO.: cmp to all func in tool_library containing GemmEx!                                                                     //TO.DO.: Need to ensure the GemmEx in tool_library will not be modified
                                                                                                                                                                                                                    //ANS.: Done by comparing its caller function to the faster_mul in tool-library
                        {
                            errs()<<"we now get the gemm func\n";
                            gemm_id++;
                            CallInst * call_inst = dyn_cast<CallInst>(inst);            //Only in this way can we get the true arguments used in callinst

                            //Dont directly use errs() to print, use a print_func in self-made library(which can use errs() inside)
                            //We only need to replace the original gemmex with a wrapper_gemmex which includes the ouput function inside
                            IRBuilder<> builder(dyn_cast<Instruction>(inst));
                            std::vector<Value*> args;

                            char dimension_file_path_char[200];
                            strcpy(dimension_file_path_char,dimension_file_path.c_str());
                            Value * argv_dimension_file_path = builder.CreateGlobalStringPtr(dimension_file_path_char);
                            
                            char funcID_file_path_char[200];
                            strcpy(funcID_file_path_char,FuncID_run_file_path.c_str());
                            Value * argv_funcID_file_path = builder.CreateGlobalStringPtr(funcID_file_path_char);
                            
                            ConstantInt * ID = builder.getInt32(gemm_id);
                            Value * argv_ID = dynamic_cast<Value*>(ID);
                            
                            std::string Matrix_File_Name = father_path + "GEMM_" + std::to_string(gemm_id);         //TODO.: It needs to fullfilled by runtime counter and ".bin" in implementation of rocblas_tool.hip
                            char matrix_file_path_char[200];
                            strcpy(matrix_file_path_char,Matrix_File_Name.c_str());
                            Value * argv_matrix_file_path = builder.CreateGlobalStringPtr(matrix_file_path_char);

                            Value * rc_v = builder.CreateLoad(run_counters[gemm_id]);
                            if(rc_v == nullptr)
                            {
                                errs()<<"Cannot load runtime counter for gemm_"<<gemm_id<<"\n";
                                exit(1);
                            }
                            ConstantInt * Immediate_One = builder.getInt32(1);
                            Value * added_rc_v = builder.CreateAdd(rc_v,dyn_cast<Value>(Immediate_One));
                            if(!builder.CreateStore(added_rc_v,run_counters[gemm_id]))
                            {
                                errs()<<"Cannot create store inst for added rc value\n";
                                exit(1);
                            }

                            //replace the function
                            //get original args
                            unsigned int arg_num = call_inst->arg_size();
                            for(unsigned int i = 0; i < arg_num; i++)
                            {
                                //Value * arg = dynamic_cast<Value*>(call_inst->getArgOperand(i));
                                Value * arg = call_inst->getArgOperand(i);
                                if(arg == nullptr)
                                {
                                    std::cout<<"The "<<i<<"th argument is nullptr"<<std::endl;
                                    exit(1);
                                }                                 
                                Value * argv = dynamic_cast<Value*>(arg);                           //TO.DO.: For hipblashandle, because we want a reference to the original handle, so we need to transform it to a reference
                                args.push_back(argv);
                            }
                            //add new args
                            args.push_back(argv_ID);
                            args.push_back(added_rc_v);
                            args.push_back(argv_dimension_file_path);
                            args.push_back(argv_funcID_file_path);
                            args.push_back(argv_matrix_file_path);
                            Value * wrapper_GemmEx_ret_v = builder.CreateCall(wrapper_GemmEx_func_type,wrapper_GemmEx_func_ptr,makeArrayRef(args));
                            if(!wrapper_GemmEx_ret_v)                                               //TO.DO.: The return value should replace all uses of ret-val of original gemmex
                            {
                                errs()<<"Cannot create the call inst of wrapper_GemmEx_func\n";
                                exit(1);
                            }
                            call_inst->replaceAllUsesWith(wrapper_GemmEx_ret_v);

                            //delete the original func
                            //(call_)inst->eraseFromParent() will cause Assertion `Val && "isa<> used on a null pointer"' failed.
                            //We need a container to store all these need-to-delete functions and delete them later
                            inst_del.push_back(call_inst);
                            
                        }
                        else
                        {
                            //errs()<<"This is not a call to function or a GemmEx function\n";
                        }
                    }
                }
            }
        }

        //delete all replaced function
        for(auto inst : inst_del)
        {
            inst->eraseFromParent();
        }

        return true;
    }

}


char MDFID::ID = 0;

static RegisterPass<MDFID> X("mdfid","A Pass to Mark GEMM Dimension and IDs of Called GEMM Functions",
                                true, false);

static RegisterStandardPasses Y(
    PassManagerBuilder::EP_EarlyAsPossible,
    [] (const PassManagerBuilder & Builder, 
    legacy::PassManagerBase & PM) {PM.add(new MDFID());});
