#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include<chrono>
#include<iostream>
#include <hip/hip_runtime.h>
#include <rocblas.h>
#include "hip/hip_fp16.h"
#include <cstdio>
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include<math.h>

/*
Hardware Info
SM的数量：120
每个线程块的共享内存大小：64 KB
每个线程块的最大线程数：1024
每个SM的最大线程数：2560
每个SM的最大线程束数：80        //Assume 32 threads in a wrap

/mnt/sdb1/public/yum/env/v1/spack/opt/spack/linux-centos7-zen/gcc-8.5.0/rocblas-4.3.1-fhdrqvdjmkc7jrobubifh2mgvg7se4rc
*/

//TO.DO.: FIX the call arguments of rocblas_gemm_ex

//forward declare
rocblas_status mzw_faster_mul(rocblas_handle blas_handle, rocblas_operation Trans_A, rocblas_operation Trans_B, int m, int n, int k, void * alpha_addr,
                    void * device_fp32MatrixA, rocblas_datatype Atype, int ldA, void * device_fp32MatrixB, rocblas_datatype Btype, int ldB, void * beta_addr,
                    void * device_fp32MatrixC, rocblas_datatype Ctype, int ldC, void * device_fp32MatrixD, rocblas_datatype Dtype, int ldD,
                    rocblas_datatype computeType, rocblas_gemm_algo algo, int32_t solution_index, uint32_t flags,
                    void * memory_pool, int p1, int p2, int p3);

void mzw_Result_Check_Record_Performance(void * A, void * B, int m, int n, int p1, int p2, int p3, int GEMM_ID, int runtime_counter, float running_time, char * file_path, int flag);

bool mzw_WriteFile(const std::string &filePath, const void *buffer, size_t size);

char *mzw_ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize);

#define HIP_CHECK(x)                        \
    if(x != hipSuccess)                     \
    {                                       \
        std::cout<<"Wrong HIP"<<std::endl;  \
        std::cout<<mzw_check_hip_error(x)<<std::endl;  \
        exit(1);                            \
    }

#define ROC_BLAS_CHECK(x)                           \
    if(x != rocblas_status_success)                 \
    {                                               \
        std::cout<<"Wrong ROCBLAS: ";              \
        std::cout<<mzw_check_rocblas_error(x)<<std::endl; \
        exit(1);                                    \
    }

void mzw_output_dimension_file(int M, int N, int K, char * path)
{
    //std::string path = "/mnt/data/home/mzw/workspace/test_space/llvm_test/mark_test/dimension_file.txt";
    std::fstream fs(path,std::ios::app);
    if(!fs)
    {
        std::cout<<"Cant open the dimension file"<<std::endl;
        exit(1);
    }
    else
    {
        fs<<M<<" "<<N<<" "<<K<<std::endl;
        fs.close();
    }
}

void mzw_output_FuncID_file(int id, char * path)
{
    //std::string path = "/mnt/data/home/mzw/workspace/test_space/llvm_test/mark_test/FuncID_file.txt";
    std::fstream fs(path,std::ios::app);
    if(!fs)
    {
        std::cout<<"Cant open the funcID file"<<std::endl;
        exit(1);
    }
    else
    {
        fs<<id<<std::endl;
        fs.close();
    }
}

//TO.DO.: Does it make sense to judge dependency according to printed in-time address?
//void mzw_output_matrix_mem_addr_file(int id, )

//This is used when reading the old single_pass_file
void mzw_flush_single_pass_file(char * file_path)
{
    std::fstream fs(file_path,std::ios::out | std::ios::trunc);
    if(!fs)
    {
        std::cout<<"Cant open the funcID file"<<std::endl;
        exit(1);
    }
    else
    {
        fs<<""<<std::endl;
        fs.close();
    }
}

void mzw_output_Err_Performance_file(int id, int runtime_counter, int p1, int p2, int p3, float time, float avg_err, float max_err, char * path)
{
    std::fstream fs(path,std::ios::app);
    if(!fs)
    {
        std::cout<<"Cant open the Err_Performance file"<<std::endl;
        exit(1);
    }
    else
    {
        fs<<id<<" "<<runtime_counter<<" "<<p1<<" "<<p2<<" "<<p3<<" "<<time<<" "<<avg_err<<" "<<max_err<<std::endl;
        fs.close();
    }
}

const char * mzw_check_hip_error(hipError_t x)
{
    switch(x)
    {
        case hipErrorInvalidContext:
            return "hipErrorInvalidContext";
        case hipErrorInvalidKernelFile:
            return "hipErrorInvalidKernelFile";
        case hipErrorMemoryAllocation:
            return "hipErrorMemoryAllocation";
        case hipErrorInitializationError:
            return "hipErrorInitializationError";
        case hipErrorLaunchFailure:
            return "hipErrorLaunchFailure";
        case hipErrorLaunchOutOfResources:
            return "hipErrorLaunchOutOfResources";
        case hipErrorInvalidDevice:
            return "hipErrorInvalidDevice";
        case hipErrorInvalidValue:
            return "hipErrorInvalidValue";
        case hipErrorInvalidDevicePointer:
            return "hipErrorInvalidDevicePointer";
        case hipErrorInvalidMemcpyDirection:
            return "hipErrorInvalidMemcpyDirection";
        case hipErrorUnknown:
            return "hipErrorUnknown";
        case hipErrorInvalidResourceHandle:
            return "hipErrorInvalidResourceHandle";
        case hipErrorNotReady:
            return "hipErrorNotReady";
        case hipErrorNoDevice:
            return "hipErrorNoDevice";
        case hipErrorPeerAccessAlreadyEnabled:
            return "hipErrorPeerAccessAlreadyEnabled";
        case hipErrorPeerAccessNotEnabled:
            return "hipErrorPeerAccessNotEnabled";
        case hipErrorRuntimeMemory:
            return "hipErrorRuntimeMemory";
        case hipErrorRuntimeOther:
            return "hipErrorRuntimeOther";
        case hipErrorHostMemoryAlreadyRegistered:
            return "hipErrorHostMemoryAlreadyRegistered";
        case hipErrorHostMemoryNotRegistered:
            return "hipErrorHostMemoryNotRegistered";
        case hipErrorMapBufferObjectFailed:
            return "hipErrorMapBufferObjectFailed";
        case hipErrorTbd:
            return "hipErrorTbd";
        default:
            return "unknown error";
    }
    
}

const char * mzw_check_rocblas_error(rocblas_status error)
{
    switch (error)
    {
        case rocblas_status_success:
            return "rocblas_status_success";

        case rocblas_status_invalid_handle:
            return "rocblas_status_invalid_handle";

        case rocblas_status_not_implemented:
            return "rocblas_status_not_implemented";

        case rocblas_status_invalid_pointer:
            return "rocblas_status_invalid_pointer";

        case rocblas_status_invalid_size:
            return "rocblas_status_invalid_size";

        case rocblas_status_memory_error:
            return "rocblas_status_memory_error";

        case rocblas_status_internal_error:
            return "rocblas_status_internal_error";

        case rocblas_status_perf_degraded:
            return "rocblas_status_perf_degraded";
        
        case rocblas_status_size_query_mismatch:
            return "rocblas_status_size_query_mismatch";
        
        case rocblas_status_size_increased:
            return "rocblas_status_size_increased";

        case rocblas_status_size_unchanged:
            return "rocblas_status_size_unchanged";
        
        case rocblas_status_invalid_value:
            return "rocblas_status_invalid_value";
        
        case rocblas_status_continue:
            return "rocblas_status_continue";
        
        default:
            return "rocblas_status_check_numerics_fail";
    }

    return "<unknown>";
}

//to replace FP32 GemmEx in pass 1
//TO.DO.: Output the memory address of three matrix
rocblas_status mzw_wrapper_GemmEx(rocblas_handle handle, rocblas_operation TA, rocblas_operation TB, int m, int n, int k, void * alpha_addr, 
                                    void * MatrixA, rocblas_datatype Atype, int ldA, void * MatrixB, rocblas_datatype Btype, int ldB, void * beta_addr,
                                    void * MatrixC, rocblas_datatype Ctype, int ldC, void * MatrixD, rocblas_datatype Dtype, int ldD,
                                    rocblas_datatype computeType, rocblas_gemm_algo algo, int32_t solution_index, uint32_t flags, int Gemm_ID, int runtime_counter,
                                    char * dimension_file_path, char * funcID_file_path, char * Matrix_file_name)                   //QUES.: If I want to use a handle reference, what should I do?
{
    mzw_output_dimension_file(m,n,k,dimension_file_path);
    mzw_output_FuncID_file(Gemm_ID,funcID_file_path);
    rocblas_status s = rocblas_gemm_ex(handle,TA,TB,m,n,k,alpha_addr,MatrixA,Atype,ldA,MatrixB,Btype,ldB,beta_addr,MatrixC,Ctype,ldC,MatrixC,Ctype,ldC,computeType,algo,solution_index,flags);
    hipDeviceSynchronize();
    size_t MatrixC_size = sizeof(float)*m*n;
    void * host_MatrixC = (void*)malloc(MatrixC_size);
    HIP_CHECK(hipMemcpy(host_MatrixC,MatrixC,MatrixC_size,hipMemcpyDeviceToHost));
    
    std::string Matrix_file_path = {Matrix_file_name};
    Matrix_file_path += "_"+std::to_string(runtime_counter)+".bin";
    //std::cout<<Matrix_file_path<<std::endl;
    mzw_WriteFile(Matrix_file_path,host_MatrixC,MatrixC_size);
    free(host_MatrixC);
    return s;
}

//to replace FP32 GemmEx in pass 2
rocblas_status mzw_checker_GemmEx(rocblas_handle handle, rocblas_operation TA, rocblas_operation TB, int m, int n, int k, void * alpha_addr, 
                                    void * MatrixA, rocblas_datatype Atype, int ldA, void * MatrixB, rocblas_datatype Btype, int ldB, void * beta_addr,
                                    void * MatrixC, rocblas_datatype Ctype, int ldC, void * MatrixD, rocblas_datatype Dtype, int ldD, 
                                    rocblas_datatype computeType, rocblas_gemm_algo algo, int32_t solution_index, uint32_t flags, int Gemm_ID, int runtime_counter,
                                    int p1, int p2, int p3, char * Matrix_file_name, char * single_pass_error_file_path)        //The single pass error file should be read before we run the pass, and delete all its data after reading
{
    float * fp32_result_matrix = (float*)malloc(sizeof(float)*m*n);
    //run the faster_mul
    hipEvent_t start,end;
    hipEventCreate(&start);
    hipEventCreate(&end);
    float time;
    hipEventRecord(start);
    rocblas_status s = rocblas_gemm_ex(handle, TA, TB, m, n, k, alpha_addr, 
                                        MatrixA, Atype, ldA, MatrixB, Btype, ldB, beta_addr,
                                        MatrixC, Ctype, ldC, MatrixD, Dtype, ldD,
                                        computeType, algo, solution_index, flags);
    size_t size;
    hipEventRecord(end);
    hipEventSynchronize(end);
    hipEventElapsedTime(&time, start, end);

    //check the result and output it
    
    std::string Matrix_file_path = {Matrix_file_name};
    Matrix_file_path += "_"+std::to_string(runtime_counter)+".bin";
    mzw_ReadFile(Matrix_file_path,size,fp32_result_matrix,sizeof(float)*m*n);
    mzw_Result_Check_Record_Performance((void*)MatrixC,(void*)fp32_result_matrix,m,n,p1,p2,p3,Gemm_ID,runtime_counter,time,single_pass_error_file_path,0);

    free(fp32_result_matrix);
    return s;
}

//for those optimized faster_mul with result checker, used in pass 2
rocblas_status mzw_checker_faster_mul(rocblas_handle handle, rocblas_operation TA, rocblas_operation TB, int m, int n, int k, void * alpha_addr, 
                                        void * MatrixA, rocblas_datatype Atype, int ldA, void * MatrixB, rocblas_datatype Btype, int ldB, void * beta_addr,
                                        void * MatrixC, rocblas_datatype Ctype, int ldC, void * MatrixD, rocblas_datatype Dtype, int ldD,
                                        rocblas_datatype computeType, rocblas_gemm_algo algo, int32_t solution_index, uint32_t flags, void * memory_pool, int Gemm_ID, int runtime_counter, 
                                        int p1, int p2, int p3, char * Matrix_file_name, char * single_pass_error_file_path)        //The single pass error file should be read before we run the pass, and delete all its data after reading
{
    float * fp32_result_matrix = (float*)malloc(sizeof(float)*m*n);
    //run the faster_mul
    hipEvent_t start,end;
    hipEventCreate(&start);
    hipEventCreate(&end);
    float time;
    hipEventRecord(start);
    rocblas_status s = mzw_faster_mul(handle, TA, TB, m, n, k, alpha_addr, 
                                        MatrixA, Atype, ldA, MatrixB, Btype, ldB, beta_addr,
                                        MatrixC, Ctype, ldC, MatrixD, Dtype, ldD, computeType, algo, solution_index, flags,
                                        memory_pool, p1, p2, p3);
    hipEventRecord(end);
    hipEventSynchronize(end);
    hipEventElapsedTime(&time, start, end);

    //check the result and output it
    size_t size;
    
    std::string Matrix_file_path = {Matrix_file_name};
    Matrix_file_path += "_"+std::to_string(runtime_counter)+".bin";
    mzw_ReadFile(Matrix_file_path,size,fp32_result_matrix,sizeof(float)*m*n);
    mzw_Result_Check_Record_Performance((void*)MatrixC,(void*)fp32_result_matrix,m,n,p1,p2,p3,Gemm_ID,runtime_counter,time,single_pass_error_file_path,0);

    free(fp32_result_matrix);
    return s;
}


//to try optimizing parameters of faster_mul in pass 2
rocblas_status mzw_tuning_GemmEx(rocblas_handle handle, rocblas_operation TA, rocblas_operation TB, int m, int n, int k, void * alpha_addr, 
                                void * MatrixA, rocblas_datatype Atype, int ldA, void * MatrixB, rocblas_datatype Btype, int ldB, void * beta_addr,
                                void * MatrixC, rocblas_datatype Ctype, int ldC, void * MatrixD, rocblas_datatype Dtype, int ldD,
                                rocblas_datatype computeType, rocblas_gemm_algo algo, int32_t solution_index, uint32_t flags, int Gemm_ID, int runtime_counter,
                                char * optimized_path, char * Matrix_file_name, void * memory_pool)        //file_path was handled by opt pass 
{
    void * original_MatrixC = nullptr;
    void * result_Matrix = nullptr;

    /*
    //For m=n=k=16394, pool is 9GB
    size_t pool_size = sizeof(float)*(
        2*m*k +                  //for hp_matrix A / RA
        2*k*n +                  //for hp_matrix B / RB
        m*n*2) +                                    //for hp_matrix C / TMP
    sizeof(unsigned short)*(
        m*k + m*k +              //for lp_matrix A / RA
        k*n + k*n +              //for lp_matrix B / RB
        m*n*2);                                     //for lp_matrix C / TMP
    HIP_CHECK(hipMalloc((void**)&memory_pool,pool_size));
    */

    size_t st_m = m;
    size_t st_n = n;
    void * original_matrixC = (void*)malloc(sizeof(float)*st_m*st_n);
    void * result_matrix = (void*)malloc(sizeof(float)*st_m*st_n);
    HIP_CHECK(hipMemcpy(original_matrixC,MatrixC,sizeof(float)*m*n,hipMemcpyDeviceToHost));

    void * fp32_result_matrix = (void*)malloc(sizeof(float)*m*n);

    float alpha = (float)*(float*)alpha_addr;
    float beta = (float)*(float*)beta_addr;
    
    hipEvent_t start,end;
    hipEventCreate(&start);
    hipEventCreate(&end);
    float time;

    hipEventRecord(start);
    rocblas_status original_status = rocblas_gemm_ex(handle,TA,TB,m,n,k,alpha_addr,MatrixA,Atype,ldA,MatrixB,Btype,ldB,
                                                        beta_addr,MatrixC,Ctype,ldC,MatrixD,Dtype,ldD,computeType,algo,solution_index,flags);
    hipEventRecord(end);
    hipEventSynchronize(end);
    hipEventElapsedTime(&time, start, end);
    size_t size;

    HIP_CHECK(hipMemcpy(result_matrix,MatrixC,sizeof(float)*st_m*st_n,hipMemcpyDeviceToHost));

    std::string Matrix_file_path = {Matrix_file_name};
    Matrix_file_path += "_"+std::to_string(runtime_counter)+".bin";
    mzw_ReadFile(Matrix_file_path,size,fp32_result_matrix,sizeof(float)*st_m*st_n);
    mzw_Result_Check_Record_Performance((void*)MatrixC,(void*)fp32_result_matrix,m,n,-1,-1,-1,Gemm_ID,runtime_counter,time,optimized_path,0);

    //NOTE: Dont mess around the variable k presenting the dimention of Matrix  
    for(int x = 0 ; x <= 1; x++)
    {
        for(int y = 0; y <= 1; y++)
        {
            for(int z = 0; z <= 1; z++)
            {
                //std::cout<<"x: "<<x<<" y: "<<y<<" z: "<<z<<std::endl;
                hipMemcpy(MatrixC,original_matrixC,sizeof(float)*st_m*st_n,hipMemcpyHostToDevice);
                hipDeviceSynchronize();
                hipEventRecord(start);
                mzw_faster_mul(handle,TA,TB,m,n,k,alpha_addr,MatrixA,Atype,ldA,
                                MatrixB,Btype,ldB,beta_addr,MatrixC,Ctype,ldC,MatrixC,Ctype,ldC,
                                computeType,algo,solution_index,flags,memory_pool,x,y,z);
                hipEventRecord(end);
                hipEventSynchronize(end);
                hipEventElapsedTime(&time, start, end);
                mzw_Result_Check_Record_Performance((void*)MatrixC,(void*)fp32_result_matrix,m,n,x,y,z,Gemm_ID,runtime_counter,time,optimized_path,0);
            }
        }
    }

    HIP_CHECK(hipMemcpy(MatrixC,result_matrix,sizeof(float)*st_m*st_n,hipMemcpyHostToDevice));

    //hipFree(original_MatrixC);
    //hipFree(result_Matrix);
    hipEventDestroy(start);
    hipEventDestroy(end);
    free(original_matrixC);
    free(result_matrix);
    free(fp32_result_matrix);
    return original_status;
}


//function to check ERROR between low precision GEMM and fp32 GEMM
void mzw_Result_Check_Record_Performance(void * A, void * B, int m, int n, int p1, int p2, int p3, int GEMM_ID, int runtime_counter,
                                        float running_time, char * file_path, int flag)      //avg_err = (A-B)/B
{
    if(flag == 1)
    {
        //both A and B are on device
        float * host_m1 = (float*)malloc(sizeof(float)*m*n);
        float * host_m2 = (float*)malloc(sizeof(float)*m*n);
        hipMemcpy(host_m1,(float*)A,sizeof(float)*m*n,hipMemcpyDeviceToHost);
        hipMemcpy(host_m2,(float*)B,sizeof(float)*m*n,hipMemcpyDeviceToHost);
        float err = 0;
        float max_err = 0;
        for(int i = 0; i < m*n; i ++)
        {
            float cur_err = host_m1[i] - host_m2[i];
            if(cur_err > max_err || max_err < 0-cur_err)
            {
                max_err = cur_err > 0 ? cur_err : 0-cur_err;
            }
            cur_err /= host_m2[i];
            if(cur_err <0) err -= cur_err;
            else err += cur_err;    
        }
        err /= (float)(m*n);
    
        free(host_m1);
        free(host_m2);
    
        //write back to file
        mzw_output_Err_Performance_file(GEMM_ID,runtime_counter,p1,p2,p3,running_time,err,max_err,file_path);
    }
    else if(flag == 0)
    {
        //B is on host, A is on device
        float * host_m1 = (float*)malloc(sizeof(float)*m*n);
        float * host_m2 = (float*)B;
        hipMemcpy(host_m1,(float*)A,sizeof(float)*m*n,hipMemcpyDeviceToHost);
        float err = 0;
        float fenmu = 0;
        float max_err = 0;
        for(int i = 0; i < m*n; i ++)
        {
            if(!(host_m1[i] == host_m1[i]))
            {
                std::cout<<"NAN in matrix A"<<std::endl;
                exit(1);
            }
            if(!(host_m2[i] == host_m2[i]))
            {
                std::cout<<"NAN in matrix B"<<std::endl;
                exit(1);
            }
            float cur_err = host_m1[i] - host_m2[i];
            if(cur_err > max_err || max_err < 0-cur_err)
            {
                max_err = cur_err > 0 ? cur_err : 0-cur_err;
            }
            fenmu += host_m2[i] * host_m2[i];
            err += cur_err * cur_err;
        }
        err = sqrt(err)/sqrt(fenmu);
    
        free(host_m1);
    
        //write back to file
        mzw_output_Err_Performance_file(GEMM_ID,runtime_counter,p1,p2,p3,running_time,err,max_err,file_path);
    }
    
}

//read matrix data
char *mzw_ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize)
{
    //std::cout<<"Begin to read file"<<std::endl;
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        std::cout<<"failed to get file"<<std::endl;
        return nullptr;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        std::cout<<filePath.c_str()<<" is not a file, please enter a file"<<std::endl;
        return nullptr;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        std::cout<<"Open file failed. path "<<filePath.c_str()<<std::endl;
        return nullptr;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        std::cout<<"file size is 0"<<std::endl;
        file.close();
        return nullptr;
    }
    if (size > bufferSize) {
        std::cout<<"file size = "<<size<<" is larger than buffer size = "<<bufferSize<<std::endl;
        file.close();
        return nullptr;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    //std::cout<<"Finish read file"<<std::endl;
    return static_cast<char *>(buffer);
}

bool mzw_WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        //ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    FILE *outputFile = fopen(filePath.c_str(), "wb");
    if (outputFile == nullptr) {
        //ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    fwrite(buffer, size, sizeof(char), outputFile);
    fclose(outputFile);

    return true;
}


__global__ void mzw_fp32_cast_fp16(float * __restrict__ src_ptr, __half * __restrict__ dst_ptr,int m, int n)
{
    /*
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int start = (row * n + col)*1024;
    for(int i = 0; i < 1024; i++)
    {   
        dst_ptr[start + i] = __float2half_rn(src_ptr[start + i]);
    }
    */
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    for(; i < m*n; i += gridDim.x * blockDim.x)
    {
        dst_ptr[i] = __float2half_rn(src_ptr[i]);
    }
}

__global__ void mzw_fp16_cast_fp32(__half * __restrict__ src_ptr, float * __restrict__ dst_ptr,int m, int n)
{
    /*
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int start = (row*n+col)*1024;
    for(int i = 0; i < 1024; i++)
    {
        dst_ptr[start + i] = __half2float(src_ptr[start + i]);
    }
    */
    size_t i = (blockIdx.x * blockDim.x) + threadIdx.x;
    for(; i < m*n; i += gridDim.x * blockDim.x)
    {
        dst_ptr[i] = __half2float(src_ptr[i]);
    }
}

void mzw_quan(void * src_ptr, void * dst_ptr, int m, int n, float * scaling_factor, rocblas_datatype lowp, bool flag)
{
    //Now we assume that lowp passed can be rocblas_datatype_f16_r or HIPBLAS_R_8I
    if(lowp == rocblas_datatype_f16_r)
    {   
        //typecast
        //TO.DO.: Optimize it.
        //dim3 ThreadPerBlock(32,32);
        //dim3 NumBlock((m/ThreadPerBlock.x)/32,(n/ThreadPerBlock.y)/32);
        dim3 ThreadPerBlock(256);
        dim3 NumBlock(256);
        mzw_fp32_cast_fp16<<<NumBlock, ThreadPerBlock>>>((float*)src_ptr,(__half*)dst_ptr,m,n);
        //printf( "Typecast fp16 Time= %.3f msec\n",msecTotal);

    }
    else
    {

        //find the max abs value of src_matrix

        //compute the scalar factor

        //scalar multiply matrix

        //scalar addition

        //typecast
    
        //If flag = true, it means that we handling the mzw_quan of original Matrix A  and B, so we need to compute and rewrite s.f.
        //If flag = false, it means that we handling the mzw_quan of matrix RA and RB, so we just use s.f.
    }

}

void mzw_dequan(void * src_ptr, void * dst_ptr, int m, int n, float * scaling_factor, rocblas_datatype lowp)
{
    if(lowp == rocblas_datatype_f16_r)
    {   
        //typecast
        //TO.DO.: Optimize it
        //dim3 ThreadPerBlock(32,32);
        //dim3 NumBlock((m/ThreadPerBlock.x)/32,(n/ThreadPerBlock.y)/32);
        dim3 ThreadPerBlock(256);
        dim3 NumBlock(256);
        mzw_fp16_cast_fp32<<<NumBlock, ThreadPerBlock>>>((__half*)src_ptr,(float*)dst_ptr,m,n);
        //printf( "Typecast fp32 Time= %.3f msec\n",msecTotal);

    }


    //We dont need flag to indicate whether we need to rewrite the s.f. or not
}


rocblas_status mzw_faster_mul(rocblas_handle blas_handle, rocblas_operation Trans_A, rocblas_operation Trans_B, int m, int n, int k, void * alpha_addr,
                    void * device_fp32MatrixA, rocblas_datatype Atype, int ldA, void * device_fp32MatrixB, rocblas_datatype Btype, int ldB, void * beta_addr,
                    void * device_fp32MatrixC, rocblas_datatype Ctype, int ldC, void * device_fp32MatrixD, rocblas_datatype Dtype, int ldD, 
                    rocblas_datatype computeType, rocblas_gemm_algo algo, int32_t solution_index, uint32_t flags, void * memory_pool, int p1, int p2, int p3)
//TO.DO.: Exchange the order of arguments to keep the same as original GemmEx                               //DONE
{
    //QUES.: How to make the Trans_A and Trans_B right?
    //ANS.:  To be clear, if Trans_A/Trans_B == T, then make the corresponding one to become col-major
    /*如果前边的参数是't'，那么leading dimesion 就是矩阵的列数，因为此时的矩阵是按照C语言以行优先的方式来存储的；
    反之如果前边的参数是'n'，那么leading dimesion 就是矩阵的行数，此时的矩阵保持CUBLAS的列优先存储方式*/
    //因为fp16下的gemmex在NN的情况下跑得最快，所以我们第一步：假定我们只需要面对NN输入 第二步：将所有情况强行转为NN
    rocblas_operation New_Trans_A = rocblas_operation_none;
    rocblas_operation New_Trans_B = rocblas_operation_none;
    rocblas_datatype lowp = rocblas_datatype_f16_r;
    int new_ldA = ldA;
    int new_ldB = ldB;
    if(Trans_A == rocblas_operation_transpose)
    {
        New_Trans_A = rocblas_operation_none;
        new_ldA = m;
    }
    if(Trans_B == rocblas_operation_transpose)
    {
        New_Trans_B = rocblas_operation_none;
        new_ldA = k;
    }

    if(1)           //no need to judge type of lowp
    {
        /*
    sizeof(float)*(
        max(max_p2,max_p3)*2*m*k +                  //for hp_matrix A / RA
        max(max_p1,max_p3)*2*k*n +                  //for hp_matrix B / RB
        m*n*2) +                                    //for hp_matrix C / TMP
    sizeof(unsigned short)*(
        m*k + max(max_p2,max_p3)*m*k +              //for lp_matrix A / RA
        k*n + max(max_p1,max_p3)*k*n +              //for lp_matrix B / RB
        m*n*2);                                     //for lp_matrix C / TMP
        */
        //get each ptr
        //Ptr for HP Matrix
        int A_related_iter = max(p2,p3);
        char * device_fp32_MatrixA_RA [2*A_related_iter];      //start addr of each hp_matrix related to A
        //For each iteration, its layout is like: HP~A_(i-1),HPRA_(i)
        char * HP_A_RA_start_addr = (char*)memory_pool;
        for(int i = 0; i < 2*A_related_iter; i++)
        {
            device_fp32_MatrixA_RA[i] = HP_A_RA_start_addr+i*m*k*sizeof(float);
        }
    
        int B_related_iter = max(p1,p3);
        char * HP_B_RB_start_addr = HP_A_RA_start_addr+2*A_related_iter*m*k*sizeof(float);
        //For each iteration, its layout is like: HP~B_(i-1),HPRB_(i)
        char * device_fp32_MatrixB_RB [2*B_related_iter];        //start addr of each hp_matrix related to B
        for(int i = 0; i < 2*B_related_iter; i++)
        {
            device_fp32_MatrixB_RB[i] = HP_B_RB_start_addr+i*n*k*sizeof(float);
        }

        char * HP_appro_MatrixC_start_addr = HP_B_RB_start_addr + 2*B_related_iter*n*k*sizeof(float);
        char * HP_TMP_start_addr = HP_appro_MatrixC_start_addr + m * n * sizeof(float);
    

        //Ptr for LP Matrix
        char * LP_A_RA_start_addr = HP_TMP_start_addr+ m * n * sizeof(float);
        char * device_lp_MatrixA_RA [1+A_related_iter];
        for(int i = 0; i < 1+A_related_iter; i++)
        {
            device_lp_MatrixA_RA[i] = LP_A_RA_start_addr + i * m * k * sizeof(unsigned short);
        }

        char * LP_B_RB_start_addr = LP_A_RA_start_addr + (1+A_related_iter)*m*k*sizeof(unsigned short);
        char * device_lp_MatrixB_RB [1 + B_related_iter];
        for(int i = 0; i < 1+B_related_iter; i++)
        {
            device_lp_MatrixB_RB[i] = LP_B_RB_start_addr + i * n * k * sizeof(unsigned short);
        }

        //QUES.: Can we just calculate the FP32 Matrix C using FP16 Matrix A and B without quantization on FP16 Matrix C?
        char * LP_MatrixC_start_addr = LP_B_RB_start_addr + (1+B_related_iter)*n*k*sizeof(unsigned short);
        char * LP_TMP_start_addr = LP_MatrixC_start_addr + m * n * sizeof(unsigned short);

        //typecast hp_matrixA/B into lp_matrixA/B
        float scale_factor = 0;
        //std::cout<<"Begin quantization"<<std::endl;
        mzw_quan((void*)device_fp32MatrixA,(void*)device_lp_MatrixA_RA[0],m,k,&scale_factor,lowp,false);
        mzw_quan((void*)device_fp32MatrixB,(void*)device_lp_MatrixB_RB[0],k,n,&scale_factor,lowp,false);
        //std::cout<<"Finish quantization for A and B"<<std::endl;

        //run the lp_gemm
        //QUES:How to covert float to short int?                        ANS.: Use __half in the future
        float lp_alpha = static_cast<float>(1);
        float lp_beta = static_cast<float>(0);
        ROC_BLAS_CHECK(rocblas_gemm_ex(blas_handle, Trans_A, Trans_B, 
                                    m, n, k, &lp_alpha, 
                                    (__half*)device_lp_MatrixA_RA[0], rocblas_datatype_f16_r, ldA,
                                    (__half*)device_lp_MatrixB_RB[0], rocblas_datatype_f16_r, ldB,
                                    &lp_beta, (float*)HP_appro_MatrixC_start_addr, rocblas_datatype_f32_r, ldC,
                                    (float*)HP_appro_MatrixC_start_addr, rocblas_datatype_f32_r, ldC,
                                    rocblas_datatype_f32_r, rocblas_gemm_algo_standard, solution_index, flags));
        //hipDeviceSynchronize();
        //std::cout<<"Finish lp GEMM"<<std::endl;
        //dequan((void*)LP_MatrixC_start_addr,(void*)HP_appro_MatrixC_start_addr,m,n,lowp);

        float add_alpha = 1.0;
        float add_beta = -1.0;

        //FOR RB-related Calculation
        char * lasttime_HP_RB = static_cast<char *>(device_fp32MatrixB);
        char * lasttime_LP_RB = device_lp_MatrixB_RB[0];
        for(int i = 0; i < B_related_iter; i++)
        {
            mzw_dequan((void*)lasttime_LP_RB,(void*)device_fp32_MatrixB_RB[2*i],k,n,&scale_factor,lowp);
            //TO.DO.: Use GEAM() to do the sub                                              //DONE
            if(!i)
            {
                ROC_BLAS_CHECK(rocblas_sgeam(blas_handle, Trans_B, Trans_B,
                                            k,n, &add_alpha, (float*)lasttime_HP_RB, ldB,
                                            &add_beta, (float*)device_fp32_MatrixB_RB[2*i], ldB,
                                            (float*)device_fp32_MatrixB_RB[2*i+1], new_ldB));
            }
            else
            {
                ROC_BLAS_CHECK(rocblas_sgeam(blas_handle, New_Trans_B, New_Trans_B,
                                            k,n, &add_alpha, (float*)lasttime_HP_RB, new_ldB,
                                            &add_beta, (float*)device_fp32_MatrixB_RB[2*i], new_ldB,
                                            (float*)device_fp32_MatrixB_RB[2*i+1], new_ldB));
            }
            //hipDeviceSynchronize();
            //sub(lasttime_HP_RB,device_fp32_MatrixB_RB[2*i],device_fp32_MatrixB_RB[2*i+1]);
            mzw_quan((void*)device_fp32_MatrixB_RB[2*i+1],(void*)device_lp_MatrixB_RB[1+i],k,n,&scale_factor,lowp,false);
            lasttime_HP_RB = device_fp32_MatrixB_RB[2*i+1];
            lasttime_LP_RB = device_lp_MatrixB_RB[1+i];
        }
        //std::cout<<"Finish RB-related Calculation"<<std::endl;

        //FOR RA-related Calculation
        char * lasttime_HP_RA = static_cast<char*>(device_fp32MatrixA);
        char * lasttime_LP_RA = device_lp_MatrixA_RA[0];
        for(int i = 0; i < A_related_iter; i++)
        {
            mzw_dequan((void*)lasttime_LP_RA,(void*)device_fp32_MatrixA_RA[2*i],m,k,&scale_factor,lowp);
            //sub(op1,op2,res): res = op1 - op2
            if(!i)
            {
                ROC_BLAS_CHECK(rocblas_sgeam(blas_handle, Trans_A, Trans_A,
                                            m,k, &add_alpha, (float*)lasttime_HP_RA, ldA,
                                            &add_beta, (float*)device_fp32_MatrixA_RA[2*i], ldA,
                                            (float*)device_fp32_MatrixA_RA[2*i+1], new_ldA));
            }
            else
            {
                ROC_BLAS_CHECK(rocblas_sgeam(blas_handle, New_Trans_A, New_Trans_A,
                                            m,k, &add_alpha, (float*)lasttime_HP_RA, new_ldA,
                                            &add_beta, (float*)device_fp32_MatrixA_RA[2*i], new_ldA,
                                            (float*)device_fp32_MatrixA_RA[2*i+1], new_ldA));
            }
            //hipDeviceSynchronize();
            //sub(lasttime_HP_RA,device_fp32_MatrixA_RA[2*i],device_fp32_MatrixA_RA[2*i+1]);
            mzw_quan((void*)device_fp32_MatrixA_RA[2*i+1],(void*)device_lp_MatrixA_RA[1+i],m,k,&scale_factor,lowp,false);
            lasttime_HP_RA = device_fp32_MatrixA_RA[2*i+1];
            lasttime_LP_RA = device_lp_MatrixA_RA[1+i];
        }
        //std::cout<<"Finish RA-related Calculation"<<std::endl;

        //QUES.: How to deal with the transpose in refinement?
        add_beta = 1.0;
        //Refine with RB
        //unsigned short * LP_TMP = LP_MatrixC_start_addr;
        for(int i = 0; i < p1; i++)
        {
            ROC_BLAS_CHECK(rocblas_gemm_ex(blas_handle, Trans_A, New_Trans_B, 
                                        m, n, k, &add_alpha, 
                                        device_lp_MatrixA_RA[0], rocblas_datatype_f16_r, ldA,
                                        device_lp_MatrixB_RB[1+i], rocblas_datatype_f16_r, new_ldB,
                                        &add_beta, HP_appro_MatrixC_start_addr, rocblas_datatype_f32_r, ldC,
                                        HP_appro_MatrixC_start_addr, rocblas_datatype_f32_r, ldC,
                                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, solution_index, flags));
            /*
            ROC_BLAS_CHECK(rocblas_hgemm(blas_handle, Trans_A, Trans_B, 
                                        m, n, k, &lp_alpha, 
                                        device_lp_MatrixA_RA[0], ldA,
                                        device_lp_MatrixB_RB[1+i], ldB,
                                        &lp_beta, LP_TMP, ldC));
            dequan(LP_TMP,HP_TMP_start_addr,lowp);
            ROC_BLAS_CHECK(rocblas_sgeam(blas_handle, Trans_A, Trans_B,
                                        m,n, &add_alpha, HP_appro_MatrixC_start_addr, m,
                                        &add_beta, HP_TMP_start_addr, m,
                                        HP_appro_MatrixC_start_addr, m));
            */
        }
        //hipDeviceSynchronize();
        //std::cout<<"Finish RB-related refinement"<<std::endl;

        //Refine with RA
        for(int i = 0; i < p2; i++)
        {
            ROC_BLAS_CHECK(rocblas_gemm_ex(blas_handle, New_Trans_A, Trans_B, 
                                        m, n, k, &add_alpha, 
                                        device_lp_MatrixA_RA[1+i], rocblas_datatype_f16_r, new_ldA,
                                        device_lp_MatrixB_RB[0], rocblas_datatype_f16_r, ldB,
                                        &add_beta, HP_appro_MatrixC_start_addr, rocblas_datatype_f32_r, ldC,
                                        HP_appro_MatrixC_start_addr, rocblas_datatype_f32_r, ldC,
                                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, solution_index, flags));
            /*
            ROC_BLAS_CHECK(rocblas_hgemm(blas_handle, Trans_A, Trans_B, 
                                        m, n, k, &lp_alpha, 
                                        device_lp_MatrixA_RA[1+i], ldA,
                                        device_lp_MatrixB_RB[0], ldB,
                                        &lp_beta, LP_TMP, ldC));
            dequan(LP_TMP,HP_TMP_start_addr,lowp);
            ROC_BLAS_CHECK(rocblas_sgeam(blas_handle, Trans_A, Trans_B,
                                        m,n, &add_alpha, HP_appro_MatrixC_start_addr, m,
                                        &add_beta, HP_TMP_start_addr, m,
                                        HP_appro_MatrixC_start_addr, m));
            */
        }
        //hipDeviceSynchronize();
        //std::cout<<"Finish RA-related refinement"<<std::endl;

        //Refine with RB and RA
        for(int i = 0; i < p3; i++)
        {
            ROC_BLAS_CHECK(rocblas_gemm_ex(blas_handle, New_Trans_A, New_Trans_B, 
                                        m, n, k, &add_alpha, 
                                        device_lp_MatrixA_RA[1+i], rocblas_datatype_f16_r, new_ldA,
                                        device_lp_MatrixB_RB[1+i], rocblas_datatype_f16_r, new_ldB,
                                        &add_beta, HP_appro_MatrixC_start_addr, rocblas_datatype_f32_r, ldC,
                                        HP_appro_MatrixC_start_addr, rocblas_datatype_f32_r, ldC,
                                        rocblas_datatype_f32_r, rocblas_gemm_algo_standard, solution_index, flags));
            /*
            ROC_BLAS_CHECK(rocblas_hgemm(blas_handle, Trans_A, Trans_B, 
                                        m, n, k, &lp_alpha, 
                                        device_lp_MatrixA_RA[1+i], ldA,
                                        device_lp_MatrixB_RB[1+i], ldB,
                                        &lp_beta, LP_TMP, ldC));
            dequan(LP_TMP,HP_TMP_start_addr,lowp);
            ROC_BLAS_CHECK(rocblas_sgeam(blas_handle, Trans_A, Trans_B,
                                        m,n, &add_alpha, HP_appro_MatrixC_start_addr, m,
                                        &add_beta, HP_TMP_start_addr, m,
                                        HP_appro_MatrixC_start_addr, m));
            */
        }
        //hipDeviceSynchronize();
        //std::cout<<"Finish RB,RA-related refinement"<<std::endl;

        //Get the final result into the original output(device_fp32MatrixC)
        ROC_BLAS_CHECK(rocblas_sgeam(blas_handle, rocblas_operation_none, Trans_B,
                                    m,n, (float*)alpha_addr, (float*)HP_appro_MatrixC_start_addr, ldC,
                                    (float*)beta_addr, (float*)device_fp32MatrixC, ldC,
                                    (float*)device_fp32MatrixC, ldC));
        hipDeviceSynchronize();
        //std::cout<<"Finish whole faster_mul"<<std::endl;
        return rocblas_status_success;
    }
    
}
