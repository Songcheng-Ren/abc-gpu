#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#endif
#include "stdio.h"
#include "string.h"
#include "ocl_macros.h"
#define USE_HOST_MEMORY

#include<time.h>
#include<stdlib.h>
#include<cmath>
#pragma comment(lib, "OpenCL.lib")

const int NP = 128;//种群的规模，采蜜蜂+观察蜂
const int FoodNumber = NP / 2;//食物的数量，为采蜜蜂的数量
const int limit = 20;//限度，超过这个限度没有更新采蜜蜂变成侦查蜂
const int maxCycle = 10000;//停止条件

/*****函数的特定参数*****/
const int D = 2;//函数的参数个数
const double lb = -100;//函数的下界
const double ub = 100;//函数的上界


float foods[FoodNumber * 2];
float trueFit[FoodNumber];
float fitness[FoodNumber];
float prob[FoodNumber];
int trial[FoodNumber];
float result[1];
float rand_1[FoodNumber * maxCycle];
float rand_2[FoodNumber * maxCycle * 3];
float rand_3[FoodNumber * maxCycle];
float rand_4[FoodNumber * maxCycle * 2];


/*****函数的实现****/
float random(float start, float end)//随机产生区间内的随机数
{
    return start + (end - start) * rand() / (RAND_MAX + 1.0);
}

float calculationTruefit(float x, float y)//计算真实的函数值
{
    float truefit = 0;
    /******测试函数1******/
    truefit = 0.5 + (sin(sqrt(x * x + y * y)) * sin(sqrt(x * x + y * y)) - 0.5)
        / ((1 + 0.001 * (x * x + y * y)) * (1 + 0.001 * (x * x + y * y)));

    return truefit;
}

float calculationFitness(float truefit)//计算适应值
{
    float fitnessResult = 0;
    if (truefit >= 0)
    {
        fitnessResult = 1 / (truefit + 1);
    }
    else
    {
        fitnessResult = 1 + abs(truefit);
    }
    return fitnessResult;
}

void initilize()//初始化参数
{
    int i;
    for (i = 0; i < FoodNumber; i++)
    {
        foods[2 * i] = random(lb, ub);
        foods[2 * i + 1] = random(lb, ub);
        trueFit[i] = calculationTruefit(foods[2 * i], foods[2 * i + 1]);
        fitness[i] = calculationFitness(trueFit[i]);
        prob[i] = 0;
        trial[i] = 0;
    }

}



int main() {
    cl_int status = 0;
    cl_int binSize = 256;
    cl_int groupSize = 16;
    cl_int subHistgCnt;
    cl_device_type dType = CL_DEVICE_TYPE_GPU;
    cl_platform_id* platforms = NULL;
    cl_uint     num_platforms;
    cl_device_id   device;
    cl_context     context;
    cl_command_queue commandQueue;

    cl_mem foodsBuffer;
    cl_mem trueFitBuffer;
    cl_mem fitnessBuffer;
    cl_mem probBuffer;
    cl_mem trialBuffer;
    cl_mem resultBuffer;
    cl_mem RAND_1;
    cl_mem RAND_2;
    cl_mem RAND_3;
    cl_mem RAND_4;

    initilize();
    result[0] = 100;
    for (int i = 0; i < FoodNumber * maxCycle; i++) {
        rand_1[i] = random(-1, 1);
    }
    for (int i = 0; i < FoodNumber * maxCycle * 3; i++) {
        rand_2[i] = random(0, 1);
    }
    for (int i = 0; i < FoodNumber * maxCycle; i++) {
        rand_3[i] = random(-1, 1);
    }
    for (int i = 0; i < FoodNumber * maxCycle * 2; i++) {
        rand_4[i] = random(0, 1);
    }

    status = clGetPlatformIDs(0, NULL, &num_platforms);
    LOG_OCL_ERROR(status, "clGetPlatformIDs Failed");

    platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    status = clGetPlatformIDs(num_platforms, platforms, NULL);
    LOG_OCL_ERROR(status, "clGetPlatformIDs Failed");

    //Get the first available device 
    status = clGetDeviceIDs(platforms[2], dType, 1, &device, NULL);
    LOG_OCL_ERROR(status, "clGetDeviceIDs Failed.");

    //Create an execution context for the selected platform and device. 
    cl_context_properties cps[3] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platforms[2],
        0
    };
    context = clCreateContextFromType(
        cps,
        dType,
        NULL,
        NULL,
        &status);
    LOG_OCL_ERROR(status, "clCreateContextFromType Failed.");

    // Create command queue
    commandQueue = clCreateCommandQueue(context,
        device,
        CL_QUEUE_PROFILING_ENABLE,          // enable the event to record timing
        &status);
    LOG_OCL_ERROR(status, "clCreateCommandQueue Failed.");

    foodsBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        sizeof(float) * FoodNumber * 2,
        foods,
        &status);

    trueFitBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        sizeof(float) * FoodNumber,
        trueFit,
        &status);

    fitnessBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        sizeof(float) * FoodNumber,
        fitness,
        &status);

    probBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        sizeof(float) * FoodNumber,
        prob,
        &status);

    trialBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        sizeof(int) * FoodNumber,
        trial,
        &status);

    resultBuffer = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        sizeof(float),
        result,
        &status);

    RAND_1 = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        sizeof(float) * FoodNumber * maxCycle,
        rand_1,
        &status);

    RAND_2 = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        sizeof(float) * FoodNumber * maxCycle * 3,
        rand_2,
        &status);

    RAND_3 = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        sizeof(float) * FoodNumber * maxCycle,
        rand_3,
        &status);

    RAND_4 = clCreateBuffer(
        context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        sizeof(float) * FoodNumber * maxCycle * 2,
        rand_4,
        &status);



    FILE* fp;
    fopen_s(&fp, "kernel.cl", "rb");
    fseek(fp, 0L, SEEK_END);
    size_t fileSize = ftell(fp);
    rewind(fp);
    char* kernel_str = new char[fileSize + 1];
    fread(kernel_str, fileSize, 1, fp);
    kernel_str[fileSize] = '\0';
    cl_program program = clCreateProgramWithSource(context, 1,
        (const char**)&kernel_str, NULL, &status);
    LOG_OCL_ERROR(status, "clCreateProgramWithSource Failed.");

    // Build the program
    status = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (status != CL_SUCCESS)
        LOG_OCL_COMPILER_ERROR(program, device);

    // Create the OpenCL kernel
    cl_kernel sendEmployedBees = clCreateKernel(program, "sendEmployedBees", &status);
    LOG_OCL_ERROR(status, "clCreateKernel Failed.");

    cl_kernel CalculateProbabilities = clCreateKernel(program, "CalculateProbabilities", &status);
    LOG_OCL_ERROR(status, "clCreateKernel Failed.");

    cl_kernel sendOnlookerBees = clCreateKernel(program, "sendOnlookerBees", &status);
    LOG_OCL_ERROR(status, "clCreateKernel Failed.");

    cl_kernel MemorizeBestSource = clCreateKernel(program, "MemorizeBestSource", &status);
    LOG_OCL_ERROR(status, "clCreateKernel Failed.");

    cl_kernel sendScoutBees = clCreateKernel(program, "sendScoutBees", &status);
    LOG_OCL_ERROR(status, "clCreateKernel Failed.");

    // Set the arguments of the kernel
    status = clSetKernelArg(sendEmployedBees, 0, sizeof(cl_mem), (void*)&foodsBuffer);
    status |= clSetKernelArg(sendEmployedBees, 1, sizeof(cl_mem), (void*)&trueFitBuffer);
    status |= clSetKernelArg(sendEmployedBees, 2, sizeof(cl_mem), (void*)&fitnessBuffer);
    status |= clSetKernelArg(sendEmployedBees, 3, sizeof(cl_mem), (void*)&trialBuffer);
    status |= clSetKernelArg(sendEmployedBees, 4, sizeof(cl_mem), (void*)&RAND_1);
    LOG_OCL_ERROR(status, "clSetKernelArg Failed.");

    status = clSetKernelArg(CalculateProbabilities, 0, sizeof(cl_mem), (void*)&fitnessBuffer);
    status |= clSetKernelArg(CalculateProbabilities, 1, sizeof(cl_mem), (void*)&probBuffer);
    status |= clSetKernelArg(CalculateProbabilities, 2, 64 * sizeof(float), NULL);
    LOG_OCL_ERROR(status, "clSetKernelArg Failed.");

    status = clSetKernelArg(sendOnlookerBees, 0, sizeof(cl_mem), (void*)&foodsBuffer);
    status |= clSetKernelArg(sendOnlookerBees, 1, sizeof(cl_mem), (void*)&trueFitBuffer);
    status |= clSetKernelArg(sendOnlookerBees, 2, sizeof(cl_mem), (void*)&fitnessBuffer);
    status |= clSetKernelArg(sendOnlookerBees, 3, sizeof(cl_mem), (void*)&probBuffer);
    status |= clSetKernelArg(sendOnlookerBees, 4, sizeof(cl_mem), (void*)&trialBuffer);
    status |= clSetKernelArg(sendOnlookerBees, 5, sizeof(cl_mem), (void*)&RAND_2);
    status |= clSetKernelArg(sendOnlookerBees, 6, sizeof(cl_mem), (void*)&RAND_3);

    LOG_OCL_ERROR(status, "clSetKernelArg Failed.");

    status = clSetKernelArg(MemorizeBestSource, 0, sizeof(cl_mem), (void*)&trueFitBuffer);
    status |= clSetKernelArg(MemorizeBestSource, 1, sizeof(cl_mem), (void*)&resultBuffer);
    status |= clSetKernelArg(MemorizeBestSource, 2, 64 * sizeof(float), NULL);
    LOG_OCL_ERROR(status, "clSetKernelArg Failed.");

    status = clSetKernelArg(sendScoutBees, 0, sizeof(cl_mem), (void*)&foodsBuffer);
    status |= clSetKernelArg(sendScoutBees, 1, sizeof(cl_mem), (void*)&trueFitBuffer);
    status |= clSetKernelArg(sendScoutBees, 2, sizeof(cl_mem), (void*)&fitnessBuffer);
    status |= clSetKernelArg(sendScoutBees, 3, sizeof(cl_mem), (void*)&trialBuffer);
    status |= clSetKernelArg(sendScoutBees, 4, sizeof(cl_mem), (void*)&RAND_4);
    LOG_OCL_ERROR(status, "clSetKernelArg Failed.");

    size_t globalThreads = 64;
    size_t localThreads = 16;

    int gen = 0;
    //cl_event event1, event2, event3,event4, event5, event6;
    float* answer = (float*)malloc(sizeof(float));
    float* temp = (float*)malloc(sizeof(float) * 64);
    float *temps = (float*)malloc(sizeof(float) * 128);
    while (gen < maxCycle)
    {   
        
        status = clSetKernelArg(sendEmployedBees, 5, sizeof(int), &gen);
        status = clEnqueueNDRangeKernel(
            commandQueue,
            sendEmployedBees,
            1,
            NULL,
            &globalThreads,
            &localThreads,
            0,
            NULL,
            NULL);
        LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed.");
        //sendEmployedBees();
        
        
        status = clEnqueueNDRangeKernel(
            commandQueue,
            CalculateProbabilities,
            1,
            NULL,
            &globalThreads,
            &globalThreads,
            0,
            NULL,
            NULL);
        LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed.");
        
        status = clSetKernelArg(sendOnlookerBees, 7, sizeof(int), &gen);
        status = clEnqueueNDRangeKernel(
            commandQueue,
            sendOnlookerBees,
            1,
            NULL,
            &globalThreads,
            &localThreads,
            0,
            NULL,
            NULL);
        LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed.");
        
        status = clEnqueueNDRangeKernel(
            commandQueue,
            MemorizeBestSource,
            1,
            NULL,
            &globalThreads,
            &globalThreads,
            0,
            NULL,
            NULL);
        LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed.");
        
        status = clSetKernelArg(sendScoutBees, 5, sizeof(int), &gen);
        status = clEnqueueNDRangeKernel(
            commandQueue,
            sendScoutBees,
            1,
            NULL,
            &globalThreads,
            &localThreads,
            0,
            NULL,
            NULL);
        LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed.");
        
        status = clEnqueueNDRangeKernel(
            commandQueue,
            MemorizeBestSource,
            1,
            NULL,
            &globalThreads,
            &globalThreads,
            0,
            NULL,
            NULL);
        LOG_OCL_ERROR(status, "clEnqueueNDRangeKernel Failed.");
        
        status = clEnqueueReadBuffer(
            commandQueue,
            resultBuffer,
            CL_TRUE,
            0,
            sizeof(float),
            answer,
            0,
            NULL,
            NULL);
        LOG_OCL_ERROR(status, "clEnqueueReadBuffer of intermediateHistG Failed.");
        for (int i = 0; i < 1; i++) {
            printf_s("%f\n", *answer);
        }
        gen++;
    }
    
}
