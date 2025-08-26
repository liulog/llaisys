#include "../runtime_api.hpp"
#include "cuda_runtime.h"

#include <cstdlib>
#include <cstring>

namespace llaisys::device::nvidia {

namespace runtime_api {
int getDeviceCount() {
    int count;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    return count;
}

void setDevice(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}

void deviceSynchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}

llaisysStream_t createStream() {
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return (llaisysStream_t) stream;
}

void destroyStream(llaisysStream_t stream) {
    cudaError_t err = cudaStreamDestroy((cudaStream_t) stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}
void streamSynchronize(llaisysStream_t stream) {
    cudaError_t err = cudaStreamSynchronize((cudaStream_t) stream);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}

void *mallocDevice(size_t size) {
    void *d_ptr;
    cudaError_t err = cudaMalloc(&d_ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return d_ptr;
}

void freeDevice(void *ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}

void *mallocHost(size_t size) {
    void *h_ptr;
    cudaError_t err = cudaMallocHost(&h_ptr, size);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }
    return h_ptr;
}

void freeHost(void *ptr) {
    cudaError_t err = cudaFreeHost(ptr);
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}

void memcpySync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind) {
    cudaError_t err;
    switch (kind) {
        case LLAISYS_MEMCPY_H2H:
            err = cudaMemcpy(dst, src, size, cudaMemcpyHostToHost);
            break;
        case LLAISYS_MEMCPY_H2D:
            err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
            break;
        case LLAISYS_MEMCPY_D2H:
            err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
            break;
        case LLAISYS_MEMCPY_D2D:
            err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
            break;
        default:
            std::cerr << "Invalid memcpy kind!" << std::endl;
            return;
    }
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

void memcpyAsync(void *dst, const void *src, size_t size, llaisysMemcpyKind_t kind, llaisysStream_t stream) {
    cudaError_t err;
    switch (kind) {
        case LLAISYS_MEMCPY_H2H:
            err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToHost, (cudaStream_t)stream);
            break;
        case LLAISYS_MEMCPY_H2D:
            err = cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, (cudaStream_t)stream);
            break;
        case LLAISYS_MEMCPY_D2H:
            err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, (cudaStream_t)stream);
            break;
        case LLAISYS_MEMCPY_D2D:
            err = cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, (cudaStream_t)stream);
            break;
        default:
            std::cerr << "Invalid memcpy kind!" << std::endl;
            return;
    }
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}
} // namespace llaisys::device::nvidia
