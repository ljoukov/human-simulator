// macOS 15+; compile as a dynamic library and call from sensitivity.py.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstring>
#include <vector>

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
static constexpr NSUInteger lanes = 128, subjects = 6, tile = 1024;

extern "C" const char *setup_sweep(const char *source) {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr,"Metal device unavailable\n"); return nullptr; }
        queue = [device newCommandQueue];
        MTLCompileOptions *opt = [MTLCompileOptions new];
        opt.mathMode = MTLMathModeSafe;
        opt.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError *error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:[NSString stringWithUTF8String:source]
                                                      options:opt error:&error];
        if (!library) { NSLog(@"%@",error); return nullptr; }
        pipelines = [NSMutableDictionary new];
        for (NSString *name in @[@"hip_score",@"find_anchors",@"screen",@"grid_args",@"score_survivors",@"final_minimum"]) {
            id<MTLFunction> f = [library newFunctionWithName:name];
            id<MTLComputePipelineState> p = [device newComputePipelineStateWithFunction:f error:&error];
            if (!p || p.maxTotalThreadsPerThreadgroup < lanes) { NSLog(@"%@",error); return nullptr; }
            pipelines[name] = p;
        }
        return strdup(device.name.UTF8String);
    }
}

static id<MTLBuffer> input(const void *data, NSUInteger bytes) {
    return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
}
static id<MTLBuffer> buffer(NSUInteger bytes, MTLResourceOptions mode = MTLResourceStorageModePrivate) {
    return [device newBufferWithLength:bytes options:mode];
}
static void groups(id<MTLComputeCommandEncoder> enc, NSUInteger x, NSUInteger y = 1,
                   NSUInteger width = lanes) {
    [enc dispatchThreadgroups:MTLSizeMake(x,y,1) threadsPerThreadgroup:MTLSizeMake(width,1,1)];
}
static double seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// One timed call allocates all buffers, uploads all inputs, scores the population,
// finds exact winners for every resample and offset, and returns the IDs to Python.
extern "C" int run_sweep(const float *parameters, unsigned n, const float *frames,
                         const unsigned *lengths, unsigned offsets, const unsigned *weights,
                         unsigned resamples, const unsigned *pilotWeights, unsigned anchors,
                         unsigned *out, unsigned *nKept, double *times) {
    @autoreleasepool {
        if (!device || !n || !offsets || !resamples || anchors > 64) return 1;
        unsigned nFrames = 0;
        for (unsigned s=0; s<subjects; ++s) nFrames += lengths[s];
        id<MTLBuffer> par = input(parameters, NSUInteger(n)*9*4);
        id<MTLBuffer> obs = input(frames, NSUInteger(nFrames)*offsets*8*4);
        id<MTLBuffer> counts = input(weights, resamples*subjects*4);
        id<MTLBuffer> pilot = input(pilotWeights,std::max(1u,anchors)*subjects*4);
        id<MTLBuffer> losses = buffer(NSUInteger(n)*subjects*4);
        id<MTLBuffer> keep = buffer(NSUInteger(n)*4);
        id<MTLBuffer> tiles = buffer(((NSUInteger(n)+tile-1)/tile)*resamples*16);
        id<MTLBuffer> anchorsFound = buffer(64*4);
        id<MTLBuffer> stats = buffer(offsets*16,MTLResourceStorageModeShared);
        id<MTLBuffer> err = buffer(4,MTLResourceStorageModeShared);
        id<MTLBuffer> result = buffer(offsets*resamples*4,MTLResourceStorageModeShared);
        if (!par || !obs || !counts || !pilot || !losses || !keep || !tiles ||
            !anchorsFound || !stats || !err || !result) return 2;
        memset(stats.contents,0,stats.length);
        memset(err.contents,0,err.length);

        double start = seconds();
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [command computeCommandEncoder];
        for (unsigned o=0; o<offsets; ++o) {
            NSUInteger base = NSUInteger(o)*16;
            NSUInteger f = NSUInteger(o)*nFrames;
            [enc setComputePipelineState:pipelines[@"hip_score"]];
            [enc setBuffer:par offset:0 atIndex:1];
            for (unsigned s=0; s<subjects; ++s) {
                [enc setBuffer:obs offset:f*8*4 atIndex:0];
                [enc setBuffer:losses offset:NSUInteger(s)*n*4 atIndex:2];
                [enc setBytes:&lengths[s] length:4 atIndex:3];
                [enc dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(lanes,1,1)];
                f += lengths[s];
            }
            if (anchors) {
                [enc setComputePipelineState:pipelines[@"find_anchors"]];
                [enc setBuffer:losses offset:0 atIndex:0];
                [enc setBuffer:pilot offset:0 atIndex:1];
                [enc setBuffer:anchorsFound offset:0 atIndex:2];
                [enc setBytes:&n length:4 atIndex:3];
                groups(enc,anchors);
            }
            [enc setComputePipelineState:pipelines[@"screen"]];
            [enc setBuffer:losses offset:0 atIndex:0];
            [enc setBuffer:anchorsFound offset:0 atIndex:1];
            [enc setBuffer:keep offset:0 atIndex:2];
            [enc setBuffer:stats offset:base atIndex:3];
            [enc setBuffer:err offset:0 atIndex:4];
            [enc setBytes:&n length:4 atIndex:5];
            [enc setBytes:&anchors length:4 atIndex:6];
            groups(enc,(n+lanes-1)/lanes);
            [enc setComputePipelineState:pipelines[@"grid_args"]];
            [enc setBuffer:stats offset:base atIndex:0];
            [enc setBytes:&resamples length:4 atIndex:1];
            groups(enc,1,1,1);
            [enc setComputePipelineState:pipelines[@"score_survivors"]];
            [enc setBuffer:losses offset:0 atIndex:0];
            [enc setBuffer:counts offset:0 atIndex:1];
            [enc setBuffer:keep offset:0 atIndex:2];
            [enc setBuffer:stats offset:base atIndex:3];
            [enc setBuffer:tiles offset:0 atIndex:4];
            [enc setBytes:&n length:4 atIndex:5];
            [enc dispatchThreadgroupsWithIndirectBuffer:stats indirectBufferOffset:base+4
                                  threadsPerThreadgroup:MTLSizeMake(lanes,1,1)];
            [enc setComputePipelineState:pipelines[@"final_minimum"]];
            [enc setBuffer:tiles offset:0 atIndex:0];
            [enc setBuffer:stats offset:base atIndex:1];
            [enc setBuffer:result offset:NSUInteger(o)*resamples*4 atIndex:2];
            groups(enc,resamples);
        }
        [enc endEncoding];
        [command commit];
        [command waitUntilCompleted];
        times[0] = command.GPUEndTime-command.GPUStartTime;
        times[1] = seconds()-start;
        if (command.status != MTLCommandBufferStatusCompleted) { NSLog(@"%@",command.error); return 3; }
        if (*static_cast<unsigned *>(err.contents)) { fprintf(stderr,"Loss outside guaranteed exact integer range\n"); return 4; }
        memcpy(out,result.contents,result.length);
        for (unsigned o=0; o<offsets; ++o) nKept[o] = static_cast<unsigned *>(stats.contents)[4*o];
        return 0;
    }
}
