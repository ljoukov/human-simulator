#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

static double milliseconds() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static id<MTLBuffer> loadBuffer(id<MTLDevice> device, const char *path) {
    NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path]];
    if (!data.length) return nil;
    return [device newBufferWithBytes:data.bytes length:data.length options:MTLResourceStorageModeShared];
}

// Input buffers contain the exact float32 calibration frames and parameters used on CUDA.
int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc < 7) return 2;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { fprintf(stderr, "Metal device unavailable\n"); return 1; }
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]
                                                   encoding:NSUTF8StringEncoding error:&error];
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        double started = milliseconds();
        id<MTLLibrary> library = [device newLibraryWithSource:source options:options error:&error];
        if (!library) { NSLog(@"%@", error); return 1; }
        id<MTLFunction> function = [library newFunctionWithName:@"hip_score"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (!pipeline) { NSLog(@"%@", error); return 1; }
        double compileMs = milliseconds() - started;
        id<MTLCommandQueue> queue = [device newCommandQueue];
        started = milliseconds();
        id<MTLBuffer> frames = loadBuffer(device, argv[2]);
        id<MTLBuffer> parameters = loadBuffer(device, argv[3]);
        if (!frames || !parameters || frames.length % 32 || parameters.length % 36) return 1;
        unsigned frameCount = unsigned(frames.length / 32);
        unsigned capacity = unsigned(parameters.length / 36);
        id<MTLBuffer> scores = [device newBufferWithLength:capacity*sizeof(float) options:MTLResourceStorageModeShared];
        if (!scores) return 1;
        double setupMs = milliseconds() - started;
        const int repeats = atoi(argv[5]);
        NSMutableArray *rows = [NSMutableArray new];
        for (int arg = 6; arg < argc; ++arg) {
            unsigned count = unsigned(atoi(argv[arg]));
            if (count == 0 || count > capacity || repeats < 3) return 2;
            NSMutableArray *gpuTimes = [NSMutableArray new];
            NSMutableArray *wallTimes = [NSMutableArray new];
            for (int repeat = -1; repeat < repeats; ++repeat) {
                double wallStart = milliseconds();
                id<MTLCommandBuffer> command = [queue commandBuffer];
                id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
                [encoder setComputePipelineState:pipeline];
                [encoder setBuffer:frames offset:0 atIndex:0];
                [encoder setBuffer:parameters offset:0 atIndex:1];
                [encoder setBuffer:scores offset:0 atIndex:2];
                [encoder setBytes:&frameCount length:sizeof(frameCount) atIndex:3];
                [encoder dispatchThreads:MTLSizeMake(count,1,1)
                      threadsPerThreadgroup:MTLSizeMake(std::min<NSUInteger>(128,pipeline.maxTotalThreadsPerThreadgroup),1,1)];
                [encoder endEncoding];
                [command commit];
                [command waitUntilCompleted];
                if (command.status != MTLCommandBufferStatusCompleted) { NSLog(@"%@",command.error); return 1; }
                if (repeat >= 0) {
                    [wallTimes addObject:@(milliseconds()-wallStart)];
                    [gpuTimes addObject:@(1000*(command.GPUEndTime-command.GPUStartTime))];
                }
            }
            NSString *dir = [NSString stringWithUTF8String:argv[4]];
            NSString *path = [dir stringByAppendingPathComponent:[NSString stringWithFormat:@"%u.bin", count]];
            NSData *out = [NSData dataWithBytesNoCopy:scores.contents length:count*sizeof(float) freeWhenDone:NO];
            if (![out writeToFile:path atomically:YES]) return 1;
            [rows addObject:@{@"candidates": @(count), @"frames": @(frameCount),
                              @"deviceSamplesMs": gpuTimes, @"dispatchAndWaitSamplesMs": wallTimes}];
        }
        NSDictionary *result = @{@"device": device.name, @"compileMs": @(compileMs), @"loadBuffersMs": @(setupMs), @"rows": rows};
        NSData *json = [NSJSONSerialization dataWithJSONObject:result options:0 error:&error];
        if (!json) return 1;
        fwrite(json.bytes, 1, json.length, stdout);
    }
}
