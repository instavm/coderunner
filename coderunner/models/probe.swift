import CoreML
import Darwin
import Foundation
import Metal
import MetalPerformanceShaders

struct ProbeResult: Codable {
    var gpu_name: String = ""
    var unified_memory: Bool = false
    var recommended_max_working_set_gb: Double = 0
    var bandwidth_gbs: Double = 0
    var gpu_fp16_tflops: Double = 0
    var ssd_read_gbs: Double = 0
    var ane_cores: Int = 0
    var ane_verified: Bool = false
    var ane_plan_cost_pct: Double? = nil
    var ane_compile_cold_ms: Double? = nil
    var ane_load_warm_ms: Double? = nil
    var ane_infer_p50_ms: Double? = nil
    var ane_infer_p95_ms: Double? = nil
    var ane_fp16_tops: Double? = nil
    var ane_speedup_vs_cpu: Double? = nil
    var ane_speedup_vs_gpu: Double? = nil
    var gpu_contention_pct: Double? = nil
    var ane_contention_pct: Double? = nil
    var ane_contention_detected: Bool = false
    var thermal_state: String = "nominal"
    var low_power_mode: Bool = false
    var noisy: Bool = false
    var error: String? = nil
}

func nowSeconds() -> Double {
    var ts = timespec()
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts)
    return Double(ts.tv_sec) + Double(ts.tv_nsec) * 1e-9
}

func percentile(_ values: [Double], _ p: Double) -> Double {
    guard !values.isEmpty else { return 0 }
    let sorted = values.sorted()
    let idx = min(sorted.count - 1, max(0, Int(Double(sorted.count - 1) * p)))
    return sorted[idx]
}

func thermalLabel(_ state: ProcessInfo.ThermalState) -> String {
    switch state {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

func measureBandwidth(device: MTLDevice, queue: MTLCommandQueue) -> Double {
    let source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void stream_copy(device const float4 *src [[buffer(0)]],
                            device float4 *dst [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
        dst[id] = src[id];
    }
    """
    guard let lib = try? device.makeLibrary(source: source, options: nil),
          let fn = lib.makeFunction(name: "stream_copy"),
          let pipe = try? device.makeComputePipelineState(function: fn) else {
        return 0
    }
    let bytes = 128 * 1024 * 1024
    let count = bytes / MemoryLayout<SIMD4<Float>>.stride
    guard let src = device.makeBuffer(length: bytes, options: .storageModeShared),
          let dst = device.makeBuffer(length: bytes, options: .storageModeShared) else {
        return 0
    }
    memset(src.contents(), 1, bytes)
    let w = pipe.threadExecutionWidth
    let tg = MTLSize(width: min(256, max(w, 1)), height: 1, depth: 1)
    let grid = MTLSize(width: count, height: 1, depth: 1)

    func runOnce() -> Double {
        let t0 = nowSeconds()
        guard let cb = queue.makeCommandBuffer(),
              let enc = cb.makeComputeCommandEncoder() else { return 0 }
        enc.setComputePipelineState(pipe)
        enc.setBuffer(src, offset: 0, index: 0)
        enc.setBuffer(dst, offset: 0, index: 1)
        enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        return nowSeconds() - t0
    }

    for _ in 0..<2 { _ = runOnce() }
    var times: [Double] = []
    for _ in 0..<8 {
        let dt = runOnce()
        if dt > 0 { times.append(dt) }
    }
    let med = percentile(times, 0.5)
    guard med > 0 else { return 0 }
    return (Double(bytes * 2) / med) / 1e9
}

struct GemmContext {
    let matA: MPSMatrix
    let matB: MPSMatrix
    let matC: MPSMatrix
    let op: MPSMatrixMultiplication
    let flops: Double
}

func makeGemm(device: MTLDevice, n: Int = 2048) -> GemmContext? {
    let rowBytes = n * 2
    let bytes = n * rowBytes
    guard let bufA = device.makeBuffer(length: bytes, options: .storageModeShared),
          let bufB = device.makeBuffer(length: bytes, options: .storageModeShared),
          let bufC = device.makeBuffer(length: bytes, options: .storageModeShared) else {
        return nil
    }
    let desc = MPSMatrixDescriptor(rows: n, columns: n, rowBytes: rowBytes, dataType: .float16)
    let matA = MPSMatrix(buffer: bufA, descriptor: desc)
    let matB = MPSMatrix(buffer: bufB, descriptor: desc)
    let matC = MPSMatrix(buffer: bufC, descriptor: desc)
    let op = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false,
                                     resultRows: n, resultColumns: n, interiorColumns: n,
                                     alpha: 1.0, beta: 0.0)
    return GemmContext(matA: matA, matB: matB, matC: matC, op: op, flops: 2.0 * Double(n) * Double(n) * Double(n))
}

func runGemmOnce(_ ctx: GemmContext, queue: MTLCommandQueue) -> Double {
    let t0 = nowSeconds()
    guard let cb = queue.makeCommandBuffer() else { return 0 }
    ctx.op.encode(commandBuffer: cb, leftMatrix: ctx.matA, rightMatrix: ctx.matB, resultMatrix: ctx.matC)
    cb.commit()
    cb.waitUntilCompleted()
    return nowSeconds() - t0
}

func measureSsdReadGbs() -> Double {
    let bytes = 32 * 1024 * 1024
    let path = (NSTemporaryDirectory() as NSString).appendingPathComponent("coderunner-probe-\(getpid()).bin")
    defer { unlink(path) }
    let wfd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0o600)
    guard wfd >= 0 else { return 0 }
    let chunkSize = 1024 * 1024
    let buf = UnsafeMutableRawPointer.allocate(byteCount: chunkSize, alignment: 4096)
    defer { buf.deallocate() }
    memset(buf, 0x5A, chunkSize)
    var written = 0
    while written < bytes {
        let n = write(wfd, buf, chunkSize)
        if n <= 0 { break }
        written += n
    }
    fsync(wfd)
    close(wfd)
    guard written == bytes else { return 0 }

    let rfd = open(path, O_RDONLY)
    guard rfd >= 0 else { return 0 }
    _ = fcntl(rfd, F_NOCACHE, 1)
    let t0 = nowSeconds()
    var total = 0
    while true {
        let n = read(rfd, buf, chunkSize)
        if n <= 0 { break }
        total += n
    }
    let dt = nowSeconds() - t0
    close(rfd)
    guard dt > 0, total > 0 else { return 0 }
    return (Double(total) / dt) / 1e9
}

func detectAneCores() -> Int {
    if #available(macOS 14.0, *) {
        for dev in MLModel.availableComputeDevices {
            if case .neuralEngine(let ne) = dev {
                return ne.totalCoreCount
            }
        }
    }
    return 0
}

func anePlanCoverage(compiledURL: URL) -> Double? {
    if #available(macOS 14.4, *) {
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        let sem = DispatchSemaphore(value: 0)
        var pct: Double? = nil
        Task {
            do {
                let plan = try await MLComputePlan.load(contentsOf: compiledURL, configuration: cfg)
                var totalWeight = 0.0
                var aneWeight = 0.0
                if case .program(let program) = plan.modelStructure {
                    for (_, fn) in program.functions {
                        for op in fn.block.operations {
                            let w = plan.estimatedCost(of: op)?.weight ?? 1.0
                            totalWeight += w
                            if let usage = plan.deviceUsage(for: op) {
                                if case .neuralEngine = usage.preferred {
                                    aneWeight += w
                                }
                            }
                        }
                    }
                } else if case .neuralNetwork(let nn) = plan.modelStructure {
                    for layer in nn.layers {
                        let w = 1.0
                        totalWeight += w
                        if let usage = plan.deviceUsage(for: layer) {
                            if case .neuralEngine = usage.preferred {
                                aneWeight += w
                            }
                        }
                    }
                }
                if totalWeight > 0 {
                    pct = (aneWeight / totalWeight) * 100.0
                }
            } catch {
                pct = nil
            }
            sem.signal()
        }
        _ = sem.wait(timeout: .now() + 10)
        return pct
    }
    return nil
}

func makeInputProvider(channels: Int, width: Int) throws -> MLDictionaryFeatureProvider {
    let arr = try MLMultiArray(shape: [NSNumber(value: channels), 1, NSNumber(value: width)], dataType: .float16)
    let ptr = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
    for i in 0..<(channels * width) {
        ptr[i] = 0x3800 // 0.5 in FP16
    }
    return try MLDictionaryFeatureProvider(dictionary: ["x": MLFeatureValue(multiArray: arr)])
}

func timeCoreML(compiledURL: URL, units: MLComputeUnits, input: MLDictionaryFeatureProvider,
                warmup: Int = 4, iters: Int = 20) throws -> (loadMs: Double, p50Ms: Double, p95Ms: Double, model: MLModel) {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = units
    let tLoad = nowSeconds()
    let model = try MLModel(contentsOf: compiledURL, configuration: cfg)
    let loadMs = (nowSeconds() - tLoad) * 1000.0
    for _ in 0..<warmup {
        _ = try model.prediction(from: input)
    }
    var samples: [Double] = []
    for _ in 0..<iters {
        let t0 = nowSeconds()
        _ = try model.prediction(from: input)
        samples.append((nowSeconds() - t0) * 1000.0)
    }
    return (loadMs, percentile(samples, 0.5), percentile(samples, 0.95), model)
}

func parseArgs() -> (modelPath: String?, compiledCache: String?, channels: Int, width: Int, layers: Int) {
    let args = CommandLine.arguments
    var modelPath: String? = nil
    var compiledCache: String? = nil
    var channels = 256
    var width = 256
    var layers = 8
    var i = 1
    while i < args.count {
        switch args[i] {
        case "--ane-model" where i + 1 < args.count:
            modelPath = args[i + 1]; i += 2
        case "--compiled-cache" where i + 1 < args.count:
            compiledCache = args[i + 1]; i += 2
        case "--channels" where i + 1 < args.count:
            channels = Int(args[i + 1]) ?? 256; i += 2
        case "--width" where i + 1 < args.count:
            width = Int(args[i + 1]) ?? 256; i += 2
        case "--layers" where i + 1 < args.count:
            layers = Int(args[i + 1]) ?? 8; i += 2
        default:
            i += 1
        }
    }
    return (modelPath, compiledCache, channels, width, layers)
}

var out = ProbeResult()
let pi = ProcessInfo.processInfo
out.thermal_state = thermalLabel(pi.thermalState)
out.low_power_mode = pi.isLowPowerModeEnabled
out.ane_cores = detectAneCores()
out.ssd_read_gbs = (measureSsdReadGbs() * 100).rounded() / 100

var gemmCtx: GemmContext? = nil
var metalQueue: MTLCommandQueue? = nil
var gpuSoloMs: Double = 0

if let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() {
    metalQueue = queue
    out.gpu_name = device.name
    out.unified_memory = device.hasUnifiedMemory
    out.recommended_max_working_set_gb = (Double(device.recommendedMaxWorkingSetSize) / 1073741824.0 * 100).rounded() / 100
    out.bandwidth_gbs = (measureBandwidth(device: device, queue: queue) * 10).rounded() / 10
    if let g = makeGemm(device: device, n: 2048) {
        gemmCtx = g
        for _ in 0..<3 { _ = runGemmOnce(g, queue: queue) }
        var times: [Double] = []
        for _ in 0..<10 {
            let dt = runGemmOnce(g, queue: queue)
            if dt > 0 { times.append(dt) }
        }
        let med = percentile(times, 0.5)
        gpuSoloMs = med * 1000.0
        if med > 0 {
            out.gpu_fp16_tflops = ((g.flops / med) / 1e12 * 100).rounded() / 100
        }
    }
}

let cfg = parseArgs()
if let modelPath = cfg.modelPath {
    let fm = FileManager.default
    var compiledURL: URL? = nil
    if let cachePath = cfg.compiledCache, fm.fileExists(atPath: cachePath) {
        compiledURL = URL(fileURLWithPath: cachePath)
    } else {
        let tCompile = nowSeconds()
        if let tmpCompiled = try? MLModel.compileModel(at: URL(fileURLWithPath: modelPath)) {
            out.ane_compile_cold_ms = ((nowSeconds() - tCompile) * 1000.0 * 10).rounded() / 10
            if let cachePath = cfg.compiledCache {
                let dst = URL(fileURLWithPath: cachePath)
                try? fm.createDirectory(at: dst.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? fm.removeItem(at: dst)
                if (try? fm.copyItem(at: tmpCompiled, to: dst)) != nil {
                    compiledURL = dst
                } else {
                    compiledURL = tmpCompiled
                }
            } else {
                compiledURL = tmpCompiled
            }
        }
    }

    if let url = compiledURL, let input = try? makeInputProvider(channels: cfg.channels, width: cfg.width) {
        if let pct = anePlanCoverage(compiledURL: url) {
            out.ane_plan_cost_pct = (pct * 10).rounded() / 10
        }
        if let neRun = try? timeCoreML(compiledURL: url, units: .cpuAndNeuralEngine, input: input, warmup: 5, iters: 25) {
            var neP50 = neRun.p50Ms
            var neP95 = neRun.p95Ms
            if neP50 > 0 && neP95 / neP50 > 2.5 {
                out.ane_contention_detected = true
                Thread.sleep(forTimeInterval: 1.5)
                if let retry = try? timeCoreML(compiledURL: url, units: .cpuAndNeuralEngine, input: input, warmup: 3, iters: 20) {
                    neP50 = retry.p50Ms
                    neP95 = retry.p95Ms
                }
            }
            out.ane_load_warm_ms = (neRun.loadMs * 10).rounded() / 10
            out.ane_infer_p50_ms = (neP50 * 100).rounded() / 100
            out.ane_infer_p95_ms = (neP95 * 100).rounded() / 100
            let flops = Double(cfg.layers) * 2.0 * Double(cfg.channels) * Double(cfg.channels) * Double(cfg.width)
            if neP50 > 0 {
                out.ane_fp16_tops = ((flops / (neP50 * 1e-3)) / 1e12 * 100).rounded() / 100
            }
            if let cpuRun = try? timeCoreML(compiledURL: url, units: .cpuOnly, input: input, warmup: 1, iters: 5),
               neP50 > 0 {
                out.ane_speedup_vs_cpu = ((cpuRun.p50Ms / neP50) * 100).rounded() / 100
            }
            if let gpuRun = try? timeCoreML(compiledURL: url, units: .cpuAndGPU, input: input, warmup: 3, iters: 12),
               neP50 > 0 {
                out.ane_speedup_vs_gpu = ((gpuRun.p50Ms / neP50) * 100).rounded() / 100
            }
            let planOk = (out.ane_plan_cost_pct ?? 100.0) >= 80.0
            let speedOk = (out.ane_speedup_vs_cpu ?? 2.0) >= 1.5
            out.ane_verified = out.ane_cores > 0 && planOk && speedOk

            // Phase 3: Concurrent Metal GPU GEMM + Core ML ANE isolation probe (~600ms overlap).
            if let g = gemmCtx, let q = metalQueue, gpuSoloMs > 0, neP50 > 0 {
                let group = DispatchGroup()
                var gpuConc: [Double] = []
                var aneConc: [Double] = []
                let stopAt = nowSeconds() + 0.7
                group.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    while nowSeconds() < stopAt {
                        let dt = runGemmOnce(g, queue: q) * 1000.0
                        if dt > 0 { gpuConc.append(dt) }
                    }
                    group.leave()
                }
                group.enter()
                DispatchQueue.global(qos: .userInitiated).async {
                    while nowSeconds() < stopAt {
                        let t0 = nowSeconds()
                        _ = try? neRun.model.prediction(from: input)
                        aneConc.append((nowSeconds() - t0) * 1000.0)
                    }
                    group.leave()
                }
                group.wait()
                if gpuConc.count >= 3 && aneConc.count >= 3 {
                    let gMed = percentile(Array(gpuConc.dropFirst()), 0.5)
                    let aMed = percentile(Array(aneConc.dropFirst()), 0.5)
                    out.gpu_contention_pct = (((gMed - gpuSoloMs) / gpuSoloMs) * 100.0 * 10).rounded() / 10
                    out.ane_contention_pct = (((aMed - neP50) / neP50) * 100.0 * 10).rounded() / 10
                }
            }
        }
    }
}

let afterThermal = thermalLabel(pi.thermalState)
out.thermal_state = afterThermal
out.noisy = out.low_power_mode || afterThermal == "serious" || afterThermal == "critical"

let enc = JSONEncoder()
enc.outputFormatting = [.prettyPrinted, .sortedKeys]
if let data = try? enc.encode(out), let text = String(data: data, encoding: .utf8) {
    print(text)
}
