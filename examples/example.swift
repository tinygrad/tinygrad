#!/usr/bin/swift

import Foundation
import SwiftUI
import Metal


struct Tensor: Decodable {
    let id: String?
    let size: Int
    let dtype: String
}

struct Statement: Decodable {
    let kernel: String
    let args: [String]
    var global_size: [Int]
    var local_size: [Int]
}

struct Model: Decodable {
    let backend: String
    let input_size: Tensor
    let output_size: Tensor 
    let functions: [String: String]
    let statements: [Statement]
    let buffers: [String: Tensor]
}

struct TensorMetadata: Decodable {
    let dtype: String
    let shape: [Int]
    let data_offsets: [Int]
}

struct SafeTensor {
    let metadata: [String: TensorMetadata]
    let raw : Data

    func getTensor(name: String) -> [Float32] {
        let metadata = self.metadata[name]!
        let data_offsets = metadata.data_offsets
        let data = self.raw.subdata(in: data_offsets[0]..<data_offsets[1])
        return data.withUnsafeBytes {
            (pointer: UnsafePointer<Float32>) -> [Float32] in
            let buffer = UnsafeBufferPointer(start: pointer, count: data.count / MemoryLayout<Float32>.size)
            return Array<Float32>(buffer)
        }
    }
}

let safetensorFile = "net.safetensors"
let modelfile = "net.json"
let imageFile = "../docs/showcase/stable_diffusion_by_tinygrad.jpg"
let labelsFile = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
var model: Model
var safetensor: SafeTensor

var image: NSImage
var labels: [String]

do {
    let modelUrl = URL(fileURLWithPath: modelfile)
    let rawModel = try Data(contentsOf: modelUrl)
    model = try JSONDecoder().decode(Model.self, from: rawModel)

    let safetensorUrl = URL(fileURLWithPath: safetensorFile)
    let rawSafeTensors = try Data(contentsOf: safetensorUrl)
    let header = rawSafeTensors.subdata(in: 0..<8)
    let header_size = header.withUnsafeBytes {
        return $0.load(as: UInt64.self)
    }
    let json_data = rawSafeTensors.subdata(in: 8..<8+Int(header_size))
    let safetensormetadata = try JSONDecoder().decode([String: TensorMetadata].self, from: json_data)
    safetensor = SafeTensor(metadata: safetensormetadata, raw: rawSafeTensors.subdata(in: 8+Int(header_size)..<rawSafeTensors.count))

    let imageUrl = URL(fileURLWithPath: imageFile)
    let rawImage = try Data(contentsOf: imageUrl)
    image = NSImage(data: rawImage)!
    let newImage = NSImage(size: NSMakeSize(244, 244))
    newImage.lockFocus()
    image.draw(in: NSMakeRect(0, 0, 244, 244), from: NSMakeRect(0, 0, image.size.width, image.size.height), operation: NSCompositingOperation.copy, fraction: 1.0)
    newImage.unlockFocus()
    image = newImage

    let labelsUrl = URL(string: labelsFile)!
    let rawLabels = try Data(contentsOf: labelsUrl)
    labels = String(data: rawLabels, encoding: .utf8)!.split(separator: "\n").map { String($0) }
    print("LOADED LABELS \(labels.count)")
} catch {
    print("Error: \(error)")
    exit(1)
}
let device = try MTLCopyAllDevices().first!
let queue = device.makeCommandQueue()!
var kernels : [String: MTLComputePipelineState] = [:]
print("LOADING THE KERNELS")
for (name, src) in model.functions {
    do {
        let lib = try device.makeLibrary(source: src, options: nil) 
        let kernel = lib.makeFunction(name: name)!
        let pipeline = try device.makeComputePipelineState(function: kernel)
        kernels[name] = pipeline
    } catch {
        print("Error: \(name) \(error)")
        exit(1)
    }
}
print("FINISHED LOADING THE KERNELS")

print("LOADING BUFFERS")
var buffers : [String: MTLBuffer] = [:]

let pixelData = (image.cgImage(forProposedRect: nil, context: nil, hints: nil)!).dataProvider!.data
let rawimageData: UnsafePointer<UInt8> = CFDataGetBytePtr(pixelData)
var imageData: [Float32] = []
for i in 0..<244*244*4 {
    imageData.append((Float32(rawimageData[i]) / 255)*0.45 - 0.225)
}
var finalImageData: [Float32] = []
for c in 0..<3 {
    for i in 0..<244*244 {
        finalImageData.append(imageData[i*4+c])
    }
}
print("FINISHED LOADING IMAGE DATA \(finalImageData[0]) \(finalImageData[1]) \(finalImageData[2])")

buffers["input"] = device.makeBuffer(bytes: finalImageData, length: model.input_size.size * MemoryLayout<Float32>.size, options: [])!
buffers["outputs"] = device.makeBuffer(length: model.output_size.size * MemoryLayout<Float32>.size, options: [])!
for (name, buf) in model.buffers {
    let id = buf.id!
    if id != "" {
        buffers[name] = device.makeBuffer(bytes: safetensor.getTensor(name: id), length: buf.size*MemoryLayout<Float32>.size, options: [])
    } else {
        buffers[name] = device.makeBuffer(length: buf.size * MemoryLayout<Float32>.size, options: [])!
    }
}
print("FINISHED LOADING BUFFERS")

for statement in model.statements {
    let kernel = kernels[statement.kernel]!
    let commandBuffer = queue.makeCommandBuffer()!
    let encoder = commandBuffer.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(kernel)
    var i = 0
    for arg in statement.args {
        encoder.setBuffer(buffers[arg], offset: 0, index: i)
        i += 1
    }
    // print("\(statement.kernel) \(statement.global_size) \(statement.local_size)")
    // TODO: Clean this up a bit
    var local_size = MTLSizeMake(1, 1, 1)
    var global_size = MTLSizeMake(1, 1, 1)
    for (i,v) in statement.local_size.enumerated() {
         if i == 0 {
            local_size.width = v
        } else if i == 1 {
            local_size.height = v
        } else if i == 2 {
            local_size.depth = v
        }
    }
    for (i,v) in statement.global_size.enumerated() {
        if i == 0 {
            global_size.width = v
        } else if i == 1 {
            global_size.height = v
        } else if i == 2 {
            global_size.depth = v
        }
    }
    encoder.dispatchThreadgroups(global_size, threadsPerThreadgroup: local_size)
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
} 

var output = buffers["outputs"]!.contents().bindMemory(to: Float32.self, capacity: model.output_size.size * MemoryLayout<Float32>.size)
var ix = 0
for i in 0..<model.output_size.size {
    if output[i] > output[ix] {
        ix = i
    }
}
print("OUTPUT: \(labels[ix]) \(output[ix])")