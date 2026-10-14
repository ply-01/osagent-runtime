// Local artifact encoder using Apple's AVFoundation. No UI or screen control.
import Foundation
import AVFoundation
import CoreGraphics
import CoreVideo
import ImageIO

let args = CommandLine.arguments
if args.count != 4 { fatalError("usage: swift encode-track1-video.swift FRAMES CHAPTERS_JSON OUTPUT") }
let frames = URL(fileURLWithPath: args[1])
let metadata = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args[2]))) as! [String:Any]
let chapters = metadata["chapters"] as! [[String:Any]]
let output = URL(fileURLWithPath: args[3])
if FileManager.default.fileExists(atPath: output.path) { try FileManager.default.removeItem(at: output) }
let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
let settings: [String:Any] = [AVVideoCodecKey:AVVideoCodecType.h264, AVVideoWidthKey:1280, AVVideoHeightKey:720,
    AVVideoCompressionPropertiesKey:[AVVideoAverageBitRateKey:1_500_000, AVVideoExpectedSourceFrameRateKey:30]]
let input = AVAssetWriterInput(mediaType:.video, outputSettings:settings)
input.expectsMediaDataInRealTime = false
let attributes: [String:Any] = [kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA,
    kCVPixelBufferWidthKey as String:1280,kCVPixelBufferHeightKey as String:720,
    kCVPixelBufferCGImageCompatibilityKey as String:true,kCVPixelBufferCGBitmapContextCompatibilityKey as String:true]
let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput:input,sourcePixelBufferAttributes:attributes)
writer.add(input)
guard writer.startWriting() else { fatalError("writer: \(String(describing:writer.error))") }
writer.startSession(atSourceTime:.zero)
var frameIndex = 0
for (index, chapter) in chapters.enumerated() {
    let file = frames.appendingPathComponent(String(format:"frame-%02d.png",index))
    let source = CGImageSourceCreateWithURL(file as CFURL,nil)!
    let image = CGImageSourceCreateImageAtIndex(source,0,nil)!
    var pixelBuffer: CVPixelBuffer?
    guard CVPixelBufferCreate(kCFAllocatorDefault,1280,720,kCVPixelFormatType_32BGRA,attributes as CFDictionary,&pixelBuffer) == kCVReturnSuccess else { fatalError("pixel allocation") }
    let buffer = pixelBuffer!
    CVPixelBufferLockBaseAddress(buffer,[])
    let context = CGContext(data:CVPixelBufferGetBaseAddress(buffer),width:1280,height:720,bitsPerComponent:8,
        bytesPerRow:CVPixelBufferGetBytesPerRow(buffer),space:CGColorSpaceCreateDeviceRGB(),
        bitmapInfo:CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)!
    context.draw(image,in:CGRect(x:0,y:0,width:1280,height:720))
    CVPixelBufferUnlockBaseAddress(buffer,[])
    let duration = chapter["duration"] as! Int
    for _ in 0..<(duration*30) {
        while !input.isReadyForMoreMediaData {
            if writer.status == .failed { fatalError("writer: \(String(describing:writer.error))") }
            Thread.sleep(forTimeInterval:0.005)
        }
        guard adaptor.append(buffer,withPresentationTime:CMTime(value:Int64(frameIndex),timescale:30)) else { fatalError("append: \(String(describing:writer.error))") }
        frameIndex += 1
    }
}
input.markAsFinished()
let completed = DispatchSemaphore(value:0)
writer.finishWriting { completed.signal() }
completed.wait()
guard writer.status == .completed else { fatalError("finish: \(String(describing:writer.error))") }
print("Encoded \(frameIndex) frames at 30 fps; duration \(Double(frameIndex)/30) seconds")
