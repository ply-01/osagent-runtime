// Verify metadata, decode every frame, and export representative decoded PNGs.
import Foundation
import AVFoundation
import CoreImage
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
if args.count != 3 { fatalError("usage: swift verify-track1-video.swift VIDEO QA_DIR") }
let asset=AVURLAsset(url:URL(fileURLWithPath:args[1]))
let folder=URL(fileURLWithPath:args[2]);try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
let tracks=asset.tracks(withMediaType:.video)
guard let track=tracks.first else { fatalError("missing video track") }
let reader=try AVAssetReader(asset:asset)
let output=AVAssetReaderTrackOutput(track:track,outputSettings:[kCVPixelBufferPixelFormatTypeKey as String:kCVPixelFormatType_32BGRA])
reader.add(output);guard reader.startReading() else { fatalError("reader failed") }
let context=CIContext(options:[.useSoftwareRenderer:true]);var count=0
while let sample=output.copyNextSampleBuffer() {
    if [0,1620,3150,5399].contains(count),let buffer=CMSampleBufferGetImageBuffer(sample) {
        let image=CIImage(cvPixelBuffer:buffer)
        if let cg=context.createCGImage(image,from:image.extent) {
            let url=folder.appendingPathComponent(String(format:"decoded-%04d.png",count))
            let destination=CGImageDestinationCreateWithURL(url as CFURL,UTType.png.identifier as CFString,1,nil)!
            CGImageDestinationAddImage(destination,cg,nil);CGImageDestinationFinalize(destination)
        }
    }
    count+=1
}
guard reader.status == .completed else { fatalError("decode failed: \(String(describing:reader.error))") }
let duration=CMTimeGetSeconds(asset.duration)
let result:[String:Any]=["decoded_frames":count,"duration_seconds":duration,"width":Int(track.naturalSize.width),"height":Int(track.naturalSize.height),"nominal_frame_rate":track.nominalFrameRate,"audio_tracks":asset.tracks(withMediaType:.audio).count,"passed":count==5400 && abs(duration-180)<0.1]
let data=try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys])
try data.write(to:folder.appendingPathComponent("video-validation.json"))
print(String(data:data,encoding:.utf8)!)
