// CodeRunner's native helper for the file index (macOS): text in images with Vision, PDF pages with PDFKit
// (scanned pages read with Vision), and file events with FSEvents. Output is one JSON object per line.
//
//   files-helper ocr                     image paths on stdin, one per line -> {"path", "text"}
//   files-helper pdf <path> <max> <0|1>  -> {"pages": [...], "total": n}; 1 reads scanned pages with OCR
//   files-helper watch <folder>...       -> {"paths": [...]} as files change; exits when stdin closes

import AppKit
import CoreServices
import Foundation
import ImageIO
import PDFKit
import Vision

func emit(_ object: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: object),
          let line = String(data: data, encoding: .utf8) else { return }
    FileHandle.standardOutput.write((line + "\n").data(using: .utf8)!)
}

// After the Mac's disk filled, the accurate model failed on every image ("nilError"; the system's model compiler
// had stopped working) until a restart, and every picture read as empty. The fast model still worked, so it
// reads the picture when the accurate one fails.
var accurateFailed = false

func recognize(_ image: CGImage) -> String {
    func read(_ level: VNRequestTextRecognitionLevel) throws -> String {
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = level
        request.usesLanguageCorrection = true
        if #available(macOS 13.0, *) { request.automaticallyDetectsLanguage = true }
        try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
        return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(separator: "\n")
    }
    if !accurateFailed, let text = try? read(.accurate) { return text }
    accurateFailed = true               // tried again when the helper next starts
    return (try? read(.fast)) ?? ""
}

func image(at path: String) -> CGImage? {
    guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
    // At most 2048 pixels: camera photos are read in a fraction of the time, and text stays legible.
    let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                    kCGImageSourceThumbnailMaxPixelSize: 2048,
                                    kCGImageSourceCreateThumbnailWithTransform: true]
    guard let picture = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else { return nil }
    // Transparent pixels read as black, which hides dark text: put the picture on white first.
    guard picture.alphaInfo != .none, picture.alphaInfo != .noneSkipFirst, picture.alphaInfo != .noneSkipLast,
          let canvas = CGContext(data: nil, width: picture.width, height: picture.height, bitsPerComponent: 8,
                                 bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(),
                                 bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return picture }
    let area = CGRect(x: 0, y: 0, width: picture.width, height: picture.height)
    canvas.setFillColor(CGColor(gray: 1, alpha: 1))
    canvas.fill(area)
    canvas.draw(picture, in: area)
    return canvas.makeImage() ?? picture
}

func ocr() {
    while let line = readLine() {
        let path = line.trimmingCharacters(in: .whitespacesAndNewlines)
        if path.isEmpty { continue }
        autoreleasepool { emit(["path": path, "text": image(at: path).map(recognize) ?? ""]) }
    }
}

func pdf(_ path: String, maxPages: Int, readScans: Bool) {
    guard let document = PDFDocument(url: URL(fileURLWithPath: path)) else {
        emit(["error": "cannot open the PDF (damaged or password protected)"])
        return
    }
    // Bank and tax statements are often locked with a password: their pages render blank and read as nothing,
    // which looked like an empty document. Say so; the file stays findable by its name.
    if document.isLocked && !document.unlock(withPassword: "") {
        emit(["error": "the PDF is password protected; only its name is searchable"])
        return
    }
    var pages: [String] = []
    for index in 0..<min(document.pageCount, maxPages) {
        autoreleasepool {
            guard let page = document.page(at: index) else { pages.append(""); return }
            var text = page.string ?? ""
            if readScans && text.trimmingCharacters(in: .whitespacesAndNewlines).count < 20 {
                let bounds = page.bounds(for: .mediaBox)
                let scale = 2048 / max(bounds.width, bounds.height, 1)
                let picture = page.thumbnail(of: CGSize(width: bounds.width * scale, height: bounds.height * scale),
                                             for: .mediaBox)
                if let scan = picture.cgImage(forProposedRect: nil, context: nil, hints: nil) { text = recognize(scan) }
            }
            pages.append(text)
        }
    }
    emit(["pages": pages, "total": document.pageCount])
}

func watch(_ folders: [String]) {
    let callback: FSEventStreamCallback = { _, _, count, paths, _, _ in
        let changed = (unsafeBitCast(paths, to: NSArray.self) as? [String]) ?? []
        emit(["paths": Array(changed.prefix(count))])
    }
    var context = FSEventStreamContext()
    let flags = kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagUseCFTypes | kFSEventStreamCreateFlagNoDefer
    guard let stream = FSEventStreamCreate(nil, callback, &context, folders as CFArray,
                                           FSEventStreamEventId(kFSEventStreamEventIdSinceNow), 0.5,
                                           FSEventStreamCreateFlags(flags)) else { exit(1) }
    FSEventStreamSetDispatchQueue(stream, DispatchQueue.main)
    FSEventStreamStart(stream)
    // The parent holds stdin open; when it goes away, so does the watcher.
    DispatchQueue.global().async {
        while readLine() != nil {}
        exit(0)
    }
    dispatchMain()
}

setvbuf(stdout, nil, _IOLBF, 0)
let arguments = CommandLine.arguments
switch arguments.count > 1 ? arguments[1] : "" {
case "ocr":
    ocr()
case "pdf" where arguments.count >= 3:
    pdf(arguments[2], maxPages: Int(arguments.count > 3 ? arguments[3] : "500") ?? 500,
        readScans: arguments.count > 4 && arguments[4] == "1")
case "watch" where arguments.count >= 3:
    watch(Array(arguments.dropFirst(2)))
default:
    FileHandle.standardError.write("usage: files-helper ocr | pdf <path> <max> <0|1> | watch <folder>...\n"
        .data(using: .utf8)!)
    exit(2)
}
