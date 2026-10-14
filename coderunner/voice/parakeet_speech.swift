// Speech to text with Parakeet on Core ML (Neural Engine), through FluidAudio's library.
//
// Built by coderunner/voice/parakeet.py with SwiftPM. Usage:
//   coderunner-parakeet transcribe <16 kHz WAV> <model folder> <language or "auto">
//       one JSON line per phrase: {"start", "end", "text"}
//   coderunner-parakeet prepare <model folder>     loads the model once, so Core ML compiles it for this Mac
// The model folder holds the Core ML files voice setup downloaded; nothing is fetched here.
// Exit codes: 0 ok, 2 usage, 4 failed.

import CoreML
import FluidAudio
import Foundation

let version = AsrModelVersion.ultra
let longestPhrase = 30.0     // seconds; a phrase without punctuation is cut here
let pause = 0.8              // seconds of silence that end a phrase

func fail(_ message: String, _ code: Int32) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code)
}

func load(_ folder: String) async throws -> AsrManager {
    let models = try AsrModels.loadLocal(from: URL(fileURLWithPath: folder), version: version)
    let manager = AsrManager(config: ASRConfig(tdtConfig: TdtConfig(blankId: version.blankId)))
    try await manager.loadModels(models)
    return manager
}

func emit(_ start: Double, _ end: Double, _ text: String) {
    let words = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    guard !words.isEmpty else { return }
    let line: [String: Any] = ["start": (start * 100).rounded() / 100, "end": (end * 100).rounded() / 100,
                               "text": words]
    print(String(data: try! JSONSerialization.data(withJSONObject: line), encoding: .utf8)!)
}

/// Phrases from the timed tokens: a phrase ends at . ? or !, at a pause, or after `longestPhrase` seconds.
func phrases(_ result: ASRResult) {
    guard let tokens = result.tokenTimings, !tokens.isEmpty else {
        emit(0, result.duration, result.text)
        return
    }
    var text = "", start = tokens[0].startTime, end = start
    for token in tokens {
        // A character the vocabulary lacks ("&" in "Q&A") comes as <unk>: it keeps the words apart.
        let piece = token.token.replacingOccurrences(of: "<unk>", with: " ")
        if !text.isEmpty && piece.hasPrefix(" ") && (token.startTime - end > pause || token.startTime - start > longestPhrase) {
            emit(start, end, text)
            text = ""
        }
        if text.isEmpty { start = token.startTime }
        text += piece
        end = token.endTime
        if let last = piece.last, ".?!".contains(last) {
            emit(start, end, text)
            text = ""
        }
    }
    emit(start, end, text)
}

let args = CommandLine.arguments
switch args.count > 1 ? args[1] : "" {
case "transcribe" where args.count == 5:
    do {
        let manager = try await load(args[3])
        var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        let language = Language(rawValue: String(args[4].prefix(2)).lowercased())   // a hint for the script
        phrases(try await manager.transcribe(URL(fileURLWithPath: args[2]), decoderState: &state, language: language))
    } catch {
        fail("transcription failed: \(error.localizedDescription)", 4)
    }
case "prepare" where args.count == 3:
    do {
        _ = try await load(args[2])
    } catch {
        fail("could not load the speech model: \(error.localizedDescription)", 4)
    }
default:
    fail("usage: coderunner-parakeet transcribe <wav> <model folder> <language|auto> | prepare <model folder>", 2)
}
