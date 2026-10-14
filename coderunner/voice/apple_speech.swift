// On-device speech to text with Apple's SpeechAnalyzer (macOS 26 and later).
//
// Built by coderunner/native.py on first use. Usage:
//   coderunner-speech transcribe <audio file> <locale or "auto">   one JSON line per phrase: {"start", "end", "text"}
//   coderunner-speech prepare <locale or "auto">                   installs the language's model if needed
//   coderunner-speech locales                                      supported and installed locales as JSON
// The locale's speech model is downloaded and installed by the system when it is missing.
// Exit codes: 0 ok, 2 usage, 3 unavailable on this Mac or for this language, 4 failed.

import AVFoundation
import CoreMedia
import Foundation
import Speech

func fail(_ message: String, _ code: Int32) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code)
}

func pickLocale(_ wanted: String) async -> Locale? {
    let locale = wanted == "auto" || wanted.isEmpty ? Locale.current : Locale(identifier: wanted)
    if let exact = await SpeechTranscriber.supportedLocale(equivalentTo: locale) {
        return exact
    }
    // A bare language ("en", "hi") or an unsupported region: take a supported locale of that language,
    // preferring the user's own region.
    let language = locale.language.languageCode?.identifier ?? wanted
    let region = Locale.current.region?.identifier
    let same = await SpeechTranscriber.supportedLocales.filter { $0.language.languageCode?.identifier == language }
    return same.first { $0.region?.identifier == region } ?? same.first
}

func prepare(_ wanted: String) async throws -> SpeechTranscriber {
    guard SpeechTranscriber.isAvailable else { fail("speech transcription is not available on this Mac", 3) }
    guard let locale = await pickLocale(wanted) else { fail("no on-device speech model for \(wanted)", 3) }
    let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
    // Listed locales can still have no model for this Mac (Hindi: installation reports "Not Installing").
    if await AssetInventory.status(forModules: [transcriber]) == .unsupported {
        fail("no on-device speech model for \(wanted) on this Mac", 3)
    }
    if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
        try await request.downloadAndInstall()
    }
    return transcriber
}

func line(_ result: SpeechTranscriber.Result) -> String {
    let segment: [String: Any] = ["start": result.range.start.seconds, "end": CMTimeRangeGetEnd(result.range).seconds,
                                  "text": String(result.text.characters)]
    return String(data: try! JSONSerialization.data(withJSONObject: segment), encoding: .utf8)!
}

func transcribe(_ path: String, _ wanted: String) async throws {
    let transcriber = try await prepare(wanted)
    let file = try AVAudioFile(forReading: URL(fileURLWithPath: path))
    let analyzer = SpeechAnalyzer(modules: [transcriber])
    let printer = Task {
        for try await result in transcriber.results where result.isFinal {
            print(line(result))
        }
    }
    if let last = try await analyzer.analyzeSequence(from: file) {
        try await analyzer.finalizeAndFinish(through: last)
    } else {
        await analyzer.cancelAndFinishNow()
    }
    try await printer.value
}

let args = CommandLine.arguments
switch args.count > 1 ? args[1] : "" {
case "transcribe" where args.count == 4:
    do {
        try await transcribe(args[2], args[3])
    } catch {
        fail("transcription failed: \(error.localizedDescription)", 4)
    }
case "prepare" where args.count == 3:
    do {
        _ = try await prepare(args[2])
    } catch {
        fail("could not install the speech model: \(error.localizedDescription)", 4)
    }
case "locales":
    let supported = await SpeechTranscriber.supportedLocales.map { $0.identifier(.bcp47) }
    let installed = await SpeechTranscriber.installedLocales.map { $0.identifier(.bcp47) }
    let info: [String: Any] = ["available": SpeechTranscriber.isAvailable, "supported": supported.sorted(),
                               "installed": installed.sorted()]
    print(String(data: try! JSONSerialization.data(withJSONObject: info), encoding: .utf8)!)
default:
    fail("usage: coderunner-speech transcribe <audio file> <locale|auto> | prepare <locale|auto> | locales", 2)
}
