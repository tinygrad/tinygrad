import AppKit

private let dextIdentifier = "org.tinygrad.tinygpu.edriver"

private var runner: TinyGPUCLIRunner!

@main
struct TinyGPUApp {
  static func main() {
    let app = NSApplication.shared
    app.setActivationPolicy(.prohibited)

    runner = TinyGPUCLIRunner(dextIdentifier: dextIdentifier)
    runner.run(args: CommandLine.arguments) { exitCode in
      exit(exitCode.rawValue)
    }

    app.run()
  }
}
