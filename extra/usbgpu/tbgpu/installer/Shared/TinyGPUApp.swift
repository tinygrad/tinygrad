import SwiftUI

@main
struct TinyGPUApp: App {
  @State private var text = ""
  @State private var buttonText: String? = nil
  @State private var buttonAction: (() -> Void)? = nil
  @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

  init() {
    if CommandLine.arguments.count > 1 {
      let runner = TinyGPUCLIRunner(dextIdentifier: "org.tinygrad.tinygpu.edriver")
      runner.run(args: CommandLine.arguments) { exit($0.rawValue) }
      dispatchMain()
    }
  }

  var body: some Scene {
    WindowGroup("TinyGPU") {
      VStack(spacing: 12) {
        ScrollView {
          Text(text).font(.custom("Menlo", size: 11)).frame(maxWidth: .infinity, alignment: .leading).padding(8)
        }
        if let label = buttonText {
          Button(label) { buttonAction?() }.buttonStyle(.borderedProminent).controlSize(.large)
        }
      }
      .frame(width: 500, height: 300).padding()
      .onAppear { setup() }
    }
    .commands { CommandGroup(replacing: .newItem) {} }
  }

  func setup() {
    let dextIdentifier = "org.tinygrad.tinygpu.edriver"
    let bundlePath = Bundle.main.bundlePath

    if !bundlePath.hasPrefix("/Applications/") {
      text = "TinyGPU needs to be in /Applications/\n\n"
      buttonText = "Move to /Applications"
      buttonAction = {
        var error: NSDictionary?
        NSAppleScript(source: "do shell script \"mv '\(bundlePath)' '/Applications/'\" with administrator privileges")?.executeAndReturnError(&error)
        text = error == nil ? "Moved! Please reopen from /Applications/\n" : "Failed: \(error?["NSAppleScriptErrorMessage"] ?? "")\n"
        buttonText = nil
      }
      return
    }

    text = "TinyGPU - Remote PCI Device Server\n\n" + TinyGPUCLIRunner.getStatusText(TinyGPUCLIRunner.queryDextState(bundleID: dextIdentifier))
  }
}

class AppDelegate: NSObject, NSApplicationDelegate {
  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
