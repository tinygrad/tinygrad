import SwiftUI

class AppState: ObservableObject {
  @Published var text = ""
  @Published var buttonText: String?
  @Published var buttonAction: (() -> Void)?
  private var statusTimer: Timer?
  var onInstallComplete: (() -> Void)?

  func append(_ s: String) { DispatchQueue.main.async { self.text += s } }
  func setButton(_ label: String?, action: (() -> Void)?) {
    DispatchQueue.main.async { self.buttonText = label; self.buttonAction = action }
  }

  func startMonitoringInstallation(dextIdentifier: String) {
    var attempts = 0
    statusTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] timer in
      attempts += 1
      let state = TinyGPUCLIRunner.queryDextState(bundleID: dextIdentifier)
      if state == .activated {
        timer.invalidate()
        self?.onInstallComplete?()
      } else if attempts > 15 {
        timer.invalidate()
      }
    }
  }

  func stopMonitoring() {
    statusTimer?.invalidate()
    statusTimer = nil
  }
}

struct ContentView: View {
  @ObservedObject var state: AppState

  var body: some View {
    VStack(spacing: 12) {
      ScrollView {
        Text(state.text)
          .font(.custom("Menlo", size: 11))
          .frame(maxWidth: .infinity, alignment: .leading)
          .padding(8)
      }
      if let label = state.buttonText {
        Button(label) { state.buttonAction?() }
          .buttonStyle(.borderedProminent)
          .controlSize(.large)
      }
    }
    .frame(width: 500, height: 300)
    .padding()
  }
}

@main
struct TinyGPUApp: App {
  @StateObject private var state = AppState()
  @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

  init() {
    if CommandLine.arguments.count > 1 {
      NSApplication.shared.setActivationPolicy(.prohibited)
      let runner = TinyGPUCLIRunner(dextIdentifier: "org.tinygrad.tinygpu.edriver")
      runner.run(args: CommandLine.arguments) { exit($0.rawValue) }
      NSApplication.shared.run()
    }
  }

  var body: some Scene {
    WindowGroup("TinyGPU") {
      ContentView(state: state)
        .onAppear { setup() }
    }
    .commands { CommandGroup(replacing: .newItem) {} }
  }

  func setup() {
    let dextIdentifier = "org.tinygrad.tinygpu.edriver"
    let bundlePath = Bundle.main.bundlePath
    let inApps = bundlePath.hasPrefix("/Applications/")

    if !inApps {
      state.append("TinyGPU needs to be in /Applications/\n\n")
      state.setButton("Move to /Applications") {
        let src = bundlePath, dst = "/Applications/"
        let script = "do shell script \"mv '\(src)' '\(dst)'\" with administrator privileges"
        var error: NSDictionary?
        NSAppleScript(source: script)?.executeAndReturnError(&error)
        if error == nil {
          self.state.text = "Moved to /Applications!\n\nPlease reopen TinyGPU.app from /Applications/\n"
          self.state.setButton(nil, action: nil)
        } else {
          self.state.text = "Failed to move: \(error?["NSAppleScriptErrorMessage"] ?? "unknown error")\n\nManually delete /Applications/TinyGPU.app and try again.\n"
        }
      }
      return
    }

    state.append("TinyGPU - Remote PCI Device Server\n\n")

    let dextState = TinyGPUCLIRunner.queryDextState(bundleID: dextIdentifier)

    if dextState == .unloaded || dextState == .activationError {
      state.append("Driver extension not installed.\n\n")
      state.setButton("Install Extension") {
        self.state.text = "Installing extension...\n\nYou may be prompted for approval in System Settings.\n"
        self.state.setButton(nil, action: nil)

        let runner = TinyGPUCLIRunner(dextIdentifier: dextIdentifier)
        runner.run(args: [bundlePath + "/Contents/MacOS/TinyGPU", "install"]) { _ in }

        self.state.onInstallComplete = {
          DispatchQueue.main.async {
            self.state.text = ""
            self.setup()
          }
        }
        self.state.startMonitoringInstallation(dextIdentifier: dextIdentifier)
      }
    } else if dextState == .needsApproval || dextState == .activating {
      state.append("Extension is awaiting approval.\n\n")
      state.append("Please go to System Settings > Privacy & Security and approve the extension.\n\n")
      state.setButton(nil, action: nil)

      state.onInstallComplete = {
        DispatchQueue.main.async {
          self.state.text = ""
          self.setup()
        }
      }
      state.startMonitoringInstallation(dextIdentifier: dextIdentifier)
    } else {
      state.append("Ready! Run tinygrad to use your eGPU.\n\n")
      state.append("Troubleshooting:\n")
      state.append("  • Check GPU: System Report > PCI\n")
      state.append("  • Approve extension: System Settings > Privacy & Security\n")
      state.setButton(nil, action: nil)
    }
  }
}

class AppDelegate: NSObject, NSApplicationDelegate {
  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
