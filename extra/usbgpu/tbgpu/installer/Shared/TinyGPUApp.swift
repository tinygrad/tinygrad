import SwiftUI

class AppState: ObservableObject {
  @Published var text = ""
  @Published var buttonText: String?
  @Published var buttonAction: (() -> Void)?
  @Published var pciDevices = ""
  private var statusTimer: Timer?
  private var pciTimer: Timer?
  var onInstallComplete: (() -> Void)?

  func append(_ s: String) { DispatchQueue.main.async { self.text += s } }
  func setButton(_ label: String?, action: (() -> Void)?) {
    DispatchQueue.main.async { self.buttonText = label; self.buttonAction = action }
  }

  func startMonitoringInstallation(dextIdentifier: String) {
    statusTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] timer in
      let state = TinyGPUCLIRunner.queryDextState(bundleID: dextIdentifier)
      if state == .activated {
        timer.invalidate()
        self?.onInstallComplete?()
      } else {
        timer.invalidate()
      }
    }
  }

  func startPCIMonitoring() {
    updatePCIDevices()
    pciTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
      self?.updatePCIDevices()
    }
  }

  func updatePCIDevices() {
    DispatchQueue.main.async {
      self.pciDevices = TinyGPUCLIRunner.getPCIDevices()
    }
  }

  func stopMonitoring() {
    statusTimer?.invalidate()
    statusTimer = nil
    pciTimer?.invalidate()
    pciTimer = nil
  }
}

struct ContentView: View {
  @ObservedObject var state: AppState

  var body: some View {
    VStack(spacing: 12) {
      ScrollView {
        VStack(alignment: .leading, spacing: 0) {
          Text(state.text)
          if !state.pciDevices.isEmpty {
            Text(state.pciDevices)
          }
        }
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
      let runner = TinyGPUCLIRunner(dextIdentifier: "org.tinygrad.tinygpu.edriver")
      runner.run(args: CommandLine.arguments) { exitCode in
        exit(exitCode.rawValue)
      }

      // Keep process alive for async operations, but exit cleanly when done
      dispatchMain()
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

    // require /Applications
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
    state.append(TinyGPUCLIRunner.getStatusText(dextState))

    // Add buttons based on state
    if dextState == .unloaded || dextState == .activationError {
      state.setButton("Install Extension") {
        self.state.text = "Installing extension...\n\nYou may be prompted for approval in System Settings.\n"
        self.state.setButton(nil, action: nil)

        let runner = TinyGPUCLIRunner(dextIdentifier: dextIdentifier)
        runner.install { _ in }

        self.state.onInstallComplete = {
          DispatchQueue.main.async {
            self.state.text = ""
            self.setup()
          }
        }
        self.state.startMonitoringInstallation(dextIdentifier: dextIdentifier)
      }
    } else if dextState == .needsApproval || dextState == .activating {
      state.setButton(nil, action: nil)
      state.onInstallComplete = {
        DispatchQueue.main.async {
          self.state.text = ""
          self.setup()
        }
      }
      state.startMonitoringInstallation(dextIdentifier: dextIdentifier)
    } else {
      state.setButton(nil, action: nil)
    }

    state.startPCIMonitoring()
  }
}

class AppDelegate: NSObject, NSApplicationDelegate {
  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}
