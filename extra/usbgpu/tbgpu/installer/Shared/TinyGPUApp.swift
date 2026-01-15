import SwiftUI

class LogModel: ObservableObject {
  @Published var text = ""
  func append(_ str: String) {
    DispatchQueue.main.async {
      self.text += str
    }
  }
}

struct LogView: View {
  @ObservedObject var model: LogModel
  @State private var action: String?
  @State private var busy = false

  var body: some View {
    VStack(spacing: 0) {
      ScrollViewReader { proxy in
        ScrollView {
          Text(model.text)
            .font(.system(size: 11, design: .monospaced))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(8)
            .id("log")
        }
        .onChange(of: model.text) { _ in proxy.scrollTo("log", anchor: .bottom) }
      }

      if let action = action {
        Button(action: { busy = true; runCmd(action); DispatchQueue.main.asyncAfter(deadline: .now() + 1) { busy = false; checkStatus() } }) {
          Text(busy ? "wait..." : "\(action) extension>")
            .font(.system(size: 13, design: .monospaced))
            .foregroundColor(.blue)
            .padding(.vertical, 8)
        }
        .buttonStyle(.plain)
        .disabled(busy)
      }
    }
    .frame(width: 600, height: 400)
    .onAppear { checkStatus() }
  }

  func checkStatus() {
    guard Bundle.main.bundlePath.hasPrefix("/Applications/") else { return }
    action = runCmd("status").contains("Not installed") ? "install" : nil
  }

  func runCmd(_ cmd: String) -> String {
    let p = Process(); p.executableURL = URL(fileURLWithPath: "/Applications/TinyGPU.app/Contents/MacOS/TinyGPU"); p.arguments = [cmd]
    let pipe = Pipe(); p.standardOutput = pipe; try? p.run(); p.waitUntilExit()
    return String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
  }
}

@main
struct TinyGPUApp: App {
  @StateObject private var logModel = LogModel()
  @State private var runner: TinyGPUCLIRunner?
  @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

  init() {
    let hasArgs = CommandLine.arguments.count > 1

    if hasArgs {
      NSApplication.shared.setActivationPolicy(.prohibited)

      let runner = TinyGPUCLIRunner(dextIdentifier: "org.tinygrad.tinygpu.edriver")
      runner.run(args: CommandLine.arguments) { exitCode in
        exit(exitCode.rawValue)
      }
      NSApplication.shared.run()
    }
  }

  var body: some Scene {
    WindowGroup("TinyGPU") {
      LogView(model: logModel)
        .onAppear {
          let appPath = Bundle.main.bundlePath
          let isInApplications = appPath.hasPrefix("/Applications/TinyGPU.app")

          if isInApplications {
            logModel.append("TinyGPU - Remote PCI Device Server\n\n")
            logModel.append("Ready! Server starts automatically when tinygrad runs.\n\n")
          } else {
            logModel.append("TinyGPU - Remote PCI Device Server\n\n")
            logModel.append("To install: Move TinyGPU.app to /Applications/\n\n")
          }

          logModel.append("Troubleshooting:\n")
          logModel.append("  • Check GPU is connected:\n")
          logModel.append("    Apple menu > About This Mac > System Report > PCI\n")
          logModel.append("  • Approve extension in System Settings > Privacy & Security\n")
          logModel.append("  • Check status: TinyGPU status (in terminal)\n\n")
          logModel.append("[Press Cmd+Q to quit]\n")
        }
    }
  }
}

class AppDelegate: NSObject, NSApplicationDelegate {
  func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
    return true
  }
}
