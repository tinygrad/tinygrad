import Foundation
import SystemExtensions

enum TinyGPUCLIExit: Int32 {
  case ok = 0
  case usage = 2
  case failed = 3
  case needsApproval = 4
}

enum DextState {
  case unloaded, activating, needsApproval, activated, activationError
}

final class TinyGPUCLIRunner: NSObject, OSSystemExtensionRequestDelegate {
  private let dextIdentifier: String
  private var done: ((TinyGPUCLIExit) -> Void)?
  private var pendingAction: String = ""

  init(dextIdentifier: String) {
    self.dextIdentifier = dextIdentifier
  }

  private static func queryDextState(bundleID: String) -> DextState {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/systemextensionsctl")
    p.arguments = ["list"]
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = Pipe()

    guard (try? p.run()) != nil else { return .unloaded }
    p.waitUntilExit()

    guard let output = String(data: pipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8),
          let line = output.split(separator: "\n").first(where: { $0.contains(bundleID) }) else {
      return .unloaded
    }

    if line.contains("[activated enabled]") { return .activated }
    if line.contains("[activated waiting for user]") { return .needsApproval }
    if line.contains("terminated waiting to uninstall") { return .unloaded }
    return .activating
  }

  private func printUsage() {
    print("""
    TinyGPU Driver Extension Manager

    Usage: TinyGPU <command> [options]

    Commands:
      status        Show current extension status
      install       Install/activate the driver extension (requires approval)
      uninstall     Deactivate and remove the driver extension
      server <path> Start server on Unix socket at <path>
      help          Show this help message

    Examples:
      TinyGPU status                    # Check if driver is installed
      TinyGPU install                   # Install the driver extension
      TinyGPU server /tmp/tinygpu.sock  # Start server
    """)
  }

  func run(args: [String], done: @escaping (TinyGPUCLIExit) -> Void) {
    self.done = done

    // args[0] is executable path
    guard args.count > 1 else {
      printUsage()
      done(.usage)
      return
    }

    let cmd = args[1]

    switch cmd {
    case "status":
      let state = Self.queryDextState(bundleID: dextIdentifier)
      printStatus(state)
      done(.ok)

    case "install":
      installExtension()

    case "uninstall":
      uninstallExtension()

    case "server":
      guard args.count > 2 else {
        fputs("Error: server command requires socket path\n", stderr)
        printUsage()
        done(.usage)
        return
      }
      let sockPath = args[2]
      let exitCode = run_server(sockPath)
      done(exitCode == 0 ? .ok : .failed)

    case "help", "-h", "--help":
      printUsage()
      done(.ok)

    default:
      fputs("Unknown command: \(cmd)\n", stderr)
      printUsage()
      done(.usage)
    }
  }

  private func printStatus(_ state: DextState) {
    switch state {
    case .unloaded:
      print("Status: Not installed")
      print("Run 'TinyGPU install' to install the driver extension.")
    case .activating:
      print("Status: Activating...")
    case .needsApproval:
      print("Status: Waiting for user approval")
      print("Please approve the extension in System Settings > Privacy & Security.")
    case .activated:
      print("Status: Installed and active")
    case .activationError:
      print("Status: Activation failed")
      print("Check system logs for details.")
    }
  }

  private func installExtension() {
    let state = Self.queryDextState(bundleID: dextIdentifier)

    if state == .activated {
      print("Driver extension is already installed and active.")
      done?(.ok)
      return
    }

    print("Installing TinyGPU driver extension...")
    print("You may be prompted to approve the extension in System Settings.")
    print("")

    pendingAction = "install"
    let req = OSSystemExtensionRequest.activationRequest(
      forExtensionWithIdentifier: dextIdentifier,
      queue: .main
    )
    req.delegate = self
    OSSystemExtensionManager.shared.submitRequest(req)
  }

  private func uninstallExtension() {
    let state = Self.queryDextState(bundleID: dextIdentifier)

    if state == .unloaded {
      print("Driver extension is not installed.")
      done?(.ok)
      return
    }

    print("Uninstalling TinyGPU driver extension...")

    pendingAction = "uninstall"
    let req = OSSystemExtensionRequest.deactivationRequest(
      forExtensionWithIdentifier: dextIdentifier,
      queue: .main
    )
    req.delegate = self
    OSSystemExtensionManager.shared.submitRequest(req)
  }

  // MARK: OSSystemExtensionRequestDelegate

  func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
    print("")
    print("User approval required!")
    print("Please go to System Settings > Privacy & Security and allow the extension.")
    print("")
    print("If the extension was previously disabled, you need to:")
    print("  1. Open System Settings > General > Login Items & Extensions")
    print("  2. Find 'TinyGPU' under 'Driver Extensions'")
    print("  3. Toggle it ON")
    print("")
    print("After approval, run 'TinyGPU status' to verify installation.")
    done?(.needsApproval)
  }

  func request(_ request: OSSystemExtensionRequest,
               didFinishWithResult result: OSSystemExtensionRequest.Result) {
    switch result {
    case .completed:
      if pendingAction == "install" {
        print("Driver extension installed successfully!")
      } else if pendingAction == "uninstall" {
        print("Driver extension uninstalled successfully!")
      }
      done?(.ok)
    case .willCompleteAfterReboot:
      print("Extension will be activated after system reboot.")
      done?(.ok)
    @unknown default:
      print("Completed with result: \(result)")
      done?(.ok)
    }
  }

  func request(_ request: OSSystemExtensionRequest,
               didFailWithError error: Error) {
    print("")
    fputs("Error: \(error.localizedDescription)\n", stderr)

    // Provide helpful guidance based on common errors
    let nsError = error as NSError
    if nsError.domain == OSSystemExtensionErrorDomain {
      switch nsError.code {
      case 1: // OSSystemExtensionErrorUnknown
        print("Unknown error occurred.")
      case 4: // OSSystemExtensionErrorMissingEntitlement
        print("The app is missing required entitlements. Please rebuild with proper signing.")
      case 8: // OSSystemExtensionErrorExtensionNotFound
        print("Extension not found in app bundle.")
      case 9: // OSSystemExtensionErrorExtensionRequired (user disabled)
        print("")
        print("The extension is disabled by the user.")
        print("To enable it:")
        print("  1. Open System Settings > General > Login Items & Extensions")
        print("  2. Find 'TinyGPU' under 'Driver Extensions'")
        print("  3. Toggle it ON")
      default:
        print("Error code: \(nsError.code)")
      }
    }
    done?(.failed)
  }

  func request(_ request: OSSystemExtensionRequest,
               actionForReplacingExtension existing: OSSystemExtensionProperties,
               withExtension ext: OSSystemExtensionProperties)
  -> OSSystemExtensionRequest.ReplacementAction {
    print("Updating existing extension (v\(existing.bundleShortVersion) -> v\(ext.bundleShortVersion))...")
    return .replace
  }
}
