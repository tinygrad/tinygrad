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
  private var logHandler: ((String) -> Void)?

  init(dextIdentifier: String, logHandler: ((String) -> Void)? = nil) {
    self.dextIdentifier = dextIdentifier
    self.logHandler = logHandler
  }

  private func log(_ msg: String) {
    if let logHandler = logHandler { logHandler(msg) }
    else { print(msg, terminator: "") }
  }

  static func queryDextState(bundleID: String) -> DextState {
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
    log("""
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
        log("Error: server command requires socket path\n")
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
      log("Unknown command: \(cmd)\n")
      printUsage()
      done(.usage)
    }
  }

  private func printStatus(_ state: DextState) {
    switch state {
    case .unloaded:
      log("Status: Not installed\n")
      log("Run 'TinyGPU install' to install the driver extension.\n")
    case .activating:
      log("Status: Activating...\n")
    case .needsApproval:
      log("Status: Waiting for user approval\n")
      log("Please approve the extension in System Settings > Privacy & Security.\n")
    case .activated:
      log("Status: Installed and active\n")
    case .activationError:
      log("Status: Activation failed\n")
      log("Check system logs for details.\n")
    }
  }

  private func installExtension() {
    let state = Self.queryDextState(bundleID: dextIdentifier)

    if state == .activated {
      log("Driver extension is already installed and active.\n")
      done?(.ok)
      return
    }

    log("Installing TinyGPU driver extension...\n")
    log("You may be prompted to approve the extension in System Settings.\n\n")

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
      log("Driver extension is not installed.\n")
      done?(.ok)
      return
    }

    log("Uninstalling TinyGPU driver extension...\n")

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
    log("\nUser approval required!\n")
    log("Please go to System Settings > Privacy & Security and allow the extension.\n\n")
    log("If the extension was previously disabled, you need to:\n")
    log("  1. Open System Settings > General > Login Items & Extensions\n")
    log("  2. Find 'TinyGPU' under 'Driver Extensions'\n")
    log("  3. Toggle it ON\n\n")
    log("After approval, connect the gpu and use it with tinygrad.\n")
    done?(.needsApproval)
  }

  func request(_ request: OSSystemExtensionRequest,
               didFinishWithResult result: OSSystemExtensionRequest.Result) {
    switch result {
    case .completed:
      if pendingAction == "install" {
        log("Driver extension installed successfully!\n")
      } else if pendingAction == "uninstall" {
        log("Driver extension uninstalled successfully!\n")
      }
      done?(.ok)
    case .willCompleteAfterReboot:
      log("Extension will be activated after system reboot.\n")
      done?(.ok)
    @unknown default:
      log("Completed with result: \(result)\n")
      done?(.ok)
    }
  }

  func request(_ request: OSSystemExtensionRequest,
               didFailWithError error: Error) {
    log("\nError: \(error.localizedDescription)\n")

    // Provide helpful guidance based on common errors
    let nsError = error as NSError
    if nsError.domain == OSSystemExtensionErrorDomain {
      switch nsError.code {
      case 1: // OSSystemExtensionErrorUnknown
        log("Unknown error occurred.\n")
      case 4: // OSSystemExtensionErrorMissingEntitlement
        log("The app is missing required entitlements. Please rebuild with proper signing.\n")
      case 8: // OSSystemExtensionErrorExtensionNotFound
        log("Extension not found in app bundle.\n")
      case 9: // OSSystemExtensionErrorExtensionRequired (user disabled)
        log("\nThe extension is disabled by the user.\n")
        log("To enable it:\n")
        log("  1. Open System Settings > General > Login Items & Extensions\n")
        log("  2. Find 'TinyGPU' under 'Driver Extensions'\n")
        log("  3. Toggle it ON\n")
      default:
        log("Error code: \(nsError.code)\n")
      }
    }
    done?(.failed)
  }

  func request(_ request: OSSystemExtensionRequest,
               actionForReplacingExtension existing: OSSystemExtensionProperties,
               withExtension ext: OSSystemExtensionProperties)
  -> OSSystemExtensionRequest.ReplacementAction {
    log("Updating existing extension (v\(existing.bundleShortVersion) -> v\(ext.bundleShortVersion))...\n")
    return .replace
  }
}
