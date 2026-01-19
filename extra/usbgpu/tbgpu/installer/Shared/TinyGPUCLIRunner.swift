import Foundation
import SystemExtensions
import IOKit

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

  static func getPCIDevices() -> String {
    guard let matching = IOServiceMatching("IOPCIDevice") else { return "Connected PCI Devices: (read error)\n\n" }
    var iterator: io_iterator_t = 0
    guard IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iterator) == KERN_SUCCESS else { return "Connected PCI Devices:\n\n" }
    defer { IOObjectRelease(iterator) }

    var devices: [String] = []
    while case let service = IOIteratorNext(iterator), service != 0 {
      defer { IOObjectRelease(service) }

      var vendorID: UInt16 = 0, deviceID: UInt16 = 0, isGPU = false
      if let data = IORegistryEntryCreateCFProperty(service, "vendor-id" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() as? Data, data.count >= 2 {
        vendorID = data.withUnsafeBytes { $0.load(as: UInt16.self) }
      }
      if let data = IORegistryEntryCreateCFProperty(service, "device-id" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() as? Data, data.count >= 2 {
        deviceID = data.withUnsafeBytes { $0.load(as: UInt16.self) }
      }
      if let data = IORegistryEntryCreateCFProperty(service, "class-code" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() as? Data, data.count >= 3 {
        isGPU = data[2] == 0x03
      }

      let name = String(format: "%04x:%04x", vendorID, deviceID)
      devices.append(isGPU ? "\(name) (supported)" : name)
    }

    return devices.isEmpty ? "PCI Devices: none\n\n" : "PCI Devices:\n" + devices.map { "  • \($0)\n" }.joined() + "\n"
  }

  func install(completion: @escaping (TinyGPUCLIExit) -> Void) {
    self.done = completion
    installExtension()
  }

  private func printUsage() {
    log("""
    Usage: TinyGPU <command> [options]

    Commands:
      status        Show current extension status
      install       Install/activate the driver extension (requires approval)
      uninstall     Deactivate and remove the driver extension
      server <path> Start server on Unix socket at <path>
    """)
  }

  func run(args: [String], done: @escaping (TinyGPUCLIExit) -> Void) {
    guard args.count > 1 else {
      printUsage()
      done(.usage)
      return
    }

    let cmd = args[1]

    switch cmd {
    case "status":
      printStatus(Self.queryDextState(bundleID: dextIdentifier))
      done(.ok)

    case "install":
      self.done = done
      installExtension()

    case "uninstall":
      self.done = done
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

  static func getApprovalInstructions() -> String {
    """
    Please go to System Settings > Privacy & Security and allow the extension.

    If the extension was previously disabled, you need to:
      1. Open System Settings > General > Login Items & Extensions
      2. Find 'TinyGPU' under 'Driver Extensions'
      3. Toggle it ON

    """
  }

  static func getStatusText(_ state: DextState) -> String {
    switch state {
    case .unloaded:
      return "Driver extension not installed.\n\n"
    case .activating:
      return "Extension is activating...\n\n"
    case .needsApproval:
      return "Extension is awaiting approval.\n\n" + getApprovalInstructions()
    case .activated:
      return "Extension is ready! Run tinygrad to use your eGPU.\n\n"
    case .activationError:
      return "Extension activation failed.\nCheck system logs for details.\n\n"
    }
  }

  private func printStatus(_ state: DextState) {
    log(Self.getStatusText(state))
    log(Self.getPCIDevices())
  }

  private func installExtension() {
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
    log("\nUser approval required!\n\n")
    log(Self.getApprovalInstructions())
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
        log("\nThe extension is disabled by the user.\n\n")
        log(Self.getApprovalInstructions())
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
