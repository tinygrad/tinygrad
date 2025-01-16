import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v0.27.1/full/pyodide.mjs";

let initPyodide = async () => {
  console.log("initializing")
  const pyodide = await loadPyodide({
    env: {
      "WEBGPU": "1"
    }
  });
  await pyodide.loadPackage("numpy");
  await pyodide.loadPackage("sqlite3");
  await pyodide.loadPackage(`${self.location.origin}/tinygrad/tinygrad-0.10.0-py3-none-any.whl`);
  console.log("Pyodide initialized")
  return pyodide
}

let pyodideReadyPromise = initPyodide()

self.onmessage = async (event) => {
  // make sure loading is done
  const pyodide = await pyodideReadyPromise;
  pyodide.setStdout({
    batched: (output) => {
      self.postMessage(output)
    }
  })
  pyodide.setStderr({
    batched: (output) => {
      self.postMessage(output)
    }
  })
  const { python } = event.data;
  try {
    await pyodide.runPythonAsync(python);
  } catch (e) {
    self.postMessage(e.message + e.stack)
  }
};