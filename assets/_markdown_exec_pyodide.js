var _sessions = {};

function getSession(name, pyodide) {
    if (!(name in _sessions)) {
        _sessions[name] = pyodide.globals.get("dict")();
    }
    return _sessions[name];
}

function writeOutput(element, string) {
    element.innerHTML += string + '\n';
}

function clearOutput(element) {
    element.innerHTML = '';
}

async function evaluatePython(pyodide, editor, output, session) {
    pyodide.setStdout({ batched: (string) => { writeOutput(output, new Option(string).innerHTML); } });
    let result, code = editor.getValue();
    clearOutput(output);
    try {
        result = await pyodide.runPythonAsync(code, { globals: getSession(session, pyodide) });
    } catch (error) {
        writeOutput(output, new Option(error.toString()).innerHTML);
    }
    if (result) writeOutput(output, new Option(result).innerHTML);
    hljs.highlightElement(output);
}

async function initPyodide() {
    try {
        let pyodide = await loadPyodide();
        await pyodide.loadPackage("micropip");
        return pyodide;
    } catch(error) {
        return null;
    }
}

function getTheme() {
    return document.body.getAttribute('data-md-color-scheme');
}

function setTheme(editor, currentTheme, light, dark) {
    // https://gist.github.com/RyanNutt/cb8d60997d97905f0b2aea6c3b5c8ee0
    if (currentTheme === "default") {
        editor.setTheme("ace/theme/" + light);
        document.querySelector(`link[title="light"]`).removeAttribute("disabled");
        document.querySelector(`link[title="dark"]`).setAttribute("disabled", "disabled");
    } else if (currentTheme === "slate") {
        editor.setTheme("ace/theme/" + dark);
        document.querySelector(`link[title="dark"]`).removeAttribute("disabled");
        document.querySelector(`link[title="light"]`).setAttribute("disabled", "disabled");
    }
}

function updateTheme(editor, light, dark) {
    // Create a new MutationObserver instance
    const observer = new MutationObserver((mutations) => {
        // Loop through the mutations that occurred
        mutations.forEach((mutation) => {
            // Check if the mutation was a change to the data-md-color-scheme attribute
            if (mutation.attributeName === 'data-md-color-scheme') {
                // Get the new value of the attribute
                const newColorScheme = mutation.target.getAttribute('data-md-color-scheme');
                // Update the editor theme
                setTheme(editor, newColorScheme, light, dark);
            }
        });
    });

    // Configure the observer to watch for changes to the data-md-color-scheme attribute
    observer.observe(document.body, {
        attributes: true,
        attributeFilter: ['data-md-color-scheme'],
    });
}

async function setupPyodide(
    idPrefix,
    install = null,
    themeLight = 'tomorrow',
    themeDark = 'tomorrow_night',
    session = null,
    minLines = 5,
    maxLines = 30,
) {
    const editor = ace.edit(idPrefix + "editor");
    const run = document.getElementById(idPrefix + "run");
    const clear = document.getElementById(idPrefix + "clear");
    const output = document.getElementById(idPrefix + "output");

    updateTheme(editor, themeLight, themeDark);

    editor.session.setMode("ace/mode/python");
    setTheme(editor, getTheme(), themeLight, themeDark);

    editor.setOption("minLines", minLines);
    editor.setOption("maxLines", maxLines);

    // Force editor to resize after setting options
    editor.resize();

    writeOutput(output, "Initializing...");
    let pyodide = await pyodidePromise;
    if (install && install.length) {
        try {
            micropip = pyodide.pyimport("micropip");
            for (const package of install)
                await micropip.install(package);
            clearOutput(output);
        } catch (error) {
            clearOutput(output);
            writeOutput(output, `Could not install one or more packages: ${install.join(", ")}\n`);
            writeOutput(output, new Option(error.toString()).innerHTML);
        }
    } else {
        clearOutput(output);
    }
    run.onclick = () => evaluatePython(pyodide, editor, output, session);
    clear.onclick = () => clearOutput(output);
    output.parentElement.parentElement.addEventListener("keydown", (event) => {
        if (event.ctrlKey && event.key.toLowerCase() === 'enter') {
            event.preventDefault();
            run.click();
        }
    });
}

var pyodidePromise = initPyodide();
