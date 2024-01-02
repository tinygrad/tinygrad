{ lib
, buildPythonPackage
, setuptools
, wheel
, numpy
, tqdm
, ...
}@args:

buildPythonPackage {
  pname = "tinygrad";
  version = args.version;
  pyproject = true;

  src = ./.;

  propagatedBuildInputs = [
    numpy
    tqdm
  ];

  nativeBuildInputs = [
    setuptools
    wheel
  ];

  pythonImportsCheck = [ "tinygrad" ];

  meta = with lib; {
    description = "You like pytorch? You like micrograd? You love tinygrad";
    homepage = "https://github.com/tinygrad/tinygrad";
    license = licenses.mit;
  };
}
