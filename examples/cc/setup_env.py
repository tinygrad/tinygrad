from pathlib import Path

from tinygrad.helpers import fetch

def main():
  # Create third_party dir if not exists
  dirname = Path(__file__).parent
  thirdparty_dirname = dirname / "third_party"
  if not thirdparty_dirname.exists():
    thirdparty_dirname.mkdir(parents=True)

  stb_image_file = thirdparty_dirname / "stb/stb_image.h"
  
  if not stb_image_file.exists():
    # Download stb_image lib
    fetch("https://raw.githubusercontent.com/nothings/stb/master/stb_image.h", stb_image_file)

  nlohmann_json_file = thirdparty_dirname / "nlohmann/json.hpp"

  if not nlohmann_json_file.exists():
    # Download nlohmann::json lib
    fetch("https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp", nlohmann_json_file)


if __name__ == "__main__":
  main()
