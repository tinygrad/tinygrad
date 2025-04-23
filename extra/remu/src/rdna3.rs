#[cfg(test)]
mod test_rdna3 {

    use std::process::{Stdio, Command};
    use std::io::{Result, Write};

    const LLVM_ARGS: &[&str; 3] = &["--arch=amdgcn", "--mcpu=gfx1100", "--triple=amdgcn-amd-amdhsa"];
    fn llvm_assemble(asm: &str) -> Result<Vec<u8>> {
        let mut proc = Command::new("llvm-mc").args(LLVM_ARGS).args(["-filetype=obj", "-o", "-"]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
        proc.stdin.as_mut().unwrap().write_all(asm.as_bytes())?;
        let out = proc.wait_with_output()?;
        match out.status.success() {
            true => Ok(out.stdout),
            false => Err(std::io::Error::new(std::io::ErrorKind::Other, "llvm-mc err")),
        }
    }

    fn llvm_disassemble(code: Vec<u8>) -> Result<String> {
        let mut proc = Command::new("llvm-objdump").args(LLVM_ARGS).args(["--disassemble", "-"]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
        proc.stdin.as_mut().unwrap().write_all(&code)?;
        let out = proc.wait_with_output()?;
        match out.status.success() {
            true => Ok(String::from_utf8(out.stdout).unwrap()),
            false => Err(std::io::Error::new(std::io::ErrorKind::Other, "llvm-objdump err")),
        }
    }

    #[test]
    fn test_smem() {
        let lib = llvm_assemble("s_add_i32 s400, s1, s1").unwrap();
        let asm = llvm_disassemble(lib).unwrap();
    }
}
