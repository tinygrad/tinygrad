use crate::helpers::{bits, sign_ext};

const NULL: u8 = 124;

#[derive(Debug, PartialEq)]
pub enum Instruction {
    SMEM { op: u8, sdata: u8, sbase: u8, offset: i32, soffset: u8, glc: bool, dlc: bool },
}

pub fn decode(word0:u32, word1:Option<&u32>) -> Instruction {
    match bits(word0, 31, 30) {
        0b11 => {
            let word = (*word1.unwrap() as u64) << 32 | (word0 as u64);
            match bits(word, 29, 26) {
                0b1101 => {
                    let sbase = (bits(word, 5, 0) as u8) << 1;
                    let sdata = bits(word, 12, 6) as u8;
                    let dlc = bits(word, 13, 13) != 0;
                    let glc = bits(word, 14, 14) != 0;
                    let op = bits(word, 25, 18) as u8;
                    let offset = sign_ext(bits(word, 52, 32), 21) as i32;
                    let soffset = bits(word, 63, 57) as u8;
                    Instruction::SMEM { sbase, sdata, dlc, glc, op, offset, soffset }
                }
                _ => todo!(),
            }
        }
        _ => todo!(),
    }
}

#[cfg(test)]
mod test_rdna3 {
    use super::*;

    use std::process::{Stdio, Command};
    use std::io::{Result, Write};

    const LLVM_ARGS: &[&str; 3] = &["--arch=amdgcn", "--mcpu=gfx1100", "--triple=amdgcn-amd-amdhsa"];
    const OFFSET_PRG: usize = 16;
    fn llvm_assemble(asm: &str) -> Result<Vec<u8>> {
        let mut proc = Command::new("llvm-mc").args(LLVM_ARGS).args(["-filetype=obj", "-o", "-"]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
        proc.stdin.as_mut().unwrap().write_all(asm.as_bytes())?;
        let out = proc.wait_with_output()?;
        match out.status.success() {
            true => Ok(out.stdout),
            false => Err(std::io::Error::new(std::io::ErrorKind::Other, "llvm-mc err")),
        }
    }

    fn llvm_disassemble(code: &Vec<u8>) -> Result<String> {
        let mut proc = Command::new("llvm-objdump").args(LLVM_ARGS).args(["--disassemble", "-"]).stdin(Stdio::piped()).stdout(Stdio::piped()).spawn()?;
        proc.stdin.as_mut().unwrap().write_all(code)?;
        let out = proc.wait_with_output()?;
        match out.status.success() {
            true => Ok(String::from_utf8(out.stdout).unwrap()),
            false => Err(std::io::Error::new(std::io::ErrorKind::Other, "llvm-objdump err")),
        }
    }

    fn test_decode(asm: &str) -> Instruction {
        let lib = llvm_assemble(asm).unwrap();
        println!("{}", llvm_disassemble(&lib).unwrap());
        let stream: Vec<u32> = lib.chunks_exact(4).map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap())).skip(OFFSET_PRG).collect();
        decode(stream[0], stream.get(1))
    }

    #[test]
    fn test_asm_smem() {
        assert_eq!(test_decode("s_load_b128 s[4:7], s[0:1], null"), Instruction::SMEM { op: 2, sdata: 4, sbase: 0, offset: 0, soffset: NULL, glc: false, dlc: false });
        assert_eq!(test_decode("s_load_b32 s10, s[0:1], 0xc"), Instruction::SMEM { op: 0, sdata: 10, sbase: 0, offset: 0xc, soffset: NULL, glc: false, dlc: false });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], s6"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: 0, soffset: 6, glc: false, dlc: false });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], glc dlc"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: 0, soffset: NULL, glc: true, dlc: true });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], glc"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: 0, soffset: NULL, glc: true, dlc: false });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], -20"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: -20, soffset: NULL, glc: false, dlc: false });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], -1048576"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: -1048576, soffset: NULL, glc: false, dlc: false });
    }
}
