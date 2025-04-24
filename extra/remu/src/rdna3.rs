use crate::helpers::{bits, sign_ext};

const NULL: u8 = 124;

#[allow(dead_code)]
#[derive(Debug, PartialEq)]
pub enum Instruction {
    SOP2 { op: u8, ssrc0: u8, ssrc1: u8, sdst: u8 },
    SOP1 { op: u8, ssrc0: u8, sdst: u8 },
    SOPK { op: u8, simm16: i16, sdst: u8 },
    SOPP { op: u8, simm16: i16 },
    SOPC { op: u8, ssrc0: u8, ssrc1: u8 },

    SMEM { op: u8, sdata: u8, sbase: u8, offset: i32, soffset: u8, glc: bool, dlc: bool },

    VOP1 { op: u8, },
    VOP2 { op: u8, },
    VOPC { op: u8, },
    VOP3 { op: u8, },
    VOP3P { op: u8, },

    DS { op: u8, },

    FLAT { op: u8, },
}

pub fn decode(word:u32, word1:Option<&u32>) -> Instruction {
    match bits(word, 31, 30) {
        0b11 => {
            let word = (*word1.unwrap() as u64) << 32 | (word as u64);
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
        0b10 => {
            if bits(word, 29, 23) == 0b1111101 {
                let ssrc0 = bits(word, 7, 0) as u8;
                let op = bits(word, 15, 8) as u8;
                let sdst = bits(word, 22, 16) as u8;
                return Instruction::SOP1 { ssrc0, sdst, op }
            }
            if bits(word, 29, 23) == 0b1111111 {
                let simm16 = bits(word, 15, 0) as i16;
                let op = bits(word, 22, 16) as u8;
                return Instruction::SOPP { simm16, op }
            }
            if bits(word, 29, 23) == 0b1111110 {
                let ssrc0 = bits(word, 7, 0) as u8;
                let ssrc1 = bits(word, 15, 8) as u8;
                let op = bits(word, 22, 16) as u8;
                return Instruction::SOPC { ssrc0, ssrc1, op }
            }
            if bits(word, 29, 28) == 0b11 {
                let simm16 = bits(word, 15, 0) as i16;
                let sdst = bits(word, 22, 16) as u8;
                let op = bits(word, 27, 23) as u8;
                return Instruction::SOPK { simm16, sdst, op }
            }
            let ssrc0 = bits(word, 7, 0) as u8;
            let ssrc1 = bits(word, 15, 8) as u8;
            let sdst = bits(word, 22, 16) as u8;
            let op = bits(word, 29, 23) as u8;
            return Instruction::SOP2 { ssrc0, ssrc1, sdst, op }
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
    fn test_decode_smem() {
        assert_eq!(test_decode("s_load_b128 s[4:7], s[0:1], null"), Instruction::SMEM { op: 2, sdata: 4, sbase: 0, offset: 0, soffset: NULL, glc: false, dlc: false });
        assert_eq!(test_decode("s_load_b32 s10, s[0:1], 0xc"), Instruction::SMEM { op: 0, sdata: 10, sbase: 0, offset: 0xc, soffset: NULL, glc: false, dlc: false });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], s6"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: 0, soffset: 6, glc: false, dlc: false });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], glc dlc"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: 0, soffset: NULL, glc: true, dlc: true });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], glc"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: 0, soffset: NULL, glc: true, dlc: false });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], -20"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: -20, soffset: NULL, glc: false, dlc: false });
        assert_eq!(test_decode("s_load_b32 s0, s[4:5], -1048576"), Instruction::SMEM { op: 0, sdata: 0, sbase: 4, offset: -1048576, soffset: NULL, glc: false, dlc: false });
    }

    #[test]
    fn test_decode_salu() {
        assert_eq!(test_decode("s_add_u32 s1 s2 s3"), Instruction::SOP2 { op: 0, ssrc0: 2, ssrc1: 3, sdst: 1 });
        assert_eq!(test_decode("s_add_u32 vcc_hi exec_lo vcc_lo"), Instruction::SOP2 { op: 0, ssrc0: 126, ssrc1: 106, sdst: 107 });
        assert_eq!(test_decode("s_mov_b32 s1 -0.5"), Instruction::SOP1 { op: 0, ssrc0: 241, sdst: 1 });
        assert_eq!(test_decode("s_cmpk_eq_i32 s0 -30"), Instruction::SOPK { op: 3, sdst: 0, simm16: -30 });
        assert_eq!(test_decode("s_cmpk_eq_u32 s0 65535"), Instruction::SOPK { op: 9, sdst: 0, simm16: -1 });
        assert_eq!(test_decode("s_cmp_ge_i32 s1 s2"), Instruction::SOPC { op: 3, ssrc0: 1, ssrc1: 2 });
    }
}
