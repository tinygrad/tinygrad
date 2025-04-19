use crate::helpers::{Colorize, DEBUG};
use crate::state::{Register, VecDataStore, WaveValue, VGPR};
use crate::thread::{Thread, END_PRG};
use std::collections::HashMap;

pub struct WorkGroup<'a> {
    dispatch_dim: u32,
    id: [u32; 3],
    lds: VecDataStore,
    kernel: &'a Vec<u32>,
    kernel_args: *const u64,
    launch_bounds: [u32; 3],
    wave_state: HashMap<usize, WaveState>,
}

struct WaveState {
    scalar_reg: Vec<u32>,
    scc: u32,
    vec_reg: VGPR,
    vcc: WaveValue,
    exec: WaveValue,
    pc: usize,
    sds: HashMap<usize, VecDataStore>,
}

const SYNCS: [u32; 5] = [0xBF89FC07, 0xBFBD0000, 0xBC7C0000, 0xBF890007, 0xbFB60003];
const BARRIERS: [[u32; 2]; 5] = [
    [SYNCS[0], SYNCS[0]],
    [SYNCS[0], SYNCS[1]],
    [SYNCS[0], SYNCS[2]],
    [SYNCS[3], SYNCS[1]],
    [SYNCS[1], SYNCS[0]],
];
impl<'a> WorkGroup<'a> {
    pub fn new(dispatch_dim: u32, id: [u32; 3], launch_bounds: [u32; 3], kernel: &'a Vec<u32>, kernel_args: *const u64) -> Self {
        return Self {
            dispatch_dim,
            id,
            kernel,
            launch_bounds,
            kernel_args,
            lds: VecDataStore::new(),
            wave_state: HashMap::new(),
        };
    }

    pub fn exec_waves(&mut self) -> Result<(), i32> {
        let mut blocks = vec![];
        for z in 0..self.launch_bounds[2] {
            for y in 0..self.launch_bounds[1] {
                for x in 0..self.launch_bounds[0] {
                    blocks.push([x, y, z])
                }
            }
        }
        let waves = blocks.chunks(32).map(|w| w.to_vec()).collect::<Vec<_>>();

        let mut sync = false;
        for (i, x) in self.kernel.iter().enumerate() {
            if i != 0 && BARRIERS.contains(&[*x, self.kernel[i - 1]]) {
                sync = true;
                break;
            }
        }
        for _ in 0..=(sync as usize) {
            for w in waves.iter().enumerate() {
                self.exec_wave(w)?
            }
        }
        Ok(())
    }

    fn exec_wave(&mut self, (wave_id, threads): (usize, &Vec<[u32; 3]>)) -> Result<(), i32> {
        let wave_state = self.wave_state.get(&wave_id);
        let mut sds = match wave_state {
            Some(val) => val.sds.clone(),
            None => {
                let mut sds = HashMap::new();
                for i in 0..=31 {
                    sds.insert(i, VecDataStore::new());
                }
                sds
            }
        };
        let (mut scalar_reg, mut scc, mut pc) = match wave_state {
            Some(val) => (val.scalar_reg.to_vec(), val.scc, val.pc),
            None => {
                let mut scalar_reg = vec![0; 256];
                scalar_reg.write64(0, self.kernel_args as u64);
                let [gx, gy, gz] = self.id;
                match self.dispatch_dim {
                    3 => (scalar_reg[13], scalar_reg[14], scalar_reg[15]) = (gx, gy, gz),
                    2 => (scalar_reg[14], scalar_reg[15]) = (gx, gy),
                    _ => scalar_reg[15] = gx,
                }
                (scalar_reg, 0, 0)
            }
        };
        let (mut vec_reg, mut vcc) = match wave_state {
            Some(val) => (val.vec_reg.clone(), val.vcc.clone()),
            None => (VGPR::new(), WaveValue::new(0, threads.len())),
        };
        let mut exec = match wave_state {
            Some(val) => val.exec.clone(),
            None => {
                let active = match threads.len() == 32 {
                    true => u32::MAX,
                    false => (1 << threads.len()) - 1,
                };
                WaveValue::new(active, threads.len())
            }
        };

        let mut seeded_lanes = vec![];
        loop {
            if self.kernel[pc] == END_PRG {
                break Ok(());
            }
            if BARRIERS.contains(&[self.kernel[pc], self.kernel[pc + 1]]) && wave_state.is_none() {
                self.wave_state.insert(
                    wave_id,
                    WaveState {
                        scalar_reg,
                        scc,
                        vec_reg,
                        vcc,
                        exec,
                        pc,
                        sds,
                    },
                );
                break Ok(());
            }
            if SYNCS.contains(&self.kernel[pc]) || self.kernel[pc] >> 20 == 0xbf8 || self.kernel[pc] == 0x7E000000 {
                pc += 1;
                continue;
            }

            let mut sgpr_co = None;
            for (lane_id, [x, y, z]) in threads.iter().enumerate() {
                vec_reg.default_lane = Some(lane_id);
                vcc.default_lane = Some(lane_id);
                exec.default_lane = Some(lane_id);
                if *DEBUG {
                    let lane = format!("{:<2} {:08X} ", lane_id, self.kernel[pc]);
                    let state = match exec.read() {
                        true => "green",
                        false => "gray",
                    };
                    let [id0, id1, id2] = self.id;
                    print!("[{id0:<3} {id1:<3} {id2:<3}] [{x:<3} {y:<3} {z:<3}] {}", lane.color(state));
                }
                if !seeded_lanes.contains(&lane_id) && self.wave_state.get(&wave_id).is_none() {
                    match (self.launch_bounds[1] != 1, self.launch_bounds[2] != 1) {
                        (false, false) => vec_reg[0] = *x,
                        _ => vec_reg[0] = (z << 20) | (y << 10) | x,
                    }
                    seeded_lanes.push(lane_id);
                }
                let mut thread = Thread {
                    scalar_reg: &mut scalar_reg,
                    scc: &mut scc,
                    vec_reg: &mut vec_reg,
                    vcc: &mut vcc,
                    exec: &mut exec,
                    lds: &mut self.lds,
                    sds: &mut sds.get_mut(&lane_id).unwrap(),
                    pc_offset: 0,
                    stream: self.kernel[pc..self.kernel.len()].to_vec(),
                    scalar: false,
                    simm: None,
                    warp_size: threads.len(),
                    sgpr_co: &mut sgpr_co,
                };
                thread.interpret()?;
                if *DEBUG {
                    println!();
                }
                if thread.scalar {
                    pc = ((pc as isize) + 1 + (thread.pc_offset as isize)) as usize;
                    break;
                }
                if lane_id == threads.len() - 1 {
                    pc = ((pc as isize) + 1 + (thread.pc_offset as isize)) as usize;
                }
            }

            if vcc.mutations.is_some() {
                vcc.apply_muts();
                vcc.mutations = None;
            }
            if exec.mutations.is_some() {
                exec.apply_muts();
                exec.mutations = None;
            }
            if let Some((idx, mut wv)) = sgpr_co.take() {
                wv.apply_muts();
                scalar_reg[idx] = wv.value;
            }
        }
    }
}

#[cfg(test)]
mod test_workgroup {
    use super::*;

    // TODO: make this generic by adding the assembler
    fn global_store_sgpr(addr: u64, instructions: Vec<u32>, src: u32) -> Vec<u32> {
        [
            instructions,
            vec![
                0x7E020200 + src,
                0x7E0402FF,
                addr as u32,
                0x7E0602FF,
                (addr >> 32) as u32,
                0xDC6A0000,
                0x007C0102,
            ],
            vec![END_PRG],
        ]
        .concat()
    }

    #[test]
    fn test_wave_value_state_vcc() {
        let mut ret: u32 = 0;
        let kernel = vec![
            0xBEEA00FF,
            0b11111111111111111111111111111111, // initial vcc state
            0x7E140282,
            0x7C94010A, // cmp blockDim.x == 2
        ];
        let addr = (&mut ret as *mut u32) as u64;
        let kernel = global_store_sgpr(addr, kernel, 106);
        let mut wg = WorkGroup::new(1, [0, 0, 0], [3, 1, 1], &kernel, [addr].as_ptr());
        wg.exec_waves().unwrap();
        assert_eq!(ret, 0b100);
    }

    #[test]
    fn test_wave_value_state_exec() {
        let mut ret: u32 = 0;
        let kernel = vec![
            0xBEFE00FF,
            0b11111111111111111111111111111111,
            0x7E140282,
            0x7D9C010A, // cmpx blockDim.x <= 2
        ];
        let addr = (&mut ret as *mut u32) as u64;
        let kernel = global_store_sgpr(addr, kernel, 126);
        let mut wg = WorkGroup::new(1, [0, 0, 0], [4, 1, 1], &kernel, [addr].as_ptr());
        wg.exec_waves().unwrap();
        assert_eq!(ret, 0b0111);
    }

    #[test]
    fn test_wave_value_sgpr_co() {
        let mut ret: u32 = 0;
        let kernel = vec![0xBE8D00FF, 0x7FFFFFFF, 0x7E1402FF, u32::MAX, 0xD700000A, 0x0002010A];
        let addr = (&mut ret as *mut u32) as u64;
        let kernel = global_store_sgpr(addr, kernel, 0);
        let mut wg = WorkGroup::new(1, [0, 0, 0], [5, 1, 1], &kernel, [addr].as_ptr());
        wg.exec_waves().unwrap();
        assert_eq!(ret, 0b11110);
    }
}
