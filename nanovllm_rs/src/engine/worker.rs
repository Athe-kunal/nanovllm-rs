use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::JoinHandle;

use candle_core::Device;
use cudarc::nccl::sys::ncclUniqueId;

use crate::config::{Config, EngineConfig};
use crate::engine::model_runner::ModelRunner;
use crate::engine::sequence::Sequence;
use crate::layers::nccl::Comm;

enum Command {
    FinishSetup(usize),
    Run(Vec<Sequence>),
    Exit,
}

pub struct WorkerHandle {
    tx: Sender<Command>,
    thread: Option<JoinHandle<()>>,
}

impl WorkerHandle {
    pub fn run(&self, seqs: Vec<Sequence>) {
        self.tx.send(Command::Run(seqs)).expect("worker thread panicked");
    }

    fn finish_setup(&self, num_blocks: usize) {
        self.tx.send(Command::FinishSetup(num_blocks)).expect("worker thread panicked");
    }
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(Command::Exit);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn spawn_worker(
    config: Config,
    engine_config: EngineConfig,
    rank: usize,
    world_size: usize,
    id: ncclUniqueId,
    probe_tx: Sender<usize>,
) -> WorkerHandle {
    let (tx, rx) = mpsc::channel::<Command>();

    let thread = std::thread::spawn(move || {
        // Binds this thread's CUDA context to `rank`'s device before NCCL captures it via
        // cudaGetDevice() — otherwise every fresh thread defaults to device 0 and NCCL sees
        // every rank claiming the same GPU ("Duplicate GPU detected").
        Device::cuda_if_available(rank).expect("failed to create device");
        let comm = Comm::init_rank(rank, world_size, id).expect("nccl comm_init_rank failed");
        let mut runner = ModelRunner::new(&config, &engine_config, rank, Some(Arc::new(comm)));
        probe_tx.send(runner.probe_num_kvcache_blocks()).expect("coordinator dropped");

        while let Ok(cmd) = rx.recv() {
            match cmd {
                Command::FinishSetup(n) => runner.finish_setup(n),
                Command::Run(mut seqs) => {
                    runner.run(&mut seqs);
                }
                Command::Exit => break,
            }
        }
        runner.exit();
    });

    WorkerHandle { tx, thread: Some(thread) }
}

/// Brings up tensor parallelism for `engine_config.tensor_parallel_size > 1`: spawns
/// one thread per rank `1..world_size` (rank 0 runs on the caller's thread since
/// `LLMEngine` drives it directly), joins the NCCL communicator on every rank, then
/// reconciles each rank's independently-probed KV-cache budget down to the minimum
/// before any rank commits to it.
pub fn init_tensor_parallel(config: &Config, engine_config: &EngineConfig) -> (ModelRunner, Vec<WorkerHandle>) {
    let world_size = engine_config.tensor_parallel_size;
    let id = Comm::new_id().expect("nccl get_uniqueid failed");

    let (probe_tx, probe_rx): (Sender<usize>, Receiver<usize>) = mpsc::channel();
    let handles: Vec<WorkerHandle> = (1..world_size)
        .map(|rank| spawn_worker(config.clone(), engine_config.clone(), rank, world_size, id, probe_tx.clone()))
        .collect();
    drop(probe_tx);

    Device::cuda_if_available(0).expect("failed to create device");
    let comm0 = Comm::init_rank(0, world_size, id).expect("nccl comm_init_rank failed");
    let mut model_runner = ModelRunner::new(config, engine_config, 0, Some(Arc::new(comm0)));

    let mut min_blocks = model_runner.probe_num_kvcache_blocks();
    for _ in 1..world_size {
        min_blocks = min_blocks.min(probe_rx.recv().expect("worker died before reporting its kv cache probe"));
    }

    model_runner.finish_setup(min_blocks);
    for h in &handles {
        h.finish_setup(min_blocks);
    }

    (model_runner, handles)
}
