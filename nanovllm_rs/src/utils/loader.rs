use candle_core::{safetensors, Device, Result, Tensor};
use std::collections::HashMap;

/// A packed sub-weight's shard id, which the checkpoint loader passes to the target
/// layer's `weight_loader`. `QKVParallelLinear` keys shards by name ("q"/"k"/"v"),
/// `MergedColumnParallelLinear` keys shards by index (0/1) — mirrors Python's dict
/// mixing string ids (for qkv_proj) and int ids (for gate_up_proj) in one mapping.
#[derive(Debug, Clone, Copy)]
pub enum ShardId {
    Name(&'static str),
    Index(usize),
}

pub trait ModelWeights {
    fn packed_modules_mapping(&self) -> HashMap<String, (String, ShardId)> {
        HashMap::new()
    }

    /// Routes a single checkpoint tensor to wherever it belongs in the model.
    /// `param_name` has already had any packed-module substitution applied (e.g.
    /// `q_proj` -> `qkv_proj`). Implementations should silently no-op on names they
    /// don't recognize (checkpoints commonly carry extra buffers that aren't real
    /// parameters) rather than erroring the whole load over one unexpected key.
    fn load_weight(&mut self, param_name: &str, loaded_weight: Tensor, shard_id: Option<ShardId>) -> Result<()>;
}

pub fn load_model<M: ModelWeights>(model: &mut M, path: &str) -> Result<()> {
    let packed_modules_mapping = model.packed_modules_mapping();

    let entries = std::fs::read_dir(path).map_err(candle_core::Error::wrap)?;
    for entry in entries {
        let entry = entry.map_err(candle_core::Error::wrap)?;
        let file_path = entry.path();
        if file_path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }

        let tensors: HashMap<String, Tensor> = safetensors::load(&file_path, &Device::Cpu)?;

        for (weight_name, loaded_weight) in tensors {
            let mut matched = false;

            for (k, (v, shard_id)) in packed_modules_mapping.iter() {
                if weight_name.contains(k.as_str()) {
                    let param_name = weight_name.replace(k.as_str(), v.as_str());
                    model.load_weight(&param_name, loaded_weight.clone(), Some(*shard_id))?;
                    matched = true;
                    break;
                }
            }

            if !matched {
                model.load_weight(&weight_name, loaded_weight, None)?;
            }
        }
    }

    Ok(())
}
