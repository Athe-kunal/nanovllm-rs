use candle_core::{safetensors, Device, Result, Tensor};
use std::collections::HashMap;

pub type WeightLoader = fn(&mut Tensor, Tensor, Option<usize>) -> Result<()>;

pub fn default_weight_loader(param: &mut Tensor, loaded_weight: Tensor, _shard_id: Option<usize>) -> Result<()> {
    *param = loaded_weight;
    Ok(())
}

pub trait ModelWeights {
    fn packed_modules_mapping(&self) -> HashMap<String, (String, usize)> {
        HashMap::new()
    }

    fn get_parameter_mut(&mut self, name: &str) -> Option<&mut Tensor>;

    fn weight_loader(&self, _param_name: &str) -> WeightLoader {
        default_weight_loader
    }
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
                    let loader = model.weight_loader(&param_name);
                    if let Some(param) = model.get_parameter_mut(&param_name) {
                        loader(param, loaded_weight.clone(), Some(*shard_id))?;
                    }
                    matched = true;
                    break;
                }
            }

            if !matched {
                let loader = model.weight_loader(&weight_name);
                if let Some(param) = model.get_parameter_mut(&weight_name) {
                    loader(param, loaded_weight, None)?;
                }
            }
        }
    }

    Ok(())
}
