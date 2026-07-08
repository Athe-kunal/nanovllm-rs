use candle_core::{Device, Result, Tensor, DType, D};
use crate::layers::nccl::Comm;
use std::sync::Arc;
use crate::config::Config;
use crate::layers::{activation,attention,dist_util,layernorm,linear,embed_head,rotary_embedding};
use attention::Attention;
use layernorm::RMSNorm;
use linear::{MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear};
use activation::SiluAndMul;
use rotary_embedding::RotaryEmbedding;
use embed_head::{ParallelLMHead, VocabParallelEmbedding};
use std::collections::HashMap;
use pyo3::{Py, PyAny};
use crate::utils::loader::{ModelWeights, ShardId};

pub struct Qwen3Attention{
    total_num_heads: usize,
    num_heads: usize,
    total_num_kv_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    q_size: usize,
    kv_size: usize,
    scaling: f32,
    qkv_bias: bool,
    qkv_proj: QKVParallelLinear,
    o_proj: RowParallelLinear,
    rotary_emb: RotaryEmbedding,
    attn: Attention,
    q_norm: Option<RMSNorm>,
    k_norm: Option<RMSNorm>,
}

fn split_qkv(qkv: &Tensor, q_size: usize, kv_size: usize) -> Result<(Tensor, Tensor, Tensor)>{
    let q = qkv.narrow(D::Minus1, 0, q_size)?;
    let k = qkv.narrow(D::Minus1, q_size, kv_size)?;
    let v = qkv.narrow(D::Minus1, q_size + kv_size, kv_size)?;
    Ok((q,k,v))
}

impl Qwen3Attention{
    #[allow(clippy::too_many_arguments)]
    fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        max_position: usize,
        head_dim: Option<usize>,
        rms_norm_eps: f32,
        qkv_bias: bool,
        rope_theta: f32,
        comm: Option<Arc<Comm>>,
        device: &Device,
    ) -> Result<Self> {
        let tp_size = match &comm {
            Some(comm) => comm.world_size(),
            None => 1,
        };

        let total_num_heads = num_heads;
        assert_eq!(total_num_heads % tp_size, 0);
        let num_heads = total_num_heads / tp_size;

        let total_num_kv_heads = num_kv_heads;
        assert_eq!(total_num_kv_heads % tp_size, 0);
        let num_kv_heads = total_num_kv_heads / tp_size;

        let head_dim = head_dim.unwrap_or(hidden_size / total_num_heads);
        let q_size = num_heads * head_dim;
        let kv_size = num_kv_heads * head_dim;
        let scaling = (head_dim as f32).powf(-0.5);

        let qkv_proj = QKVParallelLinear::new(
            hidden_size,
            head_dim,
            total_num_heads,
            Some(total_num_kv_heads),
            qkv_bias,
            comm.clone(),
            device,
        )?;
        let o_proj = RowParallelLinear::new(
            total_num_heads * head_dim,
            hidden_size,
            false,
            comm,
            device,
        )?;
        let rotary_emb = RotaryEmbedding::new(head_dim, head_dim, max_position, rope_theta, device)?;
        let attn = Attention::new(num_heads, head_dim, scaling, num_kv_heads);

        let (q_norm, k_norm) = if !qkv_bias {
            (
                Some(RMSNorm::new(head_dim, rms_norm_eps, device)?),
                Some(RMSNorm::new(head_dim, rms_norm_eps, device)?),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            total_num_heads,
            num_heads,
            total_num_kv_heads,
            num_kv_heads,
            head_dim,
            q_size,
            kv_size,
            scaling,
            qkv_bias,
            qkv_proj,
            o_proj,
            rotary_emb,
            attn,
            q_norm,
            k_norm,
        })
    }

    pub fn set_kv_cache(&mut self, k_cache: Py<PyAny>, v_cache: Py<PyAny>) {
        self.attn.set_kv_cache(k_cache, v_cache);
    }

    pub fn forward(&mut self, positions: &Tensor, hidden_states: &Tensor) -> Result<Tensor>{
        let qkv = self.qkv_proj.forward(hidden_states)?;
        let (q,k,v) = split_qkv(&qkv, self.q_size, self.kv_size)?;
        // q = [num_tokens, num_heads, head_dim]
        let num_tokens = q.dim(0)?;
        let q = q.reshape((num_tokens, self.num_heads, self.head_dim))?;
        let k = k.reshape((num_tokens, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((num_tokens, self.num_kv_heads, self.head_dim))?;
        let (q, k) = if !self.qkv_bias {
            let q = self.q_norm.as_ref().unwrap().forward(&q)?;
            let k = self.k_norm.as_ref().unwrap().forward(&k)?;
            (q, k)
        } else {
            (q, k)
        };
        let (q, k) = self.rotary_emb.forward(positions, &q, &k)?;
        let o = self.attn.forward(&q, &k, &v)?;
        // o = [num_tokens, num_heads,head_dim]
        let o_flatten = o.reshape((num_tokens, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&o_flatten)
    }
}

pub struct Qwen3MLP{
    gate_up_proj: MergedColumnParallelLinear,
    down_proj: RowParallelLinear,
    act_fn: SiluAndMul,
}

impl Qwen3MLP{
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        hidden_act: &str,
        comm: Option<Arc<Comm>>,
        device: &Device,
    ) -> Result<Self> {
        let gate_up_proj = MergedColumnParallelLinear::new(
            hidden_size,
            vec![intermediate_size; 2],
            false,
            comm.clone(),
            device,
        )?;
        let down_proj = RowParallelLinear::new(intermediate_size, hidden_size, false, comm, device)?;
        assert_eq!(hidden_act, "silu");
        let act_fn = SiluAndMul;

        Ok(Self { gate_up_proj, down_proj, act_fn })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let x = self.act_fn.forward(&gate_up)?;
        self.down_proj.forward(&x)
    }
}

pub struct Qwen3DecoderLayer{
    self_attn: Qwen3Attention,
    mlp: Qwen3MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl Qwen3DecoderLayer{
    pub fn new(config: &Config, comm: Option<Arc<Comm>>, device: &Device) -> Result<Self> {
        let self_attn = Qwen3Attention::new(
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.max_position_embeddings,
            Some(config.head_dim),
            config.rms_norm_eps as f32,
            config.attention_bias,
            config.rope_theta as f32,
            comm.clone(),
            device,
        )?;
        let mlp = Qwen3MLP::new(
            config.hidden_size,
            config.intermediate_size,
            &config.hidden_act,
            comm,
            device,
        )?;
        let input_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps as f32, device)?;
        let post_attention_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps as f32, device)?;

        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm })
    }

    pub fn set_kv_cache(&mut self, k_cache: Py<PyAny>, v_cache: Py<PyAny>) {
        self.self_attn.set_kv_cache(k_cache, v_cache);
    }

    pub fn forward(
        &mut self,
        positions: &Tensor,
        hidden_states: Tensor,
        residual: Option<Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (hidden_states, residual) = match residual {
            None => {
                let normed = self.input_layernorm.forward(&hidden_states)?;
                (normed, hidden_states)
            }
            Some(residual) => self.input_layernorm.residual_forward(hidden_states, residual)?,
        };
        let hidden_states = self.self_attn.forward(positions, &hidden_states)?;
        let (hidden_states, residual) = self.post_attention_layernorm.residual_forward(hidden_states, residual)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        Ok((hidden_states, residual))
    }
}

pub struct Qwen3Model{
    embed_tokens: VocabParallelEmbedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RMSNorm,
}

impl Qwen3Model{
    pub fn new(config: &Config, comm: Option<Arc<Comm>>, device: &Device) -> Result<Self> {
        let (tp_rank, tp_size) = match &comm {
            Some(comm) => (comm.rank(), comm.world_size()),
            None => (0, 1),
        };
        let embed_tokens = VocabParallelEmbedding::new(
            config.vocab_size,
            config.hidden_size,
            tp_rank,
            tp_size,
            comm.clone(),
            device,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::new(config, comm.clone(), device)?);
        }

        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps as f32, device)?;

        Ok(Self { embed_tokens, layers, norm })
    }

    pub fn set_kv_caches(&mut self, kv_caches: Vec<(Py<PyAny>, Py<PyAny>)>) {
        assert_eq!(kv_caches.len(), self.layers.len());
        for (layer, (k_cache, v_cache)) in self.layers.iter_mut().zip(kv_caches) {
            layer.set_kv_cache(k_cache, v_cache);
        }
    }

    pub fn forward(&mut self, input_ids: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embed_tokens.forward(input_ids)?;
        let mut residual: Option<Tensor> = None;

        for layer in self.layers.iter_mut() {
            let (hs, res) = layer.forward(positions, hidden_states, residual)?;
            hidden_states = hs;
            residual = Some(res);
        }

        let (hidden_states, _) = self.norm.residual_forward(hidden_states, residual.unwrap())?;
        Ok(hidden_states)
    }
}

pub struct Qwen3ForCausalLM{
    model: Qwen3Model,
    lm_head: ParallelLMHead,
    tie_word_embeddings: bool,
}

impl Qwen3ForCausalLM{
    pub fn new(config: &Config, comm: Option<Arc<Comm>>, device: &Device) -> Result<Self> {
        let model = Qwen3Model::new(config, comm.clone(), device)?;

        let (tp_rank, tp_size) = match &comm {
            Some(comm) => (comm.rank(), comm.world_size()),
            None => (0, 1),
        };
        let lm_head = ParallelLMHead::new(
            config.vocab_size,
            config.hidden_size,
            false,
            tp_rank,
            tp_size,
            comm,
            device,
        )?;

        // Not tied here yet: candle's `Tensor` is an immutable value, unlike
        // torch's `.data`, which shares mutable storage — `tie_weights` only
        // snapshots `embed_tokens.weight` at the moment it's called, so it must
        // run after checkpoint loading (`tie_weights_if_configured`), not before.
        Ok(Self { model, lm_head, tie_word_embeddings: config.tie_word_embeddings })
    }

    /// Call after `loader::load_model` has populated `embed_tokens.weight` from
    /// the checkpoint, so the tied `lm_head.weight` snapshot reflects real values.
    pub fn tie_weights_if_configured(&mut self) {
        if self.tie_word_embeddings {
            self.lm_head.tie_weights(&self.model.embed_tokens);
        }
    }

    pub fn set_kv_caches(&mut self, kv_caches: Vec<(Py<PyAny>, Py<PyAny>)>) {
        self.model.set_kv_caches(kv_caches);
    }

    pub fn forward(&mut self, input_ids: &Tensor, positions: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids, positions)
    }

    pub fn compute_logits(&self, hidden_states: &Tensor) -> Result<Tensor> {
        self.lm_head.forward(hidden_states)
    }
}

impl ModelWeights for Qwen3ForCausalLM {
    fn packed_modules_mapping(&self) -> HashMap<String, (String, ShardId)> {
        HashMap::from([
            ("q_proj".to_string(), ("qkv_proj".to_string(), ShardId::Name("q"))),
            ("k_proj".to_string(), ("qkv_proj".to_string(), ShardId::Name("k"))),
            ("v_proj".to_string(), ("qkv_proj".to_string(), ShardId::Name("v"))),
            ("gate_proj".to_string(), ("gate_up_proj".to_string(), ShardId::Index(0))),
            ("up_proj".to_string(), ("gate_up_proj".to_string(), ShardId::Index(1))),
        ])
    }

    fn load_weight(&mut self, param_name: &str, loaded_weight: Tensor, shard_id: Option<ShardId>) -> Result<()> {
        if let Some(rest) = param_name.strip_prefix("model.layers.") {
            let mut parts = rest.splitn(2, '.');
            let idx: usize = match parts.next().and_then(|s| s.parse().ok()) {
                Some(idx) => idx,
                None => return Ok(()),
            };
            let suffix = parts.next().unwrap_or("");
            let layer = match self.model.layers.get_mut(idx) {
                Some(layer) => layer,
                None => return Ok(()),
            };

            return match suffix {
                "self_attn.qkv_proj.weight" => match shard_id {
                    Some(ShardId::Name(shard)) => layer.self_attn.qkv_proj.weight_loader(loaded_weight, shard),
                    _ => candle_core::bail!("qkv_proj requires a name shard id, got {param_name}"),
                },
                "self_attn.o_proj.weight" => layer.self_attn.o_proj.weight_loader(loaded_weight),
                "self_attn.q_norm.weight" => {
                    if let Some(q_norm) = layer.self_attn.q_norm.as_mut() {
                        q_norm.weight_loader(loaded_weight);
                    }
                    Ok(())
                }
                "self_attn.k_norm.weight" => {
                    if let Some(k_norm) = layer.self_attn.k_norm.as_mut() {
                        k_norm.weight_loader(loaded_weight);
                    }
                    Ok(())
                }
                "mlp.gate_up_proj.weight" => match shard_id {
                    Some(ShardId::Index(shard)) => layer.mlp.gate_up_proj.weight_loader(loaded_weight, shard),
                    _ => candle_core::bail!("gate_up_proj requires an index shard id, got {param_name}"),
                },
                "mlp.down_proj.weight" => layer.mlp.down_proj.weight_loader(loaded_weight),
                "input_layernorm.weight" => {
                    layer.input_layernorm.weight_loader(loaded_weight);
                    Ok(())
                }
                "post_attention_layernorm.weight" => {
                    layer.post_attention_layernorm.weight_loader(loaded_weight);
                    Ok(())
                }
                // Unrecognized per-layer key (e.g. a saved rotary-embedding buffer) — skip.
                _ => Ok(()),
            };
        }

        match param_name {
            "model.embed_tokens.weight" => self.model.embed_tokens.weight_loader(loaded_weight),
            "model.norm.weight" => {
                self.model.norm.weight_loader(loaded_weight);
                Ok(())
            }
            "lm_head.weight" => self.lm_head.weight_loader(loaded_weight),
            // Unrecognized top-level key — skip.
            _ => Ok(()),
        }
    }
}