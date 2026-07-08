use std::collections::{HashMap, HashSet, VecDeque};
use xxhash_rust::xxh64::Xxh64;
use crate::engine::sequence::{self, Sequence};

pub struct Block {
    pub block_id: usize,
    pub ref_count: usize,
    pub hash: i64,
    pub token_ids: Vec<u32>,
}

impl Block {
    pub fn new(block_id: usize) -> Self {
        Self {
            block_id,
            ref_count: 0,
            hash: -1,
            token_ids: Vec::new(),
        }
    }

    pub fn update(&mut self, hash: i64, token_ids: Vec<u32>) {
        self.hash = hash;
        self.token_ids = token_ids;
    }

    pub fn reset(&mut self) {
        self.ref_count = 1;
        self.hash = -1;
        self.token_ids = Vec::new();
    }
}

pub struct BlockManager{
    pub block_size: usize,
    pub blocks: Vec<Block>,
    pub hash_to_block_id: HashMap<i64, usize>,
    pub free_block_ids: VecDeque<usize>,
    pub used_block_ids: HashSet<usize>,
}

impl BlockManager{
    pub fn new(num_blocks: usize, block_size: usize)->Self{
        Self {
            block_size,
            blocks: (0..num_blocks).map(Block::new).collect(),
            hash_to_block_id: HashMap::new(),
            free_block_ids: (0..num_blocks).collect(),
            used_block_ids: HashSet::new(),
        }
    }

    pub fn compute_hash(token_ids: &[i64], prefix: i64) -> i64{
        let mut hasher = Xxh64::new(0);

        if prefix != -1 {
            hasher.update(&prefix.to_le_bytes());
        }

        for &t in token_ids{
            hasher.update(&t.to_le_bytes());
        }
        // return the hash value
        hasher.digest() as i64
    }

    fn allocate_block(&mut self, block_id: usize) -> &Block {
        assert_eq!(self.blocks[block_id].ref_count, 0);

        let hash = self.blocks[block_id].hash;
        if self.hash_to_block_id.get(&hash) == Some(&block_id) {
            self.hash_to_block_id.remove(&hash);
        }

        self.blocks[block_id].reset();

        if let Some(pos) = self.free_block_ids.iter().position(|&id| id == block_id) {
            self.free_block_ids.remove(pos);
        }
        self.used_block_ids.insert(block_id);

        &self.blocks[block_id]
    }

    fn deallocate_block(&mut self, block_id: usize) {
        assert_eq!(self.blocks[block_id].ref_count, 0);
        self.used_block_ids.remove(&block_id);
        self.free_block_ids.push_back(block_id);
    }

    // "Only for seq in the waiting queue."
    pub fn can_allocate(&self, num_tokens: usize) -> bool {
        // for sequences in the waiting queue
        self.free_block_ids.len() >= (num_tokens + self.block_size - 1) / self.block_size
    }

    // Only for seq in the waiting queue.
    pub fn get_token_layout(&self, seq: &Sequence) -> (usize, usize, usize) {
        assert!(seq.block_table.is_empty());

        let mut num_new_tokens = 0usize;
        let mut num_new_computed_tokens_in_used = 0usize;
        let mut num_new_computed_tokens_in_free = 0usize;
        let mut h: i64 = -1;
        let mut cache_miss = false;

        for i in 0..seq.num_blocks() {
            let token_ids = seq.block(i);
            let token_ids_i64: Vec<i64> = token_ids.iter().map(|&t| t as i64).collect();

            h = if token_ids.len() == self.block_size {
                // hash_i = xxhash(prefix_bytes = hash_{i-1}, token_bytes = block_i's token IDs)
                Self::compute_hash(&token_ids_i64, h)
            } else {
                -1
            };

            let block_id: i64 = self.hash_to_block_id.get(&h).map(|&id| id as i64).unwrap_or(-1);
            // check hash and token ids for any false cache match
            if block_id == -1
                || self.blocks[block_id as usize].token_ids != token_ids
                || i == seq.num_blocks() - 1
            {
                cache_miss = true;
            }

            if cache_miss {
                num_new_tokens += token_ids.len();
            } else if self.used_block_ids.contains(&(block_id as usize)) {
                num_new_computed_tokens_in_used += token_ids.len();
            } else {
                num_new_computed_tokens_in_free += token_ids.len();
            }
        }

        (num_new_computed_tokens_in_used, num_new_computed_tokens_in_free, num_new_tokens)
    }

    pub fn allocate(&mut self, seq: &mut Sequence){
        // Sequences in the waiting queue
        assert!(seq.block_table.is_empty());
        let mut h: i64 = -1;

        for i in 0..seq.num_blocks(){
            let token_ids = seq.block(i).to_vec();
            let token_ids_i64: Vec<i64> = token_ids.iter().map(|&t| t as i64).collect();

            h = if token_ids.len() == self.block_size {
                Self::compute_hash(&token_ids_i64, h)
            } else {
                -1
            };

            let block_id: i64 = self.hash_to_block_id.get(&h).copied().map(|id| id as i64).unwrap_or(-1);
            if block_id == -1
                || self.blocks[block_id as usize].token_ids != token_ids
                || i == seq.num_blocks() - 1
                {
                    break; // cache miss
                }
            seq.num_cached_tokens += self.block_size;

            let block_id = block_id as usize;
            if self.used_block_ids.contains(&block_id){
                self.blocks[block_id].ref_count += 1
            } else {
                self.allocate_block(block_id);
            }
            self.blocks[block_id].update(h, token_ids);
            self.hash_to_block_id.insert(h, block_id);
            seq.block_table.push(block_id);
        }

        let start = seq.num_cached_tokens;
        let end = seq.num_cached_tokens + seq.num_new_tokens;
        let mut i = start;
        while i < end {
            let chunk_end = std::cmp::min(i + self.block_size, end);
            let token_ids = seq.slice_tokens(i, chunk_end);
            let token_ids_i64: Vec<i64> = token_ids.iter().map(|&t| t as i64).collect();

            if i != start {
                h = if token_ids.len() == self.block_size {
                    Self::compute_hash(&token_ids_i64, h)
                } else {
                    -1
                };
            }

            let block_id = self.free_block_ids[0];
            self.allocate_block(block_id);

            if h != -1 {
                self.blocks[block_id].update(h, token_ids.to_vec());
                self.hash_to_block_id.insert(h, block_id);
            }
            seq.block_table.push(block_id);

            i += self.block_size;
        }
    }

    // For finished seq or preempted seq in the running queue.
    pub fn deallocate(&mut self, seq: &mut Sequence) {
        for &block_id in seq.block_table.iter().rev() {
            self.blocks[block_id].ref_count -= 1;
            if self.blocks[block_id].ref_count == 0 {
                self.deallocate_block(block_id);
            }
        }
        seq.num_cached_tokens = 0;
        seq.num_new_tokens = 0;
        seq.block_table.clear();
    }

    // Only for seq in the running queue.
    pub fn can_append(&self, seq: &Sequence, num_new_tokens: usize) -> bool {
        let block_size = self.block_size as i64;

        let mut last_computed_block_capacity = block_size - (seq.num_cached_tokens as i64 % block_size);
        if last_computed_block_capacity == block_size {
            last_computed_block_capacity = 0;
        }

        let needed_blocks =
            (num_new_tokens as i64 - last_computed_block_capacity + block_size - 1).div_euclid(block_size);

        needed_blocks <= self.free_block_ids.len() as i64
    }

    pub fn may_append(&mut self, seq: &mut Sequence){
        let range_start = seq.num_cached_blocks() * self.block_size;
        let range_end = seq.num_cached_tokens + seq.num_new_tokens;

        let mut i = range_start;
        while i < range_end{
            let chunk_end = std::cmp::min(i+self.block_size, range_end);
            let token_ids = seq.slice_tokens(i, chunk_end).to_vec();
            let token_ids_i64: Vec<i64> = token_ids.iter().map(|&t| t as i64).collect();

            let block_idx = i / self.block_size;
            let current_block_id: i64 = if block_idx < seq.block_table.len() {
                seq.block_table[block_idx] as i64
            } else {
                -1
            };
            // current_block_id -1 means the last block that is currently being generated
            // or the cache has not been generated yet
            if current_block_id != -1 {
                assert_eq!(self.blocks[current_block_id as usize].hash, -1);
            }

            if token_ids.len() % self.block_size == 0{
                let previous_block_id: i64 = if i >= self.block_size {
                    seq.block_table[block_idx - 1] as i64
                } else {
                    -1
                };
                let prefix: i64 = if previous_block_id != -1 {
                    self.blocks[previous_block_id as usize].hash
                } else {
                    -1
                };
                let h = Self::compute_hash(&token_ids_i64, prefix);

                let current_block_id = if current_block_id == -1 {
                    let block_id = self.free_block_ids[0];
                    self.allocate_block(block_id);
                    seq.block_table.push(block_id);
                    block_id
                } else {
                    current_block_id as usize
                };

                self.blocks[current_block_id].update(h, token_ids.to_vec());
                self.hash_to_block_id.insert(h, self.blocks[current_block_id].block_id);
                } else if current_block_id == -1 {
                    let block_id = self.free_block_ids[0];
                    self.allocate_block(block_id);
                    seq.block_table.push(block_id);
                }

            i += self.block_size;
        }
    }

}

