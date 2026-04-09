mod cache_options;

pub use cache_options::{CacheOptions, LuaAnalysisPhase};
use emmylua_parser::{LuaExpr, LuaSyntaxId, LuaVarExpr};
use hashbrown::{HashMap, HashSet};
use std::{rc::Rc, sync::Arc};

use crate::{
    FileId, FlowId, LuaFunctionType,
    db_index::LuaType,
    semantic::infer::{ConditionFlowAction, VarRefId},
};

#[derive(Debug)]
pub enum CacheEntry<T> {
    Ready,
    Cache(T),
}

#[derive(Debug, Clone)]
pub(in crate::semantic) struct FlowConditionInfo {
    pub expr: LuaExpr,
    pub index_var_ref_id: Option<VarRefId>,
    pub index_prefix_var_ref_id: Option<VarRefId>,
}

#[derive(Debug, Clone)]
pub(in crate::semantic) struct FlowAssignmentInfo {
    pub vars: Vec<LuaVarExpr>,
    pub exprs: Vec<LuaExpr>,
    pub var_ref_ids: Vec<Option<VarRefId>>,
}

#[derive(Debug)]
pub struct LuaInferCache {
    file_id: FileId,
    config: CacheOptions,
    pub expr_cache: HashMap<LuaSyntaxId, CacheEntry<LuaType>>,
    pub call_cache:
        HashMap<(LuaSyntaxId, Option<usize>, LuaType), CacheEntry<Arc<LuaFunctionType>>>,
    pub(in crate::semantic) flow_cache_var_ref_ids: HashMap<VarRefId, u32>,
    pub(in crate::semantic) next_flow_cache_var_ref_id: u32,
    pub(crate) flow_node_cache: Vec<HashMap<u32, [Option<CacheEntry<LuaType>>; 2]>>,
    pub(in crate::semantic) flow_branch_antecedent_cache: Vec<Option<Rc<Vec<FlowId>>>>,
    pub(in crate::semantic) flow_condition_info_cache: Vec<Option<Rc<FlowConditionInfo>>>,
    pub(in crate::semantic) flow_assignment_info_cache: Vec<Option<Rc<FlowAssignmentInfo>>>,
    pub(in crate::semantic) condition_flow_cache:
        Vec<HashMap<u32, [Option<CacheEntry<ConditionFlowAction>>; 2]>>,
    pub index_ref_origin_type_cache: HashMap<VarRefId, CacheEntry<LuaType>>,
    pub expr_var_ref_id_cache: HashMap<LuaSyntaxId, VarRefId>,
    pub narrow_by_literal_stop_position_cache: HashSet<LuaSyntaxId>,
}

impl LuaInferCache {
    pub fn new(file_id: FileId, config: CacheOptions) -> Self {
        Self {
            file_id,
            config,
            expr_cache: HashMap::new(),
            call_cache: HashMap::new(),
            flow_cache_var_ref_ids: HashMap::new(),
            next_flow_cache_var_ref_id: 0,
            flow_node_cache: Vec::new(),
            flow_branch_antecedent_cache: Vec::new(),
            flow_condition_info_cache: Vec::new(),
            flow_assignment_info_cache: Vec::new(),
            condition_flow_cache: Vec::new(),
            index_ref_origin_type_cache: HashMap::new(),
            expr_var_ref_id_cache: HashMap::new(),
            narrow_by_literal_stop_position_cache: HashSet::new(),
        }
    }

    pub fn get_config(&self) -> &CacheOptions {
        &self.config
    }

    pub fn get_file_id(&self) -> FileId {
        self.file_id
    }

    pub fn set_phase(&mut self, phase: LuaAnalysisPhase) {
        self.config.analysis_phase = phase;
    }

    pub fn clear(&mut self) {
        self.expr_cache.clear();
        self.call_cache.clear();
        self.flow_cache_var_ref_ids.clear();
        self.next_flow_cache_var_ref_id = 0;
        self.flow_node_cache.clear();
        self.flow_branch_antecedent_cache.clear();
        self.flow_condition_info_cache.clear();
        self.flow_assignment_info_cache.clear();
        self.condition_flow_cache.clear();
        self.index_ref_origin_type_cache.clear();
        self.expr_var_ref_id_cache.clear();
    }
}
