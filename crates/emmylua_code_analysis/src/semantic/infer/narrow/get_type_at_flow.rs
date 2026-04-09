use emmylua_parser::{LuaAssignStat, LuaAstNode, LuaChunk, LuaExpr, LuaVarExpr, UnaryOperator};
use hashbrown::{HashMap, HashSet};
use std::rc::Rc;

use crate::{
    CacheEntry, DbIndex, FlowId, FlowNode, FlowNodeKind, FlowTree, InferFailReason, LuaDeclId,
    LuaInferCache, LuaMemberId, LuaSignatureId, LuaType, TypeOps, check_type_compact, infer_expr,
    semantic::{
        cache::{FlowAssignmentInfo, FlowConditionInfo},
        infer::{
            InferResult, VarRefId, infer_expr_list_value_type_at,
            narrow::{
                ResultTypeOrContinue,
                condition_flow::{
                    ConditionFlowAction, InferConditionFlow, PendingConditionNarrow,
                    get_type_at_condition_flow,
                },
                get_multi_antecedents, get_single_antecedent,
                get_type_at_cast_flow::get_type_at_cast_flow,
                get_var_ref_type, narrow_down_type,
                var_ref_id::get_var_expr_var_ref_id,
            },
        },
        member::find_members,
    },
};

enum FlowEvalFrame {
    Enter {
        flow_id: FlowId,
        use_condition_narrowing: bool,
    },
    PendingUnion {
        flow_id: FlowId,
        use_condition_narrowing: bool,
        branch_flow_ids: Rc<Vec<FlowId>>,
        next_pending_index: usize,
        pending_condition_narrows: Vec<Rc<PendingConditionNarrow>>,
        branch_result_type: LuaType,
    },
}

enum FlowEvalStep {
    Result(LuaType),
    Union {
        branch_flow_ids: Rc<Vec<FlowId>>,
        pending_condition_narrows: Vec<Rc<PendingConditionNarrow>>,
    },
}

pub fn get_type_at_flow(
    db: &DbIndex,
    tree: &FlowTree,
    cache: &mut LuaInferCache,
    root: &LuaChunk,
    var_ref_id: &VarRefId,
    flow_id: FlowId,
) -> InferResult {
    let var_ref_cache_id = get_flow_cache_var_ref_id(cache, var_ref_id);
    get_type_at_flow_internal(
        db,
        tree,
        cache,
        root,
        var_ref_id,
        var_ref_cache_id,
        flow_id,
        true,
    )
}

pub(in crate::semantic) fn get_flow_cache_var_ref_id(
    cache: &mut LuaInferCache,
    var_ref_id: &VarRefId,
) -> u32 {
    if let Some(var_ref_cache_id) = cache.flow_cache_var_ref_ids.get(var_ref_id) {
        return *var_ref_cache_id;
    }

    // Hot flow caches are direct-indexed by this synthetic id to avoid hashing VarRefId on
    // every backward step.
    let var_ref_cache_id = cache.next_flow_cache_var_ref_id;
    cache.next_flow_cache_var_ref_id += 1;
    cache
        .flow_cache_var_ref_ids
        .insert(var_ref_id.clone(), var_ref_cache_id);
    var_ref_cache_id
}

fn get_dense_bool_cache_entry<T>(
    cache: &[HashMap<u32, [Option<CacheEntry<T>>; 2]>],
    outer_index: usize,
    inner_index: u32,
    flag: bool,
) -> Option<&CacheEntry<T>> {
    cache
        .get(outer_index)
        .and_then(|entries| entries.get(&inner_index))
        .and_then(|entry| entry[flag as usize].as_ref())
}

fn get_dense_bool_cache_slot<T>(
    cache: &mut Vec<HashMap<u32, [Option<CacheEntry<T>>; 2]>>,
    outer_index: usize,
    inner_index: u32,
    flag: bool,
) -> &mut Option<CacheEntry<T>> {
    if cache.len() <= outer_index {
        cache.resize_with(outer_index + 1, HashMap::new);
    }
    let entries = &mut cache[outer_index];
    &mut entries.entry(inner_index).or_insert_with(|| [None, None])[flag as usize]
}

fn get_dense_rc_entry<T>(cache: &[Option<Rc<T>>], index: usize) -> Option<&Rc<T>> {
    cache.get(index).and_then(|entry| entry.as_ref())
}

fn get_dense_rc_slot<T>(cache: &mut Vec<Option<Rc<T>>>, index: usize) -> &mut Option<Rc<T>> {
    if cache.len() <= index {
        cache.resize(index + 1, None);
    }
    &mut cache[index]
}

fn get_flow_node_cache_entry(
    cache: &LuaInferCache,
    var_ref_cache_id: u32,
    flow_id: FlowId,
    use_condition_narrowing: bool,
) -> Option<&CacheEntry<LuaType>> {
    get_dense_bool_cache_entry(
        &cache.flow_node_cache,
        var_ref_cache_id as usize,
        flow_id.0,
        use_condition_narrowing,
    )
}

fn get_flow_node_cache_slot(
    cache: &mut LuaInferCache,
    var_ref_cache_id: u32,
    flow_id: FlowId,
    use_condition_narrowing: bool,
) -> &mut Option<CacheEntry<LuaType>> {
    get_dense_bool_cache_slot(
        &mut cache.flow_node_cache,
        var_ref_cache_id as usize,
        flow_id.0,
        use_condition_narrowing,
    )
}

fn get_condition_flow_cache_entry(
    cache: &LuaInferCache,
    var_ref_cache_id: u32,
    flow_id: FlowId,
    is_true_condition: bool,
) -> Option<&CacheEntry<ConditionFlowAction>> {
    get_dense_bool_cache_entry(
        &cache.condition_flow_cache,
        var_ref_cache_id as usize,
        flow_id.0,
        is_true_condition,
    )
}

fn get_condition_flow_cache_slot(
    cache: &mut LuaInferCache,
    var_ref_cache_id: u32,
    flow_id: FlowId,
    is_true_condition: bool,
) -> &mut Option<CacheEntry<ConditionFlowAction>> {
    get_dense_bool_cache_slot(
        &mut cache.condition_flow_cache,
        var_ref_cache_id as usize,
        flow_id.0,
        is_true_condition,
    )
}

pub(in crate::semantic) fn get_condition_flow_action(
    db: &DbIndex,
    tree: &FlowTree,
    cache: &mut LuaInferCache,
    root: &LuaChunk,
    var_ref_id: &VarRefId,
    var_ref_cache_id: u32,
    flow_node: &FlowNode,
    condition_info: &FlowConditionInfo,
    condition_flow: InferConditionFlow,
) -> Result<ConditionFlowAction, InferFailReason> {
    if condition_info.index_var_ref_id.is_some()
        && condition_info.index_var_ref_id.as_ref() != Some(var_ref_id)
        && condition_info.index_prefix_var_ref_id.as_ref() != Some(var_ref_id)
    {
        return Ok(ConditionFlowAction::Continue);
    }

    let is_true_condition = matches!(condition_flow, InferConditionFlow::TrueCondition);
    if let Some(cache_entry) =
        get_condition_flow_cache_entry(cache, var_ref_cache_id, flow_node.id, is_true_condition)
    {
        return match cache_entry {
            CacheEntry::Cache(action) => Ok(action.clone()),
            CacheEntry::Ready => Err(InferFailReason::RecursiveInfer),
        };
    }

    *get_condition_flow_cache_slot(cache, var_ref_cache_id, flow_node.id, is_true_condition) =
        Some(CacheEntry::Ready);
    let result = get_type_at_condition_flow(
        db,
        tree,
        cache,
        root,
        var_ref_id,
        flow_node,
        condition_info.expr.clone(),
        condition_flow,
    );
    match &result {
        Ok(action) => {
            *get_condition_flow_cache_slot(
                cache,
                var_ref_cache_id,
                flow_node.id,
                is_true_condition,
            ) = Some(CacheEntry::Cache(action.clone()));
        }
        Err(_) => {
            *get_condition_flow_cache_slot(
                cache,
                var_ref_cache_id,
                flow_node.id,
                is_true_condition,
            ) = None;
        }
    }

    result
}

fn apply_pending_condition_narrows(
    db: &DbIndex,
    cache: &mut LuaInferCache,
    result_type: LuaType,
    pending_condition_narrows: Vec<Rc<PendingConditionNarrow>>,
) -> LuaType {
    pending_condition_narrows.into_iter().rev().fold(
        result_type,
        |result_type, pending_condition_narrow| {
            pending_condition_narrow.apply(db, cache, result_type)
        },
    )
}

fn can_reuse_narrowed_assignment_source(
    db: &DbIndex,
    narrowed_source_type: &LuaType,
    expr_type: &LuaType,
) -> bool {
    if matches!(expr_type, LuaType::TableConst(_) | LuaType::Object(_)) {
        return is_partial_assignment_expr_compatible(db, narrowed_source_type, expr_type);
    }

    if !is_exact_assignment_expr_type(expr_type) {
        return false;
    }

    match narrow_down_type(db, narrowed_source_type.clone(), expr_type.clone(), None) {
        Some(narrowed_expr_type) => narrowed_expr_type == *expr_type,
        None => true,
    }
}

fn preserves_assignment_expr_type(typ: &LuaType) -> bool {
    matches!(typ, LuaType::TableConst(_) | LuaType::Object(_)) || is_exact_assignment_expr_type(typ)
}

fn is_partial_assignment_expr_compatible(
    db: &DbIndex,
    source_type: &LuaType,
    expr_type: &LuaType,
) -> bool {
    if check_type_compact(db, source_type, expr_type).is_ok() {
        return true;
    }

    // Only preserve branch narrowing for concrete partial table/object literals.
    // Broader RHS expressions can carry hidden state the current flow/type model cannot represent
    // without wider semantic changes.
    if !matches!(expr_type, LuaType::TableConst(_) | LuaType::Object(_)) {
        return false;
    }

    let expr_members = find_members(db, expr_type).unwrap_or_default();

    if expr_members.is_empty() {
        return true;
    }

    let Some(source_members) = find_members(db, source_type) else {
        return false;
    };

    expr_members.into_iter().all(|expr_member| {
        match source_members
            .iter()
            .find(|source_member| source_member.key == expr_member.key)
        {
            Some(source_member) => {
                is_partial_assignment_expr_compatible(db, &source_member.typ, &expr_member.typ)
            }
            None => true,
        }
    })
}

fn is_exact_assignment_expr_type(typ: &LuaType) -> bool {
    match typ {
        LuaType::Nil | LuaType::DocBooleanConst(_) => true,
        typ if typ.is_const() => !matches!(typ, LuaType::TableConst(_)),
        LuaType::Union(union) => union.into_vec().iter().all(is_exact_assignment_expr_type),
        LuaType::MultiLineUnion(multi_union) => {
            is_exact_assignment_expr_type(&multi_union.to_union())
        }
        LuaType::TypeGuard(inner) => is_exact_assignment_expr_type(inner),
        _ => false,
    }
}

fn get_type_at_flow_internal(
    db: &DbIndex,
    tree: &FlowTree,
    cache: &mut LuaInferCache,
    root: &LuaChunk,
    var_ref_id: &VarRefId,
    var_ref_cache_id: u32,
    flow_id: FlowId,
    use_condition_narrowing: bool,
) -> InferResult {
    let mut frames = vec![FlowEvalFrame::Enter {
        flow_id,
        use_condition_narrowing,
    }];
    let mut last_result: Option<InferResult> = None;

    while let Some(frame) = frames.pop() {
        match frame {
            FlowEvalFrame::Enter {
                flow_id,
                use_condition_narrowing,
            } => {
                if let Some(cache_entry) = get_flow_node_cache_entry(
                    cache,
                    var_ref_cache_id,
                    flow_id,
                    use_condition_narrowing,
                ) {
                    last_result = Some(match cache_entry {
                        CacheEntry::Cache(narrow_type) => Ok(narrow_type.clone()),
                        CacheEntry::Ready => Err(InferFailReason::RecursiveInfer),
                    });
                    continue;
                }

                *get_flow_node_cache_slot(
                    cache,
                    var_ref_cache_id,
                    flow_id,
                    use_condition_narrowing,
                ) = Some(CacheEntry::Ready);
                match evaluate_flow_step(
                    db,
                    tree,
                    cache,
                    root,
                    var_ref_id,
                    var_ref_cache_id,
                    flow_id,
                    use_condition_narrowing,
                ) {
                    Ok(FlowEvalStep::Result(result_type)) => {
                        *get_flow_node_cache_slot(
                            cache,
                            var_ref_cache_id,
                            flow_id,
                            use_condition_narrowing,
                        ) = Some(CacheEntry::Cache(result_type.clone()));
                        last_result = Some(Ok(result_type));
                    }
                    Ok(FlowEvalStep::Union {
                        branch_flow_ids,
                        pending_condition_narrows,
                    }) => {
                        let Some(next_pending_index) = branch_flow_ids.len().checked_sub(1) else {
                            let result_type = apply_pending_condition_narrows(
                                db,
                                cache,
                                LuaType::Never,
                                pending_condition_narrows,
                            );
                            *get_flow_node_cache_slot(
                                cache,
                                var_ref_cache_id,
                                flow_id,
                                use_condition_narrowing,
                            ) = Some(CacheEntry::Cache(result_type.clone()));
                            last_result = Some(Ok(result_type));
                            continue;
                        };
                        let next_flow_id = branch_flow_ids[next_pending_index];

                        frames.push(FlowEvalFrame::PendingUnion {
                            flow_id,
                            use_condition_narrowing,
                            branch_flow_ids,
                            next_pending_index,
                            pending_condition_narrows,
                            branch_result_type: LuaType::Never,
                        });
                        frames.push(FlowEvalFrame::Enter {
                            flow_id: next_flow_id,
                            use_condition_narrowing,
                        });
                    }
                    Err(err) => {
                        *get_flow_node_cache_slot(
                            cache,
                            var_ref_cache_id,
                            flow_id,
                            use_condition_narrowing,
                        ) = None;
                        last_result = Some(Err(err));
                    }
                }
            }
            FlowEvalFrame::PendingUnion {
                flow_id,
                use_condition_narrowing,
                branch_flow_ids,
                next_pending_index,
                pending_condition_narrows,
                mut branch_result_type,
            } => {
                let Some(branch_result) = last_result.take() else {
                    *get_flow_node_cache_slot(
                        cache,
                        var_ref_cache_id,
                        flow_id,
                        use_condition_narrowing,
                    ) = None;
                    return Err(InferFailReason::None);
                };

                match branch_result {
                    Ok(branch_type) => {
                        branch_result_type =
                            TypeOps::Union.apply(db, &branch_result_type, &branch_type);
                        if next_pending_index > 0 {
                            let next_pending_index = next_pending_index - 1;
                            let next_flow_id = branch_flow_ids[next_pending_index];
                            frames.push(FlowEvalFrame::PendingUnion {
                                flow_id,
                                use_condition_narrowing,
                                branch_flow_ids,
                                next_pending_index,
                                pending_condition_narrows,
                                branch_result_type,
                            });
                            frames.push(FlowEvalFrame::Enter {
                                flow_id: next_flow_id,
                                use_condition_narrowing,
                            });
                        } else {
                            let result_type = apply_pending_condition_narrows(
                                db,
                                cache,
                                branch_result_type,
                                pending_condition_narrows,
                            );
                            *get_flow_node_cache_slot(
                                cache,
                                var_ref_cache_id,
                                flow_id,
                                use_condition_narrowing,
                            ) = Some(CacheEntry::Cache(result_type.clone()));
                            last_result = Some(Ok(result_type));
                        }
                    }
                    Err(err) => {
                        *get_flow_node_cache_slot(
                            cache,
                            var_ref_cache_id,
                            flow_id,
                            use_condition_narrowing,
                        ) = None;
                        last_result = Some(Err(err));
                    }
                }
            }
        }
    }

    last_result.unwrap_or(Err(InferFailReason::None))
}

fn evaluate_flow_step(
    db: &DbIndex,
    tree: &FlowTree,
    cache: &mut LuaInferCache,
    root: &LuaChunk,
    var_ref_id: &VarRefId,
    var_ref_cache_id: u32,
    flow_id: FlowId,
    use_condition_narrowing: bool,
) -> Result<FlowEvalStep, InferFailReason> {
    let result_type;
    let mut antecedent_flow_id = flow_id;
    let mut pending_condition_narrows: Vec<Rc<PendingConditionNarrow>> = Vec::new();
    loop {
        let flow_node = tree
            .get_flow_node(antecedent_flow_id)
            .ok_or(InferFailReason::None)?;

        match &flow_node.kind {
            FlowNodeKind::Start | FlowNodeKind::Unreachable => {
                result_type = get_var_ref_type(db, cache, var_ref_id)?;
                break;
            }
            FlowNodeKind::LoopLabel | FlowNodeKind::Break | FlowNodeKind::Return => {
                antecedent_flow_id = get_single_antecedent(flow_node)?;
            }
            FlowNodeKind::BranchLabel => {
                return Ok(FlowEvalStep::Union {
                    branch_flow_ids: get_branch_label_flow_ids(tree, cache, flow_node)?,
                    pending_condition_narrows,
                });
            }
            FlowNodeKind::NamedLabel(_) => {
                return Ok(FlowEvalStep::Union {
                    branch_flow_ids: Rc::new(get_multi_antecedents(tree, flow_node)?),
                    pending_condition_narrows,
                });
            }
            FlowNodeKind::DeclPosition(position) => {
                if *position <= var_ref_id.get_position() {
                    match get_var_ref_type(db, cache, var_ref_id) {
                        Ok(var_type) => {
                            result_type = var_type;
                            break;
                        }
                        Err(err) => {
                            if let Some(init_type) =
                                try_infer_decl_initializer_type(db, cache, root, var_ref_id)?
                            {
                                result_type = init_type;
                                break;
                            }

                            return Err(err);
                        }
                    }
                } else {
                    antecedent_flow_id = get_single_antecedent(flow_node)?;
                }
            }
            FlowNodeKind::Assignment(assign_ptr) => {
                let assignment_info =
                    get_flow_assignment_info(db, cache, root, flow_node.id, assign_ptr)?;
                if !assignment_info
                    .var_ref_ids
                    .iter()
                    .flatten()
                    .any(|assignment_var_ref_id| assignment_var_ref_id == var_ref_id)
                {
                    antecedent_flow_id = get_single_antecedent(flow_node)?;
                    continue;
                }

                let result_or_continue = get_type_at_assign_stat(
                    db,
                    tree,
                    cache,
                    root,
                    var_ref_id,
                    var_ref_cache_id,
                    flow_node,
                    &assignment_info,
                )?;

                if let ResultTypeOrContinue::Result(assign_type) = result_or_continue {
                    result_type = assign_type;
                    break;
                } else {
                    antecedent_flow_id = get_single_antecedent(flow_node)?;
                }
            }
            FlowNodeKind::ImplFunc(func_ptr) => {
                let func_stat = func_ptr.to_node(root).ok_or(InferFailReason::None)?;
                let Some(func_name) = func_stat.get_func_name() else {
                    antecedent_flow_id = get_single_antecedent(flow_node)?;
                    continue;
                };

                let Some(ref_id) = get_var_expr_var_ref_id(db, cache, func_name.to_expr()) else {
                    antecedent_flow_id = get_single_antecedent(flow_node)?;
                    continue;
                };

                if ref_id == *var_ref_id {
                    let Some(closure) = func_stat.get_closure() else {
                        return Err(InferFailReason::None);
                    };

                    result_type = LuaType::Signature(LuaSignatureId::from_closure(
                        cache.get_file_id(),
                        &closure,
                    ));
                    break;
                } else {
                    antecedent_flow_id = get_single_antecedent(flow_node)?;
                }
            }
            FlowNodeKind::TrueCondition(condition_ptr)
            | FlowNodeKind::FalseCondition(condition_ptr) => {
                if !use_condition_narrowing {
                    antecedent_flow_id = get_single_antecedent(flow_node)?;
                    continue;
                }

                let condition_info =
                    get_flow_condition_info(db, cache, root, flow_node.id, condition_ptr)?;
                let condition_flow = if matches!(&flow_node.kind, FlowNodeKind::TrueCondition(_)) {
                    InferConditionFlow::TrueCondition
                } else {
                    InferConditionFlow::FalseCondition
                };
                let condition_action = get_condition_flow_action(
                    db,
                    tree,
                    cache,
                    root,
                    var_ref_id,
                    var_ref_cache_id,
                    flow_node,
                    &condition_info,
                    condition_flow,
                )?;

                match condition_action {
                    ConditionFlowAction::Pending(pending_condition_narrow) => {
                        pending_condition_narrows.push(pending_condition_narrow);
                        antecedent_flow_id = get_single_antecedent(flow_node)?;
                    }
                    ConditionFlowAction::Result(condition_type) => {
                        result_type = condition_type;
                        break;
                    }
                    ConditionFlowAction::Continue => {
                        antecedent_flow_id = get_single_antecedent(flow_node)?;
                    }
                }
            }
            FlowNodeKind::ForIStat(_) => {
                antecedent_flow_id = get_single_antecedent(flow_node)?;
            }
            FlowNodeKind::TagCast(cast_ast_ptr) => {
                let tag_cast = cast_ast_ptr.to_node(root).ok_or(InferFailReason::None)?;
                let cast_or_continue =
                    get_type_at_cast_flow(db, tree, cache, root, var_ref_id, flow_node, tag_cast)?;

                if let ResultTypeOrContinue::Result(cast_type) = cast_or_continue {
                    result_type = cast_type;
                    break;
                } else {
                    antecedent_flow_id = get_single_antecedent(flow_node)?;
                }
            }
        }
    }

    let result_type = if use_condition_narrowing {
        apply_pending_condition_narrows(db, cache, result_type, pending_condition_narrows)
    } else {
        result_type
    };

    Ok(FlowEvalStep::Result(result_type))
}

pub(in crate::semantic) fn get_flow_condition_info(
    db: &DbIndex,
    cache: &mut LuaInferCache,
    root: &LuaChunk,
    flow_id: FlowId,
    condition_ptr: &emmylua_parser::LuaAstPtr<LuaExpr>,
) -> Result<Rc<FlowConditionInfo>, InferFailReason> {
    let flow_index = flow_id.0 as usize;
    if let Some(info) = get_dense_rc_entry(&cache.flow_condition_info_cache, flow_index) {
        return Ok(info.clone());
    }

    let expr = condition_ptr.to_node(root).ok_or(InferFailReason::None)?;
    let (index_var_ref_id, index_prefix_var_ref_id) =
        get_condition_index_var_refs(db, cache, expr.clone());
    let info = Rc::new(FlowConditionInfo {
        expr,
        index_var_ref_id,
        index_prefix_var_ref_id,
    });
    *get_dense_rc_slot(&mut cache.flow_condition_info_cache, flow_index) = Some(info.clone());
    Ok(info)
}

fn get_condition_index_var_refs(
    db: &DbIndex,
    cache: &mut LuaInferCache,
    condition: LuaExpr,
) -> (Option<VarRefId>, Option<VarRefId>) {
    match condition {
        LuaExpr::IndexExpr(index_expr) => {
            let index_var_ref_id =
                get_var_expr_var_ref_id(db, cache, LuaExpr::IndexExpr(index_expr.clone()));
            let index_prefix_var_ref_id = if index_var_ref_id.is_some() {
                index_expr
                    .get_prefix_expr()
                    .and_then(|prefix_expr| get_var_expr_var_ref_id(db, cache, prefix_expr))
            } else {
                None
            };
            (index_var_ref_id, index_prefix_var_ref_id)
        }
        LuaExpr::ParenExpr(paren_expr) => paren_expr
            .get_expr()
            .map(|expr| get_condition_index_var_refs(db, cache, expr))
            .unwrap_or((None, None)),
        LuaExpr::UnaryExpr(unary_expr) => {
            let Some(op_token) = unary_expr.get_op_token() else {
                return (None, None);
            };

            if op_token.get_op() != UnaryOperator::OpNot {
                return (None, None);
            }

            unary_expr
                .get_expr()
                .map(|expr| get_condition_index_var_refs(db, cache, expr))
                .unwrap_or((None, None))
        }
        _ => (None, None),
    }
}

fn get_branch_label_flow_ids(
    tree: &FlowTree,
    cache: &mut LuaInferCache,
    flow_node: &FlowNode,
) -> Result<Rc<Vec<FlowId>>, InferFailReason> {
    let flow_index = flow_node.id.0 as usize;
    if let Some(flow_ids) = get_dense_rc_entry(&cache.flow_branch_antecedent_cache, flow_index) {
        return Ok(flow_ids.clone());
    }

    let mut pending = get_multi_antecedents(tree, flow_node)?;
    let mut visited_labels = HashSet::with_capacity(pending.len());
    let mut branch_flow_ids = Vec::with_capacity(pending.len());

    while let Some(flow_id) = pending.pop() {
        let branch_flow_node = tree.get_flow_node(flow_id).ok_or(InferFailReason::None)?;
        match &branch_flow_node.kind {
            FlowNodeKind::BranchLabel => {
                if !visited_labels.insert(flow_id) {
                    continue;
                }

                if let Some(cached_flow_ids) =
                    get_dense_rc_entry(&cache.flow_branch_antecedent_cache, flow_id.0 as usize)
                {
                    branch_flow_ids.extend(cached_flow_ids.iter().copied());
                } else {
                    pending.extend(get_multi_antecedents(tree, branch_flow_node)?);
                }
            }
            _ => branch_flow_ids.push(flow_id),
        }
    }

    // Merge fans repeat across many queries, so keep the flattened inputs shared once cached.
    let branch_flow_ids = Rc::new(branch_flow_ids);
    *get_dense_rc_slot(&mut cache.flow_branch_antecedent_cache, flow_index) =
        Some(branch_flow_ids.clone());
    Ok(branch_flow_ids)
}

pub(in crate::semantic) fn get_flow_assignment_info(
    db: &DbIndex,
    cache: &mut LuaInferCache,
    root: &LuaChunk,
    flow_id: FlowId,
    assign_ptr: &emmylua_parser::LuaAstPtr<LuaAssignStat>,
) -> Result<Rc<FlowAssignmentInfo>, InferFailReason> {
    let flow_index = flow_id.0 as usize;
    if let Some(info) = get_dense_rc_entry(&cache.flow_assignment_info_cache, flow_index) {
        return Ok(info.clone());
    }

    let assign_stat = assign_ptr.to_node(root).ok_or(InferFailReason::None)?;
    let (vars, exprs) = assign_stat.get_var_and_expr_list();
    let var_ref_ids = vars
        .iter()
        .cloned()
        .map(|var| get_var_expr_var_ref_id(db, cache, var.to_expr()))
        .collect::<Vec<_>>();
    let info = Rc::new(FlowAssignmentInfo {
        vars,
        exprs,
        var_ref_ids,
    });
    *get_dense_rc_slot(&mut cache.flow_assignment_info_cache, flow_index) = Some(info.clone());
    Ok(info)
}

fn get_type_at_assign_stat(
    db: &DbIndex,
    tree: &FlowTree,
    cache: &mut LuaInferCache,
    root: &LuaChunk,
    var_ref_id: &VarRefId,
    var_ref_cache_id: u32,
    flow_node: &FlowNode,
    assignment_info: &FlowAssignmentInfo,
) -> Result<ResultTypeOrContinue, InferFailReason> {
    for (i, (var, maybe_ref_id)) in assignment_info
        .vars
        .iter()
        .cloned()
        .zip(assignment_info.var_ref_ids.iter())
        .enumerate()
    {
        let Some(maybe_ref_id) = maybe_ref_id.as_ref() else {
            continue;
        };

        if maybe_ref_id != var_ref_id {
            // let typ = get_var_ref_type(db, cache, var_ref_id)?;
            continue;
        }

        // Check if there's an explicit type annotation (not just inferred type)
        let var_id = match var {
            LuaVarExpr::NameExpr(name_expr) => {
                Some(LuaDeclId::new(cache.get_file_id(), name_expr.get_position()).into())
            }
            LuaVarExpr::IndexExpr(index_expr) => {
                Some(LuaMemberId::new(index_expr.get_syntax_id(), cache.get_file_id()).into())
            }
        };

        let explicit_var_type = var_id
            .and_then(|id| db.get_type_index().get_type_cache(&id))
            .filter(|tc| tc.is_doc())
            .map(|tc| tc.as_type().clone());

        let expr_type = infer_expr_list_value_type_at(db, cache, &assignment_info.exprs, i)?;
        let Some(expr_type) = expr_type else {
            return Ok(ResultTypeOrContinue::Continue);
        };

        let (source_type, reuse_source_narrowing) =
            if let Some(explicit) = explicit_var_type.clone() {
                (explicit, true)
            } else {
                let antecedent_flow_id = get_single_antecedent(flow_node)?;
                if !preserves_assignment_expr_type(&expr_type) {
                    (
                        get_type_at_flow_internal(
                            db,
                            tree,
                            cache,
                            root,
                            var_ref_id,
                            var_ref_cache_id,
                            antecedent_flow_id,
                            false,
                        )?,
                        false,
                    )
                } else {
                    let narrowed_source_type = get_type_at_flow_internal(
                        db,
                        tree,
                        cache,
                        root,
                        var_ref_id,
                        var_ref_cache_id,
                        antecedent_flow_id,
                        true,
                    )?;
                    if can_reuse_narrowed_assignment_source(db, &narrowed_source_type, &expr_type) {
                        (narrowed_source_type, true)
                    } else {
                        (
                            get_type_at_flow_internal(
                                db,
                                tree,
                                cache,
                                root,
                                var_ref_id,
                                var_ref_cache_id,
                                antecedent_flow_id,
                                false,
                            )?,
                            false,
                        )
                    }
                }
            };

        let narrowed = if source_type == LuaType::Nil {
            None
        } else {
            let declared =
                get_var_ref_type(db, cache, var_ref_id)
                    .ok()
                    .and_then(|decl| match decl {
                        LuaType::Def(_) | LuaType::Ref(_) => Some(decl),
                        _ => None,
                    });

            narrow_down_type(db, source_type.clone(), expr_type.clone(), declared)
        };

        let result_type = if reuse_source_narrowing || preserves_assignment_expr_type(&expr_type) {
            narrowed.unwrap_or_else(|| explicit_var_type.unwrap_or_else(|| expr_type.clone()))
        } else {
            expr_type
        };

        return Ok(ResultTypeOrContinue::Result(result_type));
    }

    Ok(ResultTypeOrContinue::Continue)
}

fn try_infer_decl_initializer_type(
    db: &DbIndex,
    cache: &mut LuaInferCache,
    root: &LuaChunk,
    var_ref_id: &VarRefId,
) -> Result<Option<LuaType>, InferFailReason> {
    let Some(decl_id) = var_ref_id.get_decl_id_ref() else {
        return Ok(None);
    };

    let decl = db
        .get_decl_index()
        .get_decl(&decl_id)
        .ok_or(InferFailReason::None)?;

    let Some(value_syntax_id) = decl.get_value_syntax_id() else {
        return Ok(None);
    };

    let Some(node) = value_syntax_id.to_node_from_root(root.syntax()) else {
        return Ok(None);
    };

    let Some(expr) = LuaExpr::cast(node) else {
        return Ok(None);
    };

    let expr_type = infer_expr(db, cache, expr.clone())?;
    let init_type = expr_type.get_result_slot_type(0);

    Ok(init_type)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CacheOptions, FileId};

    #[test]
    fn test_flow_caches_stay_sparse_for_large_flow_ids() {
        let mut cache = LuaInferCache::new(FileId::new(0), CacheOptions::default());

        *get_flow_node_cache_slot(&mut cache, 0, FlowId(10_000), true) = Some(CacheEntry::Ready);
        *get_condition_flow_cache_slot(&mut cache, 0, FlowId(20_000), false) =
            Some(CacheEntry::Ready);

        assert_eq!(cache.flow_node_cache.len(), 1);
        assert_eq!(cache.flow_node_cache[0].len(), 1);
        assert_eq!(cache.condition_flow_cache.len(), 1);
        assert_eq!(cache.condition_flow_cache[0].len(), 1);
        assert!(matches!(
            get_flow_node_cache_entry(&cache, 0, FlowId(10_000), true),
            Some(CacheEntry::Ready)
        ));
        assert!(matches!(
            get_condition_flow_cache_entry(&cache, 0, FlowId(20_000), false),
            Some(CacheEntry::Ready)
        ));
    }
}
