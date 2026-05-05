#!/usr/bin/env python3
"""
LogiHard-2k Builder: Natural Distribution + Tiered Hardness
- 539 C-eligible → Stratified by hardness (Easy/Medium/Hard/Expert)
- 1461 Base → Kept as atomic controls
"""

import json
import random
import re
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import Counter, defaultdict

@dataclass
class TransformConfig:
    """Hardness tier configuration"""
    tier: str
    n_options: int
    n_correct: Tuple[int, int]
    allow_exact: bool = True
    allow_or: bool = False
    allow_not: bool = False
    allow_compound: bool = False
    require_not: bool = False
    require_or: bool = False
    
    # Weights for option selection
    weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = {
                'exact': 1.0,
                'or': 0.7 if self.allow_or else 0,
                'not_single': 0.5 if self.allow_not else 0,
                'not_multi': 0.3 if self.allow_compound else 0
            }

class TieredCombinatorialTransformer:
    def __init__(self, config: TransformConfig):
        self.config = config
        self.roman = ['I', 'II', 'III', 'IV']
        
    def clean_option_text(self, text: str) -> str:
        return re.sub(r'^[A-F][\.．、\)\s]\s*', '', text)
    
    def to_statement(self, text: str, idx: int) -> str:
        content = self.clean_option_text(text)
        #if not re.match(r'[该此这那甲乙丙丁]', content):
        #    content = f"该情况下{content}"
        return f"{self.roman[idx]}. {content}"
    
    def evaluate_not(self, not_indices: Set[int], true_set: Set[int]) -> bool:
        return len(not_indices & true_set) == 0
    
    def generate_options(self, true_set: Set[int]) -> Tuple[List[str], List[str]]:
        roman = self.roman
        true_idx = list(true_set)[0]
        true_roman = roman[true_idx]
        other_indices = [i for i in range(4) if i != true_idx]
        
        candidates = []
        
        # 1. EXACT logic (always allowed)
        if self.config.allow_exact:
            candidates.append((f"仅 {true_roman}", true_set, "exact", True))
            candidates.append((f"只有 {true_roman}", true_set, "exact", True))
            candidates.append((f"{true_roman} 为真", true_set, "exact", True))
            
            # Wrong exact (distractors)
            for other in other_indices:
                candidates.append((f"仅 {roman[other]}", {other}, "exact_wrong", False))
        
        # 2. OR logic (Medium+)
        if self.config.allow_or:
            for other in other_indices:
                indices = {true_idx, other}
                candidates.append((f"{true_roman} 或 {roman[other]}", indices, "or", True))
                # Wrong OR
                if len(other_indices) >= 2:
                    others = [o for o in other_indices if o != other][:1]
                    if others:
                        wrong_pair = {other, others[0]}
                        candidates.append((f"{roman[other]} 或 {roman[others[0]]}", wrong_pair, "or_wrong", False))
        
        # 3. NOT logic (Hard+)
        if self.config.allow_not:
            for other in other_indices:
                not_idx = {other}
                is_correct = self.evaluate_not(not_idx, true_set)
                tag = "not_single" if is_correct else "not_single_wrong"
                candidates.append((f"非 {roman[other]}", not_idx, tag, is_correct))
            
            # Multi NOT (Expert)
            if self.config.allow_compound and len(other_indices) >= 2:
                for combo in combinations(other_indices, 2):
                    not_idx = set(combo)
                    is_correct = self.evaluate_not(not_idx, true_set)
                    desc = f"非 {roman[combo[0]]} 且非 {roman[combo[1]]}"
                    tag = "not_multi" if is_correct else "not_multi_wrong"
                    candidates.append((desc, not_idx, tag, is_correct))
        
        # 4. Universal distractors
        candidates.append(("以上各项均不成立", set(), "none", False))
        
        # Deduplicate
        seen = set()
        unique = []
        for c in candidates:
            if c[0] not in seen:
                unique.append(c)
                seen.add(c[0])
        
        # Separate
        correct_all = [c for c in unique if c[3]]
        wrong_all = [c for c in unique if not c[3]]
        
        # Determine target correct count
        min_c, max_c = self.config.n_correct
        target_correct = random.randint(min_c, max_c)
        target_correct = min(target_correct, len(correct_all))
        
        selected = []
        
        # Enforce requirements
        if self.config.require_not:
            not_correct = [c for c in correct_all if 'not' in c[2]]
            if not_correct:
                selected.append(random.choice(not_correct))
        
        if self.config.require_or:
            or_correct = [c for c in correct_all if c[2] == 'or']
            if or_correct and len(selected) < target_correct:
                selected.append(random.choice(or_correct))
        
        # Fill remaining with weighted sampling
        remaining = [c for c in correct_all if c not in selected]
        if remaining and len(selected) < target_correct:
            weights = [self.config.weights.get(c[2].replace('_wrong', ''), 0.5) for c in remaining]
            needed = target_correct - len(selected)
            chosen = random.choices(remaining, weights=weights, k=min(needed, len(remaining)))
            # Deduplicate
            seen_sel = {s[0] for s in selected}
            for c in chosen:
                if c[0] not in seen_sel:
                    selected.append(c)
                    seen_sel.add(c[0])
        
        # Fill wrong options
        needed_wrong = self.config.n_options - len(selected)
        if len(wrong_all) >= needed_wrong:
            selected.extend(random.sample(wrong_all, needed_wrong))
        else:
            selected.extend(wrong_all)
        
        random.shuffle(selected)
        
        # Build final
        final_options = []
        correct_letters = []
        
        for i, (desc, indices, logic_type, is_correct) in enumerate(selected[:self.config.n_options]):
            letter = chr(ord('A') + i)
            final_options.append(f"{letter}. {desc}")
            if is_correct:
                correct_letters.append(letter)
        
        return final_options, sorted(correct_letters)
    
    def transform(self, item: Dict) -> Optional[Dict]:
        orig = item.get('original', item)
        old_options = orig.get('options', [])
        
        # Filter: exactly 4 options, no E
        if len(old_options) != 4:
            return None
        if any(re.match(r'^E[\.．]', opt) for opt in old_options):
            return None
        
        # Parse answer
        old_answer = orig.get('answer', 'A')
        if isinstance(old_answer, list):
            old_answer = old_answer[0] if old_answer else 'A'
        old_answer = str(old_answer).strip().upper()
        if '.' in old_answer:
            old_answer = old_answer.split('.')[0]
        
        correct_idx = ord(old_answer) - ord('A')
        if correct_idx < 0 or correct_idx >= 4:
            correct_idx = 0
        
        # Build
        statements = [self.to_statement(opt, i) for i, opt in enumerate(old_options)]
        true_set = {correct_idx}
        new_options, correct_letters = self.generate_options(true_set)
        
        # Context
        parts = []
        if orig.get('context', '').strip():
            parts.append(orig.get('context').strip())
        if orig.get('question', '').strip():
            parts.append(orig.get('question').strip())
        base_text = '\n'.join(parts) if parts else ""
        full_context = base_text + ('\n' if base_text else '') + '\n'.join(statements)
        
        # Operators used
        ops_used = []
        for letter in correct_letters:
            idx = ord(letter) - ord('A')
            if idx < len(new_options):
                text = new_options[idx]
                if '非' in text:
                    ops_used.append('NOT')
                elif '或' in text:
                    ops_used.append('OR')
                elif '仅' in text:
                    ops_used.append('EXACT')
        
        # Metadata - 包含IRT参数所需的完整metrics
        base_metrics = item.get('metrics', {})
        metadata = {
            'tier': self.config.tier,
            'gold_score': base_metrics.get('gold_score', 72.0),  # 默认值与IRT引擎一致
            'reasoning_type': orig.get('reasoning_type', 'unknown'),
            'source': item.get('source', 'unknown'),
            'original_id': item.get('id', ''),
            'full_metrics': {
                # 基础指标
                'gold_score': base_metrics.get('gold_score', 72.0),
                'difficulty': base_metrics.get('difficulty', 'medium'),
                
                # IRT参数相关指标
                'logic_density': base_metrics.get('logic_density', 2.0),
                'abstraction_level': base_metrics.get('abstraction_level', 3.0),
                'thinking_length': base_metrics.get('thinking_length', 1500),
                'reasoning_segments': base_metrics.get('reasoning_segments', 100),
                'coherence_depth': base_metrics.get('coherence_depth', 0.15),
                'dialectic_quality': base_metrics.get('dialectic_quality', 1.0),
                'pivot_count': base_metrics.get('pivot_count', 30),
                'fallacy_risk': base_metrics.get('fallacy_risk', 0.5),
                'option_coverage': base_metrics.get('option_coverage', 0.5),
                
                # 保留原始metrics中的所有其他字段
                **{k: v for k, v in base_metrics.items() 
                   if k not in ['gold_score', 'difficulty', 'logic_density', 
                               'abstraction_level', 'thinking_length', 'reasoning_segments',
                               'coherence_depth', 'dialectic_quality', 'pivot_count',
                               'fallacy_risk', 'option_coverage']}
            }
        }
        
        return {
            "id": item.get("id", ""),
            "subset": "combinatorial",
            "is_combinatorial": True,
            "hardness_tier": self.config.tier,
            "original": {
                "context": full_context,
                "question": "以下哪项是正确的？",
                "options": new_options,
                "answer": correct_letters,
                "true_statement": self.roman[correct_idx],
                "reasoning_type": orig.get("reasoning_type", "deductive"),
                "task_format": "combinatorial_multi_select",
                "tier_config": {
                    "n_options": self.config.n_options,
                    "n_correct": len(correct_letters),
                    "operators": list(set(ops_used)),
                    "tier": self.config.tier
                }
            },
            "metadata": metadata
        }

def process_natural_with_tiers(input_path: str, output_path: str, seed: int = 42):
    """
    Natural 27/73 split with tiered hardness within C subset
    """
    random.seed(seed)
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load
    print(f"Loading from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(questions)} total questions")
    
    # Separate C-eligible vs Base
    c_eligible = []
    base_only = []
    
    for q in questions:
        orig = q.get('original', q)
        opts = orig.get('options', [])
        
        if len(opts) == 4 and not any(re.match(r'^E[\.．]', o) for o in opts):
            c_eligible.append(q)
        else:
            base_only.append(q)
    
    print(f"\nNatural eligibility:")
    print(f"  C-eligible (4 options): {len(c_eligible)}")
    print(f"  Base-only: {len(base_only)}")
    
    # Tier configurations
    tiers = {
        'Easy': TransformConfig(
            tier='Easy',
            n_options=4,
            n_correct=(1, 1),
            allow_exact=True,
            allow_or=False,
            allow_not=False
        ),
        'Medium': TransformConfig(
            tier='Medium',
            n_options=5,
            n_correct=(1, 2),
            allow_exact=True,
            allow_or=True,
            allow_not=False,
            weights={'exact': 0.6, 'or': 0.4}
        ),
        'Hard': TransformConfig(
            tier='Hard',
            n_options=5,
            n_correct=(2, 3),
            allow_exact=True,
            allow_or=True,
            allow_not=True,
            require_not=True,  # Force at least one NOT
            weights={'exact': 0.3, 'or': 0.3, 'not_single': 0.4}
        ),
        'Expert': TransformConfig(
            tier='Expert',
            n_options=6,
            n_correct=(2, 4),
            allow_exact=True,
            allow_or=True,
            allow_not=True,
            allow_compound=True,
            require_not=True,
            require_or=True,
            weights={'exact': 0.2, 'or': 0.3, 'not_single': 0.2, 'not_multi': 0.3}
        )
    }
    
    # Calculate target counts based on 539 total C
    total_c = len(c_eligible)
    targets = {
        'Easy': int(total_c * 0.20),
        'Medium': int(total_c * 0.40),
        'Hard': int(total_c * 0.30),
        'Expert': total_c - int(total_c * 0.20) - int(total_c * 0.40) - int(total_c * 0.30)  # Remainder
    }
    
    print(f"\n{'='*60}")
    print(f"TIERED DISTRIBUTION TARGETS (within {total_c} C questions)")
    print(f"{'='*60}")
    for tier, count in targets.items():
        print(f"{tier:<10}: {count:>3} ({count/total_c:.1%})")
    print(f"{'='*60}")
    
    # Shuffle C pool for random assignment
    random.shuffle(c_eligible)
    
    # Transform by tier
    c_results = []
    current_idx = 0
    
    for tier_name, target_count in targets.items():
        transformer = TieredCombinatorialTransformer(tiers[tier_name])
        tier_questions = c_eligible[current_idx:current_idx + target_count]
        current_idx += target_count
        
        print(f"\nProcessing {tier_name} ({len(tier_questions)} questions)...")
        
        for q in tier_questions:
            result = transformer.transform(q)
            if result:
                c_results.append(result)
    
    # Process remaining C-eligible (if any) as Medium
    remaining_c = c_eligible[current_idx:]
    if remaining_c:
        print(f"\nProcessing {len(remaining_c)} remaining as Medium...")
        transformer = TieredCombinatorialTransformer(tiers['Medium'])
        for q in remaining_c:
            result = transformer.transform(q)
            if result:
                c_results.append(result)
    
    # Process Base
    print(f"\nProcessing {len(base_only)} Base questions...")
    base_results = []
    for q in base_only:
        orig = q.get('original', q)
        answer = orig.get('answer', 'A')
        if isinstance(answer, list):
            answer = answer[0] if answer else 'A'
        
        base_results.append({
            "id": q.get("id", ""),
            "subset": "base",
            "is_combinatorial": False,
            "hardness_tier": "base",
            "original": {
                "context": orig.get('context', ''),
                "question": orig.get('question', ''),
                "options": orig.get('options', []),
                "answer": answer,
                "reasoning_type": orig.get('reasoning_type', 'unknown'),
                "task_format": orig.get('task_format', 'atomic')
            },
            "metadata": {
                'tier': 'base',
                'gold_score': q.get('metrics', {}).get('gold_score', 72.0),
                'reasoning_type': orig.get('reasoning_type', 'unknown'),
                'source': q.get('source', 'unknown'),
                'full_metrics': {
                    # 基础指标
                    'gold_score': q.get('metrics', {}).get('gold_score', 72.0),
                    'difficulty': q.get('metrics', {}).get('difficulty', 'medium'),
                    
                    # IRT参数相关指标 - Base问题使用保守估计
                    'logic_density': 1.5,  # Base问题逻辑密度较低
                    'abstraction_level': 2.0,
                    'thinking_length': 800,
                    'reasoning_segments': 50,
                    'coherence_depth': 0.1,
                    'dialectic_quality': 0.8,
                    'pivot_count': 20,
                    'fallacy_risk': 0.3,
                    'option_coverage': 0.4,
                    
                    # 保留原始metrics中的所有其他字段
                    **{k: v for k, v in q.get('metrics', {}).items() 
                       if k not in ['gold_score', 'difficulty', 'logic_density', 
                                   'abstraction_level', 'thinking_length', 'reasoning_segments',
                                   'coherence_depth', 'dialectic_quality', 'pivot_count',
                                   'fallacy_risk', 'option_coverage']}
                }
            }
        })
    
    # Statistics
    total = len(c_results) + len(base_results)
    print(f"\n{'='*60}")
    print(f"FINAL DISTRIBUTION")
    print(f"{'='*60}")
    
    tier_dist = Counter([q['hardness_tier'] for q in c_results])
    for tier in ['Easy', 'Medium', 'Hard', 'Expert']:
        count = tier_dist.get(tier, 0)
        print(f"C-{tier:<10}: {count:>3} ({count/total:.1%})")
    
    print(f"C-Total    : {len(c_results):>3} ({len(c_results)/total:.1%})")
    print(f"Base       : {len(base_results):>3} ({len(base_results)/total:.1%})")
    print(f"{'='*60}")
    
    # Detailed C stats
    print(f"\nC subset details:")
    n_correct_dist = Counter([q['original']['tier_config']['n_correct'] for q in c_results])
    print(f"  N-correct distribution: {dict(n_correct_dist)}")
    
    ops_dist = Counter([tuple(sorted(q['original']['tier_config']['operators'])) for q in c_results])
    print(f"  Operators used: {dict(ops_dist)}")
    
    # Write unified
    print(f"\nWriting unified file: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for q in c_results:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
        for q in base_results:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    
    # Write split files
    c_path = output_path.parent / f"{output_path.stem}_c.jsonl"
    base_path = output_path.parent / f"{output_path.stem}_base.jsonl"
    
    with open(c_path, 'w', encoding='utf-8') as f:
        for q in c_results:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    
    with open(base_path, 'w', encoding='utf-8') as f:
        for q in base_results:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    
    print(f"Split files written:")
    print(f"  C: {c_path}")
    print(f"  Base: {base_path}")
    print(f"\nDone!")

def process_mmlu_for_adaptive(input_path: str, output_path: str, seed: int = 42):
    """
    专门处理MMLU数据，使其适配eval_logihard_adaptive.py
    确保包含所有IRT参数所需的字段
    """
    random.seed(seed)
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 加载MMLU数据
    print(f"Loading MMLU data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        questions = [json.loads(line) for line in f if line.strip()]
    
    print(f"Loaded {len(questions)} MMLU questions")
    
    # 为MMLU数据添加必要的metrics字段
    enhanced_questions = []
    for q in questions:
        # 确保每个问题都有完整的metrics
        orig = q.get('original', q)
        metrics = q.get('metrics', {})
        
        # 根据问题类型和难度计算IRT相关指标
        subject = q.get('subject', 'general')
        difficulty = q.get('difficulty', 'medium')
        
        # 基于学科和难度设置IRT参数
        irt_params = {
            'gold_score': metrics.get('gold_score', 
                45 if difficulty == 'easy' else 
                65 if difficulty == 'medium' else 
                85 if difficulty == 'hard' else 72),
            'logic_density': 
                1.2 if subject in ['high_school_mathematics', 'college_mathematics'] else
                2.5 if subject in ['formal_logic', 'philosophy'] else
                2.0 if subject in ['professional_law', 'moral_scenarios'] else
                1.8,
            'abstraction_level': 
                4.0 if subject in ['philosophy', 'formal_logic'] else
                3.0 if subject in ['professional_law', 'moral_scenarios'] else
                2.5,
            'thinking_length': 
                2000 if subject in ['philosophy', 'formal_logic'] else
                1500 if subject in ['professional_law', 'moral_scenarios'] else
                1000,
            'reasoning_segments': 
                120 if subject in ['philosophy', 'formal_logic'] else
                90 if subject in ['professional_law', 'moral_scenarios'] else
                60,
            'coherence_depth': 
                0.18 if subject in ['philosophy', 'formal_logic'] else
                0.15 if subject in ['professional_law', 'moral_scenarios'] else
                0.12,
            'dialectic_quality': 
                1.2 if subject in ['philosophy', 'formal_logic'] else
                1.0 if subject in ['professional_law', 'moral_scenarios'] else
                0.8,
            'pivot_count': 
                35 if subject in ['philosophy', 'formal_logic'] else
                30 if subject in ['professional_law', 'moral_scenarios'] else
                25,
            'fallacy_risk': 
                0.4 if subject in ['logical_fallacies', 'philosophy'] else
                0.3 if subject in ['professional_law', 'moral_scenarios'] else
                0.2,
            'option_coverage': 0.6
        }
        
        # 合并原始metrics
        enhanced_metrics = {**irt_params, **metrics}
        
        enhanced_q = {
            **q,
            'metrics': enhanced_metrics,
            'source': q.get('source', 'mmlu'),
            'reasoning_type': q.get('reasoning_type', 'deductive')
        }
        enhanced_questions.append(enhanced_q)
    
    # 使用标准tiered处理流程
    process_natural_with_tiers(
        input_path=str(output_path.parent / "mmlu_enhanced_temp.jsonl"),
        output_path=str(output_path),
        seed=seed
    )
    
    # 创建临时文件用于处理
    temp_path = output_path.parent / "mmlu_enhanced_temp.jsonl"
    with open(temp_path, 'w', encoding='utf-8') as f:
        for item in enhanced_questions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 处理增强后的数据
    process_natural_with_tiers(str(temp_path), str(output_path), seed)
    
    # 清理临时文件
    temp_path.unlink()
    
    print(f"✅ MMLU数据已适配eval_logihard_adaptive.py: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build tiered LogiHard-2k")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL")
    parser.add_argument("--output", "-o", default="logihard_2k_tiered.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mmlu-mode", action="store_true", 
                       help="Process MMLU data for adaptive evaluation")
    args = parser.parse_args()
    
    if args.mmlu_mode:
        process_mmlu_for_adaptive(args.input, args.output, args.seed)
    else:
        process_natural_with_tiers(args.input, args.output, args.seed)
