#!/usr/bin/env python3
"""
LogiHard-2k Adaptive Evaluation Script (IRT-CAT)
Based on Item Response Theory (IRT) and Computerized Adaptive Testing (CAT)
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import argparse
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import logging
from collections import defaultdict

# Reuse classes from tiered evaluation
from eval_logihard_tiered import EvalConfig, OneAPIClient, TieredLogiHardEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IRTEngine:
    """IRT 3PL Model and CAT Logic"""
    
    def __init__(self):
        # Normal prior for EAP: N(0, 1)
        self.prior_mu = 0.0
        self.prior_sigma = 1.0
        
        # Quadrature points for EAP integration
        self.nodes = []
        self.weights = []
        self._setup_quadrature()

    def _setup_quadrature(self, n=61, range_val=6.0):
        """Setup standard normal quadrature points"""
        step = (2 * range_val) / (n - 1)
        for i in range(n):
            x = -range_val + i * step
            # Standard normal density
            w = math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
            self.nodes.append(x)
            self.weights.append(w * step)
        
        # Normalize weights to sum to 1
        total_w = sum(self.weights)
        self.weights = [w / total_w for w in self.weights]

    def p_correct(self, theta: float, a: float, b: float, c: float) -> float:
        """3PL IRT Model: P(correct | theta) = c + (1-c) / (1 + exp(-a(theta - b)))"""
        try:
            exp_val = math.exp(-a * (theta - b))
            return c + (1.0 - c) / (1.0 + exp_val)
        except OverflowError:
            return c if a * (theta - b) < 0 else 1.0

    def fisher_information(self, theta: float, a: float, b: float, c: float) -> float:
        """Fisher Information for 3PL: I(theta) = a^2 * (P-c)^2 / P * (1-P)/(1-c)^2"""
        p = self.p_correct(theta, a, b, c)
        if p <= 0 or p >= 1 or c >= 1:
            return 0.0
        num = (a**2) * ((p - c)**2) * (1.0 - p)
        den = p * ((1.0 - c)**2)
        return num / den

    def get_item_params(self, q: Dict) -> Tuple[float, float, float]:
        """Extract IRT parameters (a, b, c) from item metadata using cognitive metrics"""
        metadata = q.get('metadata', {})
        metrics = metadata.get('full_metrics', {})
        
        # 1. Difficulty (b) - Cognitive Load Centric
        gold_score = metrics.get('gold_score', metadata.get('gold_score', 72.0))
        logic_density = metrics.get('logic_density', 2.0)
        abstraction = metrics.get('abstraction_level', 3.0)
        thinking_len = metrics.get('thinking_length', 1500)
        segments = metrics.get('reasoning_segments', 100)
        
        # Normalization (Base Mean: 72, Stdev: 54)
        b_raw = (gold_score - 72.0) / 54.0
        
        # Adjustments based on 'thinking cost' and 'structural depth'
        # Higher thinking_length and reasoning_segments strongly push difficulty up
        think_boost = (math.log10(max(1, thinking_len)) - 3.17) * 0.8 # log10(1500) approx 3.17
        seg_boost = (segments - 100) / 200.0
        
        b_adj = b_raw + 0.1 * (logic_density - 2.0) + think_boost + seg_boost
        b = max(-4.0, min(4.0, b_adj))
        
        # 2. Discrimination (a) - Logical Quality Centric
        coherence = metrics.get('coherence_depth', 0.15)
        dialectic = metrics.get('dialectic_quality', 1.0)
        pivots = metrics.get('pivot_count', 30)
        fallacy = metrics.get('fallacy_risk', 0.5)
        
        a_base = 1.0
        orig = q.get('original', {})
        is_c = q.get('is_combinatorial') or q.get('subset') == 'combinatorial'
        if is_c:
            a_base = 1.4 # Combinatorial items are naturally more discriminative
        
        # Quality scaling: High coherence and controlled fallacy risk increase discrimination
        a_quality = (coherence / 0.15) * 0.5 + (dialectic / 1.0) * 0.3 + (fallacy / 0.5) * 0.2
        a_struct = 1.0 + (pivots - 30) / 100.0
        
        a = a_base * a_quality * a_struct
        a = max(0.5, min(3.5, a))
        
        # 3. Guessing (c) - Distractor Quality Centric
        options = orig.get('options', [])
        n_opts = len(options) if options else 4
        c_base = 1.0 / n_opts if n_opts > 0 else 0.25
        
        opt_cov = metrics.get('option_coverage', 0.5)
        # Higher coverage reduces lucky guessing
        c = c_base * (1.0 - 0.3 * opt_cov)
        
        return a, b, c

    def estimate_theta_eap(self, items_params: List[Tuple[float, float, float]], responses: List[int]) -> Tuple[float, float]:
        """Expected A Posteriori (EAP) estimate of theta and its standard error"""
        likelihoods = [1.0] * len(self.nodes)
        
        for i in range(len(self.nodes)):
            theta_node = self.nodes[i]
            for (a, b, c), resp in zip(items_params, responses):
                p = self.p_correct(theta_node, a, b, c)
                prob = p if resp == 1 else (1.0 - p)
                likelihoods[i] *= prob
        
        # Posterior = Likelihood * Prior
        posteriors = [l * w for l, w in zip(likelihoods, self.weights)]
        total_post = sum(posteriors)
        
        if total_post == 0:
            return 0.0, 1.0 # Fallback
            
        # Mean (EAP)
        expected_theta = sum(theta * p for theta, p in zip(self.nodes, posteriors)) / total_post
        
        # Variance
        expected_theta_sq = sum((theta**2) * p for theta, p in zip(self.nodes, posteriors)) / total_post
        variance = expected_theta_sq - (expected_theta**2)
        
        return expected_theta, math.sqrt(max(0.0, variance))

class AdaptiveLogiHardEvaluator(TieredLogiHardEvaluator):
    """CAT Evaluator using IRT-based item selection and ability estimation"""
    
    def __init__(self, client: OneAPIClient, config: EvalConfig, irt_engine: IRTEngine):
        super().__init__(client, config)
        self.irt = irt_engine
        
    async def evaluate_adaptive(self, item_bank: List[Dict], max_items: int = 20, se_threshold: float = 0.3, hard_mode: bool = False, checkpoint_callback=None, initial_history=None) -> Dict:
        """Run adaptive testing loop for a single 'session' (one model)"""
        # Initialize or Resume
        if initial_history:
            history = initial_history
            t = len(history)
            administered_indices = set()
            responses = []
            params_history = []
            
            # Reconstruct state from history
            for res in history:
                # Find index in item_bank by ID
                for idx, item in enumerate(item_bank):
                    if item.get('id') == res.get('id'):
                        administered_indices.add(idx)
                        break
                
                # Extract params and response
                # We need to re-extract params because they might not be in history
                item_found = False
                for item in item_bank:
                    if item.get('id') == res.get('id'):
                        a, b, c = self.irt.get_item_params(item)
                        params_history.append((a, b, c))
                        responses.append(1 if res.get('exact_match') else 0)
                        item_found = True
                        break
                if not item_found:
                    logger.warning(f"Resumed item {res.get('id')} not found in current item bank!")

            # Current estimate
            theta_hat, se_hat = self.irt.estimate_theta_eap(params_history, responses)
            logger.info(f"Resuming CAT from step {t} | Current Theta: {theta_hat:.3f} | SE: {se_hat:.3f}")
        else:
            # Standard initialization
            theta_hat = 1.5 if hard_mode else 0.0
            se_hat = 1.0
            administered_indices = set()
            history = []
            responses = []
            params_history = []
            t = 0

        error_count = 0
        max_errors = 10 
        
        # In hard mode, we filter the bank to only high-tier items if requested
        if hard_mode:
            filtered_bank = [
                (i, q) for i, q in enumerate(item_bank) 
                if q.get('hardness_tier') in ['Hard', 'Expert']
            ]
            logger.info(f"Hard Mode enabled: Filtered bank to {len(filtered_bank)} Hard/Expert items.")
        else:
            filtered_bank = list(enumerate(item_bank))

        logger.info(f"Starting Adaptive Evaluation (CAT) | Model: {self.config.model} | Hard Mode: {hard_mode}")
        
        while t < max_items and error_count < max_errors:
            # 1. Select next item
            best_idx = -1
            max_info = -1.0
            
            for i, q in filtered_bank:
                if i in administered_indices:
                    continue
                
                a, b, c = self.irt.get_item_params(q)
                
                # In hard mode, we target a difficulty slightly ABOVE current estimate
                target_theta = theta_hat + 0.5 if hard_mode else theta_hat
                info = self.irt.fisher_information(target_theta, a, b, c)
                
                if info > max_info:
                    max_info = info
                    best_idx = i
            
            if best_idx == -1:
                # If we run out of Hard/Expert items, fallback to full bank
                if hard_mode:
                    logger.warning("Ran out of Hard/Expert items, falling back to full bank.")
                    filtered_bank = list(enumerate(item_bank))
                    continue
                break
                
            administered_indices.add(best_idx)
            item = item_bank[best_idx]
            a, b, c = self.irt.get_item_params(item)
            
            # 2. Administer item
            logger.info(f"Step {t+1} (Attempt {t+1+error_count}): Administering item {item.get('id')} (b={b:.2f}, info={max_info:.3f})")
            result = await self.evaluate_single(item, best_idx)
            
            # 3. Handle Technical Errors
            if 'error' in result:
                logger.warning(f"Technical error on {item.get('id')}: {result['error']}. Skipping item and retrying with another.")
                error_count += 1
                continue
            
            # 4. Update history and theta (Only for valid responses)
            resp = 1 if result.get('exact_match') else 0
            responses.append(resp)
            params_history.append((a, b, c))
            
            theta_hat, se_hat = self.irt.estimate_theta_eap(params_history, responses)
            
            result['step'] = t + 1
            result['theta_estimate'] = theta_hat
            result['se_estimate'] = se_hat
            history.append(result)
            
            logger.info(f"Step {t+1} Result: {resp} | New Theta: {theta_hat:.3f} | SE: {se_hat:.3f}")
            
            # Checkpoint callback
            if checkpoint_callback:
                checkpoint_callback(history, theta_hat, se_hat)

            # 5. Check stopping criteria
            t += 1
            if se_hat < se_threshold:
                logger.info(f"Stopping: SE {se_hat:.3f} < threshold {se_threshold}")
                break
                
        return {
            'final_theta': theta_hat,
            'final_se': se_hat,
            'num_items': len(history),
            'total_attempts': t + error_count,
            'error_count': error_count,
            'history': history
        }

async def main():
    parser = argparse.ArgumentParser(description="Adaptive Evaluation (IRT-CAT) for LogiHard-2k")
    parser.add_argument("--input", "-i", required=True, help="Unified JSONL file")
    parser.add_argument("--output", "-o", default="./eval_results_adaptive", help="Output directory")
    parser.add_argument("--model", "-m", default="kimi-k2.5", help="Model name")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--max-items", type=int, default=60, help="Max items per session")
    parser.add_argument("--base-se-threshold", type=float, default=0.3, help="Stopping SE threshold for Base subset")
    parser.add_argument("--comb-se-threshold", type=float, default=0.35, help="Stopping SE threshold for Combinatorial subset")
    parser.add_argument("--temp", type=float, default=1, help="Temperature")
    parser.add_argument("--hard-mode", action="store_true", help="Stress test mode: focus on Hard/Expert items")
    parser.add_argument("--timeout", type=int, default=600, help="API timeout in seconds")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temp,
        timeout=args.timeout,
        max_workers=1 
    )
    
    # Load item bank
    print(f"Loading item bank: {args.input}")
    from eval_logihard_tiered import load_unified_dataset
    item_bank = load_unified_dataset(args.input)
    
    # Checkpoint setup
    model_clean = args.model.replace('/', '_').replace(':', '_')
    checkpoint_dir = Path(args.output) / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_file = checkpoint_dir / f"cat_dual_{model_clean}.json"
    
    checkpoint_data = {}
    if args.resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            print(f"Loaded checkpoint from {checkpoint_file}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    # Split item bank into Base and Combinatorial
    base_bank = [q for q in item_bank if not (q.get('is_combinatorial') or q.get('subset') == 'combinatorial')]
    c_bank = [q for q in item_bank if (q.get('is_combinatorial') or q.get('subset') == 'combinatorial')]
    
    print(f"Loaded {len(item_bank)} questions (Base: {len(base_bank)}, Combinatorial: {len(c_bank)})")
    
    irt_engine = IRTEngine()
    
    async with OneAPIClient(config) as client:
        evaluator = AdaptiveLogiHardEvaluator(client, config, irt_engine)
        
        # Session 1: Base
        base_init_history = checkpoint_data.get('base_session', {}).get('history')
        if checkpoint_data.get('base_session', {}).get('final_se', 1.0) < args.base_se_threshold and not args.hard_mode:
            print(f"\n>>> Skipping CAT Session 1: BASE (Already completed)")
            base_cat = checkpoint_data['base_session']
        else:
            print(f"\n>>> Running CAT Session 1: BASE ({len(base_bank)} items available)")
            def save_base_checkpoint(h, t, s):
                checkpoint_data['base_session'] = {'history': h, 'final_theta': t, 'final_se': s, 'num_items': len(h), 'total_attempts': len(h)}
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

            base_cat = await evaluator.evaluate_adaptive(
                base_bank, 
                max_items=args.max_items, 
                se_threshold=args.base_se_threshold,
                hard_mode=False,
                initial_history=base_init_history,
                checkpoint_callback=save_base_checkpoint
            )
            checkpoint_data['base_session'] = base_cat
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        
        # Session 2: Combinatorial
        c_init_history = checkpoint_data.get('combinatorial_session', {}).get('history')
        if checkpoint_data.get('combinatorial_session', {}).get('final_se', 1.0) < args.comb_se_threshold:
             print(f"\n>>> Skipping CAT Session 2: COMBINATORIAL (Already completed)")
             c_cat = checkpoint_data['combinatorial_session']
        else:
            print(f"\n>>> Running CAT Session 2: COMBINATORIAL ({len(c_bank)} items available)")
            def save_c_checkpoint(h, t, s):
                checkpoint_data['combinatorial_session'] = {'history': h, 'final_theta': t, 'final_se': s, 'num_items': len(h), 'total_attempts': len(h)}
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

            c_cat = await evaluator.evaluate_adaptive(
                c_bank, 
                max_items=args.max_items, 
                se_threshold=args.comb_se_threshold,
                hard_mode=args.hard_mode,
                initial_history=c_init_history,
                checkpoint_callback=save_c_checkpoint
            )
            checkpoint_data['combinatorial_session'] = c_cat
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
    
    # Compute stats for display
    def get_subset_metrics(res_list):
        if not res_list: return 0.0, 0.0, 0
        acc = sum(1 for r in res_list if r.get('exact_match')) / len(res_list)
        f1 = sum(r.get('f1', 0) for r in res_list) / len(res_list)
        correct_count = sum(1 for r in res_list if r.get('exact_match'))
        return acc, f1, correct_count

    base_acc, base_f1, base_corr = get_subset_metrics(base_cat['history'])
    c_acc, c_f1, c_corr = get_subset_metrics(c_cat['history'])
    
    # Print summary
    print("\n" + "="*70)
    print("DUAL-SUBSET ADAPTIVE EVALUATION (IRT-CAT) RESULTS")
    print("="*70)
    print(f"Model: {args.model} | Hard Mode: {args.hard_mode}")
    print("-" * 70)
    print(f"1. BASE SUBSET:")
    print(f"   Ability (Theta): {base_cat['final_theta']:.3f} | SE: {base_cat['final_se']:.3f}")
    print(f"   Items:           {base_cat['num_items']} (Attempts: {base_cat['total_attempts']})")
    print(f"   Exact Match:     {base_corr}/{base_cat['num_items']} ({base_acc:.2%})")
    print(f"   Average F1:      {base_f1:.4f}")
    print("-" * 70)
    print(f"2. COMBINATORIAL SUBSET:")
    print(f"   Ability (Theta): {c_cat['final_theta']:.3f} | SE: {c_cat['final_se']:.3f}")
    print(f"   Items:           {c_cat['num_items']} (Attempts: {c_cat['total_attempts']})")
    print(f"   Exact Match:     {c_corr}/{c_cat['num_items']} ({c_acc:.2%})")
    print(f"   Average F1:      {c_f1:.4f}")
    print("-" * 70)
    print(f"CAP ANALYSIS (High-Resolution):")
    print(f"   Theta Drop:      {base_cat['final_theta'] - c_cat['final_theta']:.4f}")
    print(f"   F1 Score Gap:    {base_f1 - c_f1:+.4f}")
    print(f"   Exact Acc Gap:   {base_acc - c_acc:+.2%}")
    print("="*70)
    
    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = args.model.replace('/', '_').replace(':', '_')
    
    output_file = out_dir / f"cat_dual_{model_clean}_{ts}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'base_session': base_cat,
            'combinatorial_session': c_cat,
            'summary': {
                'model': args.model,
                'hard_mode': args.hard_mode,
                'base_theta': base_cat['final_theta'],
                'c_theta': c_cat['final_theta'],
                'theta_drop': base_cat['final_theta'] - c_cat['final_theta'],
                'accuracy_gap': base_acc - c_acc
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved detailed dual-log to: {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
