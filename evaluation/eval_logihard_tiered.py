#!/usr/bin/env python3
"""
LogiHard-2k Tiered Evaluation Script
Handles unified file with C (Easy/Medium/Hard/Expert) + Base splits
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import argparse
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import hashlib
import logging
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvalConfig:
    base_url: str = "XXX"
    api_key: str = ""
    model: str = "kimi-k2.5"
    max_workers: int = 5  # Conservative for 64K tokens
    temperature: float = 1
    max_tokens: int = 65536  # 64K for thinking models
    timeout: int = 600 # Increased to 10 minutes for reasoning models
    retry_attempts: int = 3
    cache_dir: str = "./eval_cache"
    output_dir: str = "./eval_results"


class OneAPIClient:
    """Async API client with caching"""
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.api_key = config.api_key or os.getenv("LOGIHARD_API_KEY", "")
        if not self.api_key:
            raise ValueError("Set LOGIHARD_API_KEY env var")
        
        self.base_url = config.base_url.rstrip('/')
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _get_cache_key(self, messages: List[Dict], subset: str) -> str:
        content = json.dumps({"m": messages, "s": subset, "model": self.config.model})
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_cached(self, key: str) -> Optional[str]:
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.load(open(cache_file)).get("content")
        return None
    
    def _save_cache(self, key: str, content: str):
        with open(self.cache_dir / f"{key}.json", 'w') as f:
            json.dump({"content": content, "t": datetime.now().isoformat()}, f)
    
    async def generate(self, messages: List[Dict], subset: str = "unknown") -> str:
        """Generate with retry"""
        cache_key = self._get_cache_key(messages, subset)
        
        if cached := self._get_cached(cache_key):
            return cached
        
        for attempt in range(self.config.retry_attempts):
            try:
                payload = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
                
                async with self.session.post(
                    f"{self.base_url}/chat/completions", json=payload
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data['choices'][0]['message']['content']
                        self._save_cache(cache_key, content)
                        return content
                    else:
                        text = await resp.text()
                        logger.warning(f"API error {resp.status}: {text[:100]}")
                        if attempt < self.config.retry_attempts - 1:
                            await asyncio.sleep(1 * (2 ** attempt))
                        
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise
        
        return ""


class TieredLogiHardEvaluator:
    """Evaluator for tiered LogiHard-2k (Easy/Medium/Hard/Expert/Base)"""
    
    def __init__(self, client: OneAPIClient, config: EvalConfig):
        self.client = client
        self.config = config
        
        # System prompt optimized for both single and multi-select
        self.system_prompt = """You are taking a multiple-choice logic test.

Instructions:
- Read the context carefully (may contain statements I, II, III, IV)
- Evaluate each option (A, B, C, D, etc.)
- Select ALL options that are correct (may be one or more)
- You may show your reasoning, but put your FINAL ANSWER as just the letter(s) on the last line
- Format: "A" or "A, B" or "B, C, D" """
        
    def is_combinatorial(self, q: Dict) -> bool:
        """Check if question is combinatorial (C) or base"""
        return q.get('is_combinatorial', False) or q.get('subset') == 'combinatorial'
    
    def get_tier(self, q: Dict) -> str:
        """Get hardness tier (Easy/Medium/Hard/Expert/Base)"""
        if not self.is_combinatorial(q):
            return "Base"
        return q.get('hardness_tier', 'Unknown')
    
    def build_prompt(self, q: Dict) -> List[Dict]:
        """Build prompt for question"""
        orig = q.get('original', {})
        context = orig.get('context', '')
        question = orig.get('question', 'Which of the following is correct?')
        options = orig.get('options', [])
        
        options_text = "\n".join(options)
        
        user_content = f"{context}\n\n{question}\n\n{options_text}\n\nSelect the correct option(s) by letter. Put your final answer (letters only) on the last line:"
        
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
    
    def parse_response(self, content: str) -> List[str]:
        """Parse answer letters from response"""
        content = content.strip().upper()
        
        # Look at last non-empty line first (for CoT models)
        lines = [l.strip() for l in content.split('\n') if l.strip()]
        for line in reversed(lines):
            # Remove prefixes
            line = re.sub(r'^(ANSWER|答案|选项|正确选项|FINAL)[\s:：,，]+', '', line)
            
            # Find all letters A-F
            matches = re.findall(r'\b([A-F])\b', line)
            if matches:
                # Remove duplicates, preserve order
                seen = set()
                result = []
                for c in matches:
                    if c not in seen:
                        seen.add(c)
                        result.append(c)
                return result
        
        # Fallback: search entire text
        matches = re.findall(r'\b([A-F])\b', content)
        seen = set()
        result = []
        for c in matches:
            if c not in seen:
                seen.add(c)
                result.append(c)
        return result
    
    def get_ground_truth(self, q: Dict) -> List[str]:
        """Get ground truth as list of letters"""
        orig = q.get('original', {})
        answer = orig.get('answer', [])
        
        if isinstance(answer, list):
            return [str(a).strip().upper() for a in answer]
        elif isinstance(answer, str):
            # Single answer (Base), convert to list
            return [answer.strip().upper()] if answer else []
        return []
    
    def compute_metrics(self, predicted: List[str], ground_truth: List[str]) -> Dict:
        """Compute exact match, precision, recall, F1"""
        pred_set = set(predicted)
        gt_set = set(ground_truth)
        
        exact = pred_set == gt_set
        
        tp = len(pred_set & gt_set)
        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(gt_set) if gt_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'exact_match': exact,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'pred_count': len(pred_set),
            'gt_count': len(gt_set)
        }
    
    async def evaluate_single(self, q: Dict, idx: int) -> Dict:
        """Evaluate single question with full metadata"""
        q_id = q.get('id', f'q_{idx}')
        tier = self.get_tier(q)
        is_c = self.is_combinatorial(q)
        
        start = time.time()
        
        try:
            messages = self.build_prompt(q)
            content = await self.client.generate(messages, tier)
            predicted = self.parse_response(content)
            gt = self.get_ground_truth(q)
            
            metrics = self.compute_metrics(predicted, gt)
            
            # Extract metadata for reporting
            orig = q.get('original', {})
            tier_config = orig.get('tier_config', {})
            
            result = {
                'id': q_id,
                'tier': tier,
                'is_combinatorial': is_c,
                'predicted': predicted,
                'ground_truth': gt,
                **metrics,
                'latency': time.time() - start,
                'full_response': content, # Saved full response for case study
                
                # Metadata for stratified analysis
                'n_options': len(orig.get('options', [])),
                'n_correct_gt': len(gt),
                'operators_used': tier_config.get('operators', []) if is_c else [],
                'gold_score': q.get('metadata', {}).get('gold_score', 0),
                'reasoning_type': q.get('metadata', {}).get('reasoning_type', 'unknown')
            }
            
            logger.info(f"[{tier:<8}] {q_id}: Pred={predicted} GT={gt} | Exact={metrics['exact_match']}, F1={metrics['f1']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error on {q_id}: {e}")
            return {
                'id': q_id,
                'tier': tier,
                'error': str(e),
                'exact_match': False,
                'f1': 0,
                'precision': 0,
                'recall': 0
            }
    
    async def evaluate_all(self, questions: List[Dict]) -> List[Dict]:
        """Evaluate all questions with concurrency"""
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def bound(q, i):
            async with semaphore:
                return await self.evaluate_single(q, i)
        
        tasks = [bound(q, i) for i, q in enumerate(questions)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Task exception: {r}")
            else:
                valid.append(r)
        
        return valid
    
    def compute_tier_stats(self, results: List[Dict]) -> Dict:
        """Compute statistics by tier"""
        tiers = defaultdict(list)
        for r in results:
            tier = r.get('tier', 'Unknown')
            tiers[tier].append(r)
        
        stats = {}
        for tier in ['Easy', 'Medium', 'Hard', 'Expert', 'Base']:
            if tier in tiers:
                tier_results = tiers[tier]
                n = len(tier_results)
                exact = sum(1 for r in tier_results if r.get('exact_match'))
                f1s = [r.get('f1', 0) for r in tier_results]
                
                stats[tier] = {
                    'count': n,
                    'exact_accuracy': exact / n if n > 0 else 0,
                    'avg_f1': sum(f1s) / n if n > 0 else 0,
                    'avg_latency': sum(r.get('latency', 0) for r in tier_results) / n if n > 0 else 0,
                    'avg_options': sum(r.get('n_options', 0) for r in tier_results) / n if n > 0 else 0
                }
        
        return stats
    
    def compute_overall(self, tier_stats: Dict) -> Dict:
        """Compute weighted overall statistics"""
        total_n = sum(s['count'] for s in tier_stats.values())
        
        if total_n == 0:
            return {}
        
        # Weight by count
        weighted_exact = sum(
            s['count'] * s['exact_accuracy'] for s in tier_stats.values()
        ) / total_n
        
        weighted_f1 = sum(
            s['count'] * s['avg_f1'] for s in tier_stats.values()
        ) / total_n
        
        # C-specific (non-Base)
        c_stats = {k: v for k, v in tier_stats.items() if k != 'Base'}
        c_n = sum(s['count'] for s in c_stats.values())
        c_exact = sum(s['count'] * s['exact_accuracy'] for s in c_stats.values()) / c_n if c_n > 0 else 0
        
        base_exact = tier_stats.get('Base', {}).get('exact_accuracy', 0)
        
        return {
            'total_questions': total_n,
            'overall_exact': weighted_exact,
            'overall_f1': weighted_f1,
            'c_exact': c_exact,
            'base_exact': base_exact,
            'combinatorial_gap': base_exact - c_exact,
            'tier_breakdown': {k: {'count': v['count'], 'accuracy': v['exact_accuracy']} 
                              for k, v in tier_stats.items()}
        }


def load_unified_dataset(filepath: str) -> List[Dict]:
    """Load the unified tiered dataset"""
    questions = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            q = json.loads(line)
            questions.append(q)
    
    # Verify structure
    tiers = defaultdict(int)
    for q in questions:
        tier = q.get('hardness_tier', 'Base' if not q.get('is_combinatorial') else 'Unknown')
        tiers[tier] += 1
    
    print(f"Loaded {len(questions)} questions:")
    for tier, count in sorted(tiers.items()):
        print(f"  {tier}: {count}")
    
    return questions


async def main():
    parser = argparse.ArgumentParser(description="Evaluate LogiHard-2k Tiered")
    parser.add_argument("--input", "-i", required=True, help="Unified JSONL file")
    parser.add_argument("--output", "-o", default="./eval_results", help="Output directory")
    parser.add_argument("--model", "-m", default="kimi-k2.5", help="Model name")
    parser.add_argument("--api-key", default="", help="API key")
    parser.add_argument("--workers", "-w", type=int, default=5, help="Concurrency")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Limit questions (0=all)")
    parser.add_argument("--temp", type=float, default=1, help="Temperature")
    parser.add_argument("--by-tier", action="store_true", help="Also save per-tier result files")
    
    args = parser.parse_args()
    
    config = EvalConfig(
        api_key=args.api_key,
        model=args.model,
        max_workers=args.workers,
        temperature=args.temp
    )
    
    # Load unified dataset
    print(f"Loading dataset: {args.input}")
    questions = load_unified_dataset(args.input)
    
    if args.limit > 0:
        questions = questions[:args.limit]
        print(f"Limited to {args.limit} questions")
    
    # Evaluate
    print(f"\nStarting evaluation with model: {args.model}")
    async with OneAPIClient(config) as client:
        evaluator = TieredLogiHardEvaluator(client, config)
        results = await evaluator.evaluate_all(questions)
    
    # Compute stats
    tier_stats = evaluator.compute_tier_stats(results)
    overall = evaluator.compute_overall(tier_stats)
    
    # Print results
    print("\n" + "="*70)
    print("LOGIHARD-2K TIERED EVALUATION RESULTS")
    print("="*70)
    print(f"Model: {args.model} | Temp: {args.temp} | Total: {overall['total_questions']}")
    print("-"*70)
    
    # Tier breakdown
    for tier in ['Easy', 'Medium', 'Hard', 'Expert', 'Base']:
        if tier in tier_stats:
            s = tier_stats[tier]
            print(f"{tier:<10}: n={s['count']:<4} | Acc={s['exact_accuracy']:.1%} | "
                  f"F1={s['avg_f1']:.2f} | AvgOpts={s['avg_options']:.1f}")
    
    print("-"*70)
    print(f"Overall:    Acc={overall['overall_exact']:.1%} | F1={overall['overall_f1']:.2f}")
    print(f"C-Only:     Acc={overall['c_exact']:.1%}")
    print(f"Base-Only:  Acc={overall['base_exact']:.1%}")
    print(f"C-Base Gap: {overall['combinatorial_gap']:+.1%}")
    print("="*70)
    
    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_clean = args.model.replace('/', '_').replace(':', '_')
    
    # Main results file
    main_file = out_dir / f"logihard_tiered_{model_clean}_{ts}.json"
    with open(main_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'model': args.model,
                'temperature': args.temp,
                'max_tokens': config.max_tokens,
                'input_file': args.input,
                'timestamp': ts
            },
            'results': results,
            'tier_statistics': tier_stats,
            'overall': overall
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved: {main_file}")
    
    # Optional: per-tier files
    if args.by_tier:
        tier_groups = defaultdict(list)
        for r in results:
            tier_groups[r['tier']].append(r)
        
        for tier, tier_results in tier_groups.items():
            tier_file = out_dir / f"logihard_tiered_{tier.lower()}_{model_clean}_{ts}.json"
            with open(tier_file, 'w') as f:
                json.dump(tier_results, f, ensure_ascii=False, indent=2)
            print(f"Saved {tier}: {tier_file}")

if __name__ == "__main__":
    asyncio.run(main())