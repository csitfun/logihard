import json
import time
import hashlib
import os
import re
import concurrent.futures
import random
import signal
import sys
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from openai import OpenAI, APIError, APITimeoutError, APIConnectionError

# ==================== 实验配置 ====================

class Config:
    API_KEY = "sk-E40NAXM7qJ8uSKr4kH3UmsTL2aHeutZcaBaeijwZcjtWtYNZ"
    BASE_URL = "https://api.moonshot.cn/v1"
    MODEL_NAME = "kimi-k2.5"
    
    INPUT_FILE = "seed_data.jsonl"
    LOGIHARD_FULL = "logihard_full_sorted.json"
    CORPUS_FULL = "longcot_corpus_full.json"
    LOGIHARD_TOP2K = "logihard_top2k.json"
    CORPUS_TOP2K = "longcot_corpus_top2k.json"
    CHECKPOINT_FILE = "experiment_checkpoint.json"
    
    MAX_WORKERS = 8
    TOP_K_THRESHOLD = 2000
    MIN_GOLD_SCORE = 15.0
    MAX_TOKENS = 16000
    PROMPT_MODE = "simple"
    
    # 重试配置
    MAX_RETRIES = 3           # 单个请求最大重试次数
    RETRY_DELAY = 2           # 初始重试延迟（秒）
    SAVE_INTERVAL = 10        # 每处理N个保存一次（比原来的20更频繁）

client = OpenAI(api_key=Config.API_KEY, base_url=Config.BASE_URL)

# ==================== 工具函数 ====================

def load_json_or_jsonl(file_path: str) -> List[Dict]:
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if data:
            print(f"成功加载 {len(data)} 条数据")
            return data
    except Exception as e:
        print(f"加载失败: {e}")
        raise

def save_jsonl(file_path: str, data: List[Dict]):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

# ==================== 提示词引擎 ====================

class PromptEngine:
    @staticmethod
    def build(item: Dict, mode: str = Config.PROMPT_MODE) -> str:
        base = f"Context: {item['context']}\nQuestion: {item['question']}\nOptions: {', '.join(item['options'])}"
        
        if mode == "simple":
            return f"""请解答以下逻辑推理题，并详细展示你的思考过程。

{base}

要求：
1. 逐步分析，说明每一步的推理依据
2. 对每个选项进行评估（正确或排除理由）
3. 最后给出确定的答案

请开始："""
        elif mode == "strict":
            return f"""请解决以下逻辑推理问题。在给出最终答案前，请展示完整思考过程：

{base}

请开始你的详细推理："""
        else:
            return f"{base}\n\n请详细推理后给出答案："

# ==================== 逻辑评估器 ====================

@dataclass
class LogicalMetrics:
    gold_score: float
    coherence_depth: float
    cognitive_entropy: float
    abductive_strength: float
    dialectic_quality: float
    abstraction_level: int
    pivot_count: int
    logic_density: float
    dim_awareness: float
    option_coverage: float
    fallacy_risk: float
    thinking_length: int
    reasoning_segments: int
    is_high_quality: bool

class LogiHardEvaluator:
    def __init__(self):
        self.pivot_patterns = {
            'strong_rejection': [r'(?:不对|错误|并非如此|恰恰相反)', r'(?:推翻|否定|质疑)'],
            'path_recalculation': [r'(?:换个角度|反过来想|重新审视)', r'(?:这意味着|也就是说)'],
            'uncertainty_markers': [r'(?:可能|也许|未必|不一定)']
        }
        self.logic_operators = {
            'basic': ['因为', '所以', '因此', '由于'],
            'conditional': ['如果', '那么', '假设', '即使', '除非'],
            'modal': ['必然', '可能', '必须', '只能'],
            'meta': ['考虑到', '反观', '本质上']
        }
        self.dim_patterns = [
            r'(?:混淆了|错把.*?与.*?混为一谈)',
            r'(?:表层|深层|本质|现象)',
            r'(?:科学上|文化上|逻辑上|事实上)'
        ]

    def evaluate(self, thinking: str) -> LogicalMetrics:
        if not thinking or len(thinking) < 50:
            return LogicalMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False)
        
        pivots = self._count_pivots(thinking)
        logic_density = self._calc_density(thinking)
        dim_score = self._detect_dim(thinking)
        coverage = self._analyze_options(thinking)
        fallacy = self._detect_fallacy(thinking)
        
        entropy = self._calc_entropy(thinking)
        abductive = self._detect_abduction(thinking)
        dialectic = self._detect_dialectic(thinking)
        abstraction = self._calc_abstraction(thinking)
        
        gold = (pivots * 1.5 + logic_density * 2.0 + dim_score * 2.5 +
                coverage * 3.0 + entropy * 1.2 + abductive * 2.5 +
                dialectic * 2.0 + abstraction * 1.0 - fallacy * 2.5)
        
        is_hq = (gold >= Config.MIN_GOLD_SCORE and coverage >= 1.0 and 
                fallacy < 1.5 and len(thinking) > 200)
        
        return LogicalMetrics(
            round(gold, 2), round(min(1.0, logic_density/15), 2),
            round(entropy, 2), round(abductive, 2), round(dialectic, 2),
            abstraction, int(pivots), round(logic_density, 2),
            round(dim_score, 2), round(coverage, 2), round(fallacy, 2),
            len(thinking), len(re.split(r'[。；\n]', thinking)), is_hq
        )

    def _count_pivots(self, text: str) -> float:
        count = 0
        for cat, pats in self.pivot_patterns.items():
            w = 2.0 if cat == 'strong_rejection' else 1.0
            for p in pats:
                count += len(re.findall(p, text)) * w
        return count

    def _calc_density(self, text: str) -> float:
        score, weights = 0, {'basic': 1, 'conditional': 1.5, 'modal': 2, 'meta': 2.5}
        for tier, ops in self.logic_operators.items():
            for op in ops:
                score += text.count(op) * weights[tier]
        return score / (len(text)/100 + 1)

    def _detect_dim(self, text: str) -> float:
        return sum(1.5 for p in self.dim_patterns if re.search(p, text))

    def _analyze_options(self, text: str) -> float:
        scores = []
        for opt in ['A', 'B', 'C', 'D']:
            matches = re.findall(rf'(?:选项[{opt}]|[{opt}][\.\、])(.{{0,80}})', text)
            if matches:
                depth = sum([
                    1 if re.search(r'(?:排除|错误)', ' '.join(matches)) else 0,
                    1 if re.search(r'(?:分析|来看)', ' '.join(matches)) else 0,
                    2 if re.search(r'(?:相比|比较)', ' '.join(matches)) else 0
                ])
                scores.append(min(depth, 4))
        return sum(scores)/4 if scores else 0

    def _detect_fallacy(self, text: str) -> float:
        risks = {r'(?:要么.*?要么|非此即彼)': 0.5, r'(?:显然|绝对)': 0.3, r'(?:进而|最终)': 0.2}
        return sum(len(re.findall(p, text))*w for p, w in risks.items())

    def _calc_entropy(self, text: str) -> float:
        t = len(re.findall(r'(?:但是|然而|反过来)', text))
        b = len(re.findall(r'(?:回到|如前)', text))
        return min(1.0, (t*0.3 + b*0.5)/5)

    def _detect_abduction(self, text: str) -> float:
        return min(1.0, sum(len(re.findall(p, text)) for p in 
               [r'(?:最合理的|最有可能)', r'(?:这说明)', r'(?:排除了)']) * 0.3)

    def _detect_dialectic(self, text: str) -> float:
        return sum([bool(re.search(p, text)) for p in 
               [r'(?:首先|虽然)', r'(?:但是|然而)', r'(?:因此|综合)']]) / 3

    def _calc_abstraction(self, text: str) -> int:
        levels = [(r'(?:具体|例子)', 1), (r'(?:因此|所以)', 2), (r'(?:本质|关键)', 3),
                 (r'(?:逻辑上|结构)', 4), (r'(?:方法论)', 5)]
        return max([l for p, l in levels if re.search(p, text)] or [1])

evaluator = LogiHardEvaluator()

# ==================== 主实验类（增强版） ====================

class LogiHardExperiment:
    def __init__(self, prompt_mode: str = Config.PROMPT_MODE):
        self.prompt_mode = prompt_mode
        self.results = []
        self.corpus = []
        self.processed_ids = set()
        self.stats = {"success": 0, "failed": 0, "retried": 0}
        self.interrupted = False  # 标记是否被中断
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理 Ctrl+C 和终止信号"""
        print(f"\n\n[警告] 收到中断信号 {signum}，正在保存进度...")
        self.interrupted = True
        self._emergency_save()
        print("进度已保存，安全退出")
        sys.exit(0)

    def _emergency_save(self):
        """紧急保存，不抛出异常"""
        try:
            self.save_checkpoint()
            # 同时保存临时结果
            if self.results:
                temp_file = f"emergency_save_{int(time.time())}.json"
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump({"results": self.results[-100:], "stats": self.stats}, f, ensure_ascii=False)
                print(f"紧急备份已保存: {temp_file}")
        except Exception as e:
            print(f"紧急保存失败: {e}")

    def generate_id(self, item: Dict) -> str:
        content = f"{item.get('context', '')}{item.get('question', '')}"
        return f"LOGI-{hashlib.md5(content.encode()).hexdigest()[:12]}"

    def process_single_with_retry(self, item: Dict) -> Optional[Tuple[Dict, Dict]]:
        """
        带重试机制的处理函数
        """
        uid = self.generate_id(item)
        prompt = PromptEngine.build(item, mode=self.prompt_mode)
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                start = time.time()
                resp = client.chat.completions.create(
                    model=Config.MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0,
                    max_tokens=Config.MAX_TOKENS,
                    timeout=120  # 设置超时
                )
                latency = time.time() - start
                
                thinking = getattr(resp.choices[0].message, "reasoning_content", "")
                answer = resp.choices[0].message.content
                
                if not thinking or len(thinking) < 100:
                    print(f"[警告] {uid} 思维链过短，可能未触发思考模式")
                    return None
                
                metrics = evaluator.evaluate(thinking)
                
                result = {
                    "id": uid,
                    "prompt_mode": self.prompt_mode,
                    "original": item,
                    "metrics": asdict(metrics),
                    "thinking": thinking,
                    "answer": answer,
                    "latency": round(latency, 2),
                    "timestamp": datetime.now().isoformat()
                }
                
                # 构建语料
                corpus_item = {
                    "id": uid,
                    "messages": [
                        {"role": "system", "content": "你是一个逻辑推理专家。"},
                        {"role": "user", "content": f"Context: {item['context']}\nQuestion: {item['question']}\nOptions: {', '.join(item['options'])}"},
                        {"role": "assistant", "content": f"{thinking}\n\n**Answer**: {answer}" if answer not in thinking[-100:] else thinking}
                    ],
                    "metadata": {
                        "logihard_score": metrics.gold_score,
                        "prompt_mode": self.prompt_mode,
                        "thinking_length": metrics.thinking_length,
                        "difficulty_tier": "Expert" if metrics.gold_score >= 30 else "Hard" if metrics.gold_score >= 25 else "Medium" if metrics.gold_score >= 20 else "Standard",
                        "cognitive_features": {
                            "has_dialectic": metrics.dialectic_quality > 0.5,
                            "has_abduction": metrics.abductive_strength > 0.5,
                            "abstraction_level": metrics.abstraction_level
                        }
                    }
                }
                
                return result, corpus_item
                
            except (APIConnectionError, APITimeoutError) as e:
                # 网络错误，指数退避重试
                wait_time = Config.RETRY_DELAY * (2 ** attempt)
                print(f"[网络错误] {uid}: {e}, 等待 {wait_time}s 后重试 ({attempt+1}/{Config.MAX_RETRIES})")
                time.sleep(wait_time)
                self.stats["retried"] += 1
                continue
                
            except APIError as e:
                # API 错误，可能不需要重试
                if e.status_code == 429:  # Rate limit
                    wait_time = 60
                    print(f"[限流] {uid}: 等待 {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[API错误] {uid}: {e}")
                    break
            except Exception as e:
                print(f"[错误] {uid}: {e}")
                break
        
        # 所有重试失败
        self.stats["failed"] += 1
        return None

    def save_checkpoint(self):
        checkpoint = {
            "results": self.results,
            "corpus": self.corpus,
            "processed_ids": list(self.processed_ids),
            "stats": self.stats,
            "prompt_mode": self.prompt_mode,
            "last_save": datetime.now().isoformat()
        }
        with open(Config.CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        print(f"[检查点] 已保存 {len(self.results)} 条记录")

    def load_checkpoint(self):
        if os.path.exists(Config.CHECKPOINT_FILE):
            try:
                with open(Config.CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                    ckpt = json.load(f)
                    self.results = ckpt.get("results", [])
                    self.corpus = ckpt.get("corpus", [])
                    self.processed_ids = set(ckpt.get("processed_ids", []))
                    self.stats = ckpt.get("stats", {"success": 0, "failed": 0, "retried": 0})
                    print(f"[恢复] 已加载 {len(self.processed_ids)} 条处理记录，模式: {ckpt.get('prompt_mode', 'unknown')}")
                    
                    # 验证数据一致性
                    if len(self.results) != len(self.corpus):
                        print("[警告] results 和 corpus 长度不一致，使用较短者")
                        min_len = min(len(self.results), len(self.corpus))
                        self.results = self.results[:min_len]
                        self.corpus = self.corpus[:min_len]
                        self.processed_ids = set(r["id"] for r in self.results)
                    return True
            except Exception as e:
                print(f"[错误] 加载检查点失败: {e}，重新开始")
                return False
        return False

    def run(self):
        print("=" * 70)
        print(f"LogiHard Experiment | Prompt Mode: {self.prompt_mode}")
        print("=" * 70)
        print("提示：按 Ctrl+C 可中断并保存进度")
        print("=" * 70)
        
        self.load_checkpoint()
        
        try:
            raw_data = load_json_or_jsonl(Config.INPUT_FILE)
        except Exception as e:
            print(f"无法加载数据: {e}")
            return
        
        todo = [it for it in raw_data if self.generate_id(it) not in self.processed_ids]
        total = len(raw_data)
        completed = len(self.processed_ids)
        
        print(f"总计: {total} | 已完成: {completed} | 待处理: {len(todo)}")
        
        if not todo:
            print("所有数据已处理完毕")
            self.finalize()
            return
        
        # 使用更简单的顺序处理或限制并发以避免网络拥堵
        # 网络不稳定时建议 MAX_WORKERS=2 或改为顺序处理
        workers = min(Config.MAX_WORKERS, 4)  # 限制最大并发数防止连接问题
        
        processed_count = 0
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_item = {executor.submit(self.process_single_with_retry, it): it for it in todo}
                
                for future in concurrent.futures.as_completed(future_to_item):
                    if self.interrupted:
                        break
                    
                    result = future.result()
                    if result:
                        self.results.append(result[0])
                        self.corpus.append(result[1])
                        self.processed_ids.add(result[0]["id"])
                        self.stats["success"] += 1
                    
                    processed_count += 1
                    
                    # 定期保存（更频繁）
                    if processed_count % Config.SAVE_INTERVAL == 0:
                        self.save_checkpoint()
                        print(f"进度: {completed + processed_count}/{total} "
                              f"(成功: {self.stats['success']}, 失败: {self.stats['failed']}, "
                              f"重试: {self.stats.get('retried', 0)})")
                    
                    # 每 100 条额外保存一次结果备份
                    if processed_count % 100 == 0:
                        self._backup_results()
                        
        except Exception as e:
            print(f"\n[严重错误] 处理循环异常: {e}")
        finally:
            # 确保总是保存
            print("\n正在保存最终进度...")
            self.save_checkpoint()
            if not self.interrupted:
                self.finalize()

    def _backup_results(self):
        """定期备份结果，防止 checkpoint 损坏"""
        backup_file = f"backup_{len(self.results)}.jsonl"
        try:
            save_jsonl(backup_file, self.results[-100:])  # 只保存最近100条作为增量
        except:
            pass

    def finalize(self):
        print("\n" + "=" * 70)
        print("数据整理与输出")
        print("=" * 70)
        
        if not self.results:
            print("没有数据需要保存")
            return
        
        # 排序
        sorted_pairs = sorted(zip(self.results, self.corpus), 
                             key=lambda x: x[0]["metrics"]["gold_score"], 
                             reverse=True)
        
        sorted_results = [p[0] for p in sorted_pairs]
        sorted_corpus = [p[1] for p in sorted_pairs]
        
        # 保存
        with open(Config.LOGIHARD_FULL, 'w', encoding='utf-8') as f:
            json.dump({"metadata": {"total": len(sorted_results)}, "data": sorted_results}, f, ensure_ascii=False, indent=2)
        
        with open(Config.CORPUS_FULL, 'w', encoding='utf-8') as f:
            json.dump({"metadata": {"total": len(sorted_corpus)}, "data": sorted_corpus}, f, ensure_ascii=False, indent=2)
        
        save_jsonl(Config.LOGIHARD_FULL.replace('.json', '.jsonl'), sorted_results)
        save_jsonl(Config.CORPUS_FULL.replace('.json', '.jsonl'), sorted_corpus)
        
        # Top-K
        k = min(Config.TOP_K_THRESHOLD, len(sorted_results))
        with open(Config.LOGIHARD_TOP2K, 'w', encoding='utf-8') as f:
            json.dump({"data": sorted_results[:k]}, f, ensure_ascii=False, indent=2)
        with open(Config.CORPUS_TOP2K, 'w', encoding='utf-8') as f:
            json.dump({"data": sorted_corpus[:k]}, f, ensure_ascii=False, indent=2)
        
        # 统计
        scores = [r["metrics"]["gold_score"] for r in sorted_results]
        print(f"\n完成！总计: {len(sorted_results)}")
        print(f"分数范围: {min(scores):.1f} - {max(scores):.1f} | 平均: {sum(scores)/len(scores):.1f}")
        print(f"失败/重试: {self.stats['failed']}/{self.stats.get('retried', 0)}")

# ==================== 入口 ====================

if __name__ == "__main__":
    # 恢复时自动继续
    exp = LogiHardExperiment(prompt_mode=Config.PROMPT_MODE)
    exp.run()
    print("实验结束")