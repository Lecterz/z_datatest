"""
GraphRAG评估脚本 - 基于多问题图谱架构的评估框架
提供多种评估指标和比较分析
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import re
from typing import List, Dict, Tuple

class QuestionGraphEvaluator:
    def __init__(self):
        self.metrics = {}
    
    def load_results(self, results_file: str) -> List[Dict]:
        """加载评估结果"""
        results = []
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                results.append(json.loads(line))
        return results
    
    def exact_match_score(self, prediction: str, ground_truths: List[str]) -> float:
        """精确匹配分数"""
        prediction = prediction.strip().lower()
        for gt in ground_truths:
            if prediction == gt.strip().lower():
                return 1.0
        return 0.0
    
    def contains_answer_score(self, prediction: str, ground_truths: List[str]) -> float:
        """包含答案分数"""
        prediction = prediction.lower()
        for gt in ground_truths:
            if gt.lower() in prediction:
                return 1.0
        return 0.0
    
    def f1_score(self, prediction: str, ground_truths: List[str]) -> float:
        """F1分数（基于token重叠）"""
        def tokenize(text):
            return re.findall(r'\w+', text.lower())
        
        pred_tokens = set(tokenize(prediction))
        max_f1 = 0.0
        
        for gt in ground_truths:
            gt_tokens = set(tokenize(gt))
            
            if len(pred_tokens) == 0 and len(gt_tokens) == 0:
                f1 = 1.0
            elif len(pred_tokens) == 0 or len(gt_tokens) == 0:
                f1 = 0.0
            else:
                common = pred_tokens & gt_tokens
                precision = len(common) / len(pred_tokens)
                recall = len(common) / len(gt_tokens)
                
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1 = 2 * precision * recall / (precision + recall)
            
            max_f1 = max(max_f1, f1)
        
        return max_f1
    
    def response_quality_score(self, prediction: str) -> float:
        """响应质量分数"""
        # 检查是否是失败响应
        fail_response = "Sorry, I'm not able to provide an answer to that question."
        if prediction.strip() == fail_response:
            return 0.0
        
        # 基本质量检查
        if len(prediction.strip()) < 20:
            return 0.3  # 太短
        
        if len(prediction.strip()) > 1500:
            return 0.8  # 可能过长但包含信息
        
        # 检查是否包含有意义的内容
        words = re.findall(r'\w+', prediction.lower())
        if len(words) < 10:
            return 0.4
        
        return 1.0  # 正常响应
    
    def retrieval_effectiveness_score(self, context: str, question: str) -> float:
        """检索效果分数 - 针对多问题图谱检索"""
        if "No relevant question graphs found" in context or "No relevant context found" in context:
            return 0.0
        
        # 检查是否找到了相关问题
        relevant_question_count = context.count("## Relevant Question")
        if relevant_question_count == 0:
            return 0.2
        
        # 检查上下文是否包含与问题相关的关键词
        def extract_keywords(text):
            words = re.findall(r'\w+', text.lower())
            # 过滤停用词
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'where', 'when', 'why', 'how', 'who', 'which'}
            return [w for w in words if w not in stop_words and len(w) > 2]
        
        question_keywords = set(extract_keywords(question))
        context_keywords = set(extract_keywords(context))
        
        if len(question_keywords) == 0:
            return 0.5
        
        overlap = question_keywords & context_keywords
        base_score = min(1.0, len(overlap) / len(question_keywords) * 2)
        
        # 根据找到的相关问题数量调整分数
        question_bonus = min(0.3, relevant_question_count * 0.1)
        
        return min(1.0, base_score + question_bonus)
    
    def multi_question_utilization_score(self, context: str) -> float:
        """多问题利用率分数 - 新增指标"""
        if "No relevant question graphs found" in context:
            return 0.0
        
        # 计算使用的问题数量
        relevant_question_count = context.count("## Relevant Question")
        
        # 检查是否有实际的实体和关系信息
        entity_count = context.count("Entity:")
        relationship_count = context.count("Relationship:")
        
        if relevant_question_count == 0:
            return 0.0
        elif relevant_question_count == 1:
            return 0.3
        elif relevant_question_count == 2:
            return 0.6
        elif relevant_question_count >= 3:
            base_score = 0.8
            # 如果有丰富的实体和关系信息，额外加分
            if entity_count >= 5 and relationship_count >= 3:
                return min(1.0, base_score + 0.2)
            return base_score
        
        return 0.5
    
    def evaluate_results(self, results: List[Dict]) -> Dict:
        """评估结果"""
        metrics = {
            'exact_match': [],
            'contains_answer': [],
            'f1_score': [],
            'response_quality': [],
            'retrieval_effectiveness': [],
            'multi_question_utilization': [],
            'response_length': [],
            'context_length': [],
            'max_questions_used': []
        }
        
        for result in results:
            prediction = result['response']
            ground_truths = result.get('ground_truth', [])
            question = result['query']
            context = result.get('context', '')
            max_questions = result.get('max_questions', 1)
            
            # 计算各种指标
            metrics['exact_match'].append(
                self.exact_match_score(prediction, ground_truths)
            )
            metrics['contains_answer'].append(
                self.contains_answer_score(prediction, ground_truths)
            )
            metrics['f1_score'].append(
                self.f1_score(prediction, ground_truths)
            )
            metrics['response_quality'].append(
                self.response_quality_score(prediction)
            )
            metrics['retrieval_effectiveness'].append(
                self.retrieval_effectiveness_score(context, question)
            )
            metrics['multi_question_utilization'].append(
                self.multi_question_utilization_score(context)
            )
            metrics['response_length'].append(len(prediction))
            metrics['context_length'].append(len(context))
            metrics['max_questions_used'].append(max_questions)
        
        # 计算统计值
        summary = {}
        for metric, values in metrics.items():
            if metric == 'max_questions_used':
                summary[metric] = {
                    'mean': np.mean(values),
                    'mode': Counter(values).most_common(1)[0][0] if values else 0
                }
            else:
                summary[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary
    
    def analyze_by_question_type(self, results: List[Dict]) -> Dict:
        """按问题类型分析"""
        def classify_question(question: str) -> str:
            question_lower = question.lower()
            if question_lower.startswith('what'):
                return 'what'
            elif question_lower.startswith('who'):
                return 'who'
            elif question_lower.startswith('when'):
                return 'when'
            elif question_lower.startswith('where'):
                return 'where'
            elif question_lower.startswith('why'):
                return 'why'
            elif question_lower.startswith('how'):
                return 'how'
            else:
                return 'other'
        
        type_results = defaultdict(list)
        
        for result in results:
            question_type = classify_question(result['query'])
            type_results[question_type].append(result)
        
        type_analysis = {}
        for q_type, type_res in type_results.items():
            type_analysis[q_type] = {
                'count': len(type_res),
                'metrics': self.evaluate_results(type_res)
            }
        
        return type_analysis
    
    def compare_search_modes(self, results_dir: Path) -> Dict:
        """比较不同搜索模式的效果"""
        search_modes = ['single_best', 'multi_question']
        mode_results = {}
        
        for mode in search_modes:
            results_file = results_dir / f"evaluation_results_{mode}.jsonl"
            if results_file.exists():
                results = self.load_results(results_file)
                mode_results[mode] = self.evaluate_results(results)
        
        return mode_results
    
    def analyze_failure_cases(self, results: List[Dict]) -> Dict:
        """分析失败案例"""
        fail_response = "Sorry, I'm not able to provide an answer to that question."
        
        failure_analysis = {
            'total_failures': 0,
            'failure_by_question_type': defaultdict(int),
            'no_relevant_questions_failures': 0,
            'single_question_only': 0,
            'retrieval_quality_issues': 0
        }
        
        for result in results:
            if result['response'].strip() == fail_response:
                failure_analysis['total_failures'] += 1
                
                # 按问题类型统计
                question = result['query'].lower()
                if question.startswith('what'):
                    failure_analysis['failure_by_question_type']['what'] += 1
                elif question.startswith('who'):
                    failure_analysis['failure_by_question_type']['who'] += 1
                elif question.startswith('when'):
                    failure_analysis['failure_by_question_type']['when'] += 1
                elif question.startswith('where'):
                    failure_analysis['failure_by_question_type']['where'] += 1
                elif question.startswith('why'):
                    failure_analysis['failure_by_question_type']['why'] += 1
                elif question.startswith('how'):
                    failure_analysis['failure_by_question_type']['how'] += 1
                else:
                    failure_analysis['failure_by_question_type']['other'] += 1
                
                # 分析失败原因
                context = result.get('context', '')
                if "No relevant question graphs found" in context:
                    failure_analysis['no_relevant_questions_failures'] += 1
                elif context.count("## Relevant Question") == 1:
                    failure_analysis['single_question_only'] += 1
                elif self.retrieval_effectiveness_score(context, result['query']) < 0.3:
                    failure_analysis['retrieval_quality_issues'] += 1
        
        return failure_analysis
    
    def generate_report(self, results: List[Dict], output_file: str, search_mode: str = "unknown"):
        """生成详细评估报告"""
        # 整体评估
        overall_metrics = self.evaluate_results(results)
        
        # 按问题类型分析
        type_analysis = self.analyze_by_question_type(results)
        
        # 失败案例分析
        failure_analysis = self.analyze_failure_cases(results)
        
        # 获取最佳和最差响应示例
        best_responses = self._get_best_responses(results)
        worst_responses = self._get_worst_responses(results)
        
        # 生成报告
        report = {
            'search_mode': search_mode,
            'summary': {
                'total_questions': len(results),
                'overall_metrics': overall_metrics,
                'failure_analysis': failure_analysis
            },
            'by_question_type': type_analysis,
            'examples': {
                'best_responses': best_responses,
                'worst_responses': worst_responses
            }
        }
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        self._print_summary(report)
        
        return report
    
    def _get_best_responses(self, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """获取最佳回答示例"""
        # 基于综合分数排序
        scored_results = []
        for result in results:
            f1 = self.f1_score(result['response'], result.get('ground_truth', []))
            quality = self.response_quality_score(result['response'])
            contains = self.contains_answer_score(result['response'], result.get('ground_truth', []))
            retrieval = self.retrieval_effectiveness_score(result.get('context', ''), result['query'])
            multi_q = self.multi_question_utilization_score(result.get('context', ''))
            
            # 综合分数 - 针对多问题图谱架构调整权重
            combined_score = (f1 * 0.3 + quality * 0.2 + contains * 0.2 + retrieval * 0.15 + multi_q * 0.15)
            scored_results.append((combined_score, result))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [result for score, result in scored_results[:top_k]]
    
    def _get_worst_responses(self, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """获取最差回答示例"""
        # 基于综合分数排序
        scored_results = []
        for result in results:
            f1 = self.f1_score(result['response'], result.get('ground_truth', []))
            quality = self.response_quality_score(result['response'])
            contains = self.contains_answer_score(result['response'], result.get('ground_truth', []))
            retrieval = self.retrieval_effectiveness_score(result.get('context', ''), result['query'])
            multi_q = self.multi_question_utilization_score(result.get('context', ''))
            
            # 综合分数
            combined_score = (f1 * 0.3 + quality * 0.2 + contains * 0.2 + retrieval * 0.15 + multi_q * 0.15)
            scored_results.append((combined_score, result))
        
        scored_results.sort(key=lambda x: x[0])
        return [result for score, result in scored_results[:top_k]]
    
    def _print_summary(self, report: Dict):
        """打印评估摘要"""
        print("="*80)
        print(f"Multi-Question GraphRAG Evaluation Summary - {report['search_mode']} Mode")
        print("="*80)
        
        summary = report['summary']
        print(f"Total Questions: {summary['total_questions']}")
        print()
        
        overall = summary['overall_metrics']
        print("Overall Metrics:")
        for metric, values in overall.items():
            if metric in ['response_length', 'context_length']:
                print(f"  {metric}: {values['mean']:.1f} ± {values['std']:.1f}")
            elif metric == 'max_questions_used':
                print(f"  {metric}: mean={values['mean']:.1f}, mode={values['mode']}")
            else:
                print(f"  {metric}: {values['mean']:.3f} ± {values['std']:.3f}")
        print()
        
        # 失败分析
        failure = summary['failure_analysis']
        print("Failure Analysis:")
        print(f"  Total failures: {failure['total_failures']}/{summary['total_questions']} ({failure['total_failures']/summary['total_questions']*100:.1f}%)")
        print(f"  No relevant questions found: {failure['no_relevant_questions_failures']}")
        print(f"  Single question only: {failure['single_question_only']}")
        print(f"  Retrieval quality issues: {failure['retrieval_quality_issues']}")
        print()
        
        print("By Question Type:")
        for q_type, analysis in report['by_question_type'].items():
            count = analysis['count']
            f1_mean = analysis['metrics']['f1_score']['mean']
            contains_mean = analysis['metrics']['contains_answer']['mean']
            quality_mean = analysis['metrics']['response_quality']['mean']
            multi_q_mean = analysis['metrics']['multi_question_utilization']['mean']
            print(f"  {q_type}: {count} questions, F1={f1_mean:.3f}, Contains={contains_mean:.3f}, Quality={quality_mean:.3f}, MultiQ={multi_q_mean:.3f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Question GraphRAG Evaluation Tool")
    parser.add_argument("--results_file", help="Single results JSONL file")
    parser.add_argument("--results_dir", help="Directory containing multiple result files")
    parser.add_argument("--output_dir", required=True, help="Output directory for evaluation")
    parser.add_argument("--compare_modes", action="store_true", help="Compare different search modes")
    parser.add_argument("--search_mode", default="unknown", help="Search mode for single file evaluation")
    return parser.parse_args()

def main():
    args = parse_args()
    
    evaluator = QuestionGraphEvaluator()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.compare_modes and args.results_dir:
        # 比较不同搜索模式
        results_dir = Path(args.results_dir)
        mode_comparison = evaluator.compare_search_modes(results_dir)
        
        comparison_file = output_path / "mode_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(mode_comparison, f, indent=2, ensure_ascii=False)
        
        print("="*80)
        print("Multi-Question Graph Search Mode Comparison")
        print("="*80)
        
        # 创建比较表格
        if mode_comparison:
            print(f"{'Mode':<15} {'F1':<8} {'Contains':<10} {'Quality':<8} {'Retrieval':<10} {'MultiQ':<8}")
            print("-" * 65)
            
            for mode, metrics in mode_comparison.items():
                f1_mean = metrics['f1_score']['mean']
                contains_mean = metrics['contains_answer']['mean']
                quality_mean = metrics['response_quality']['mean']
                retrieval_mean = metrics['retrieval_effectiveness']['mean']
                multi_q_mean = metrics['multi_question_utilization']['mean']
                print(f"{mode:<15} {f1_mean:<8.3f} {contains_mean:<10.3f} {quality_mean:<8.3f} {retrieval_mean:<10.3f} {multi_q_mean:<8.3f}")
            
            # 找出最佳模式
            best_mode = max(mode_comparison.items(), 
                          key=lambda x: (x[1]['f1_score']['mean'] + 
                                       x[1]['response_quality']['mean'] + 
                                       x[1]['multi_question_utilization']['mean']) / 3)
            print(f"\n🏆 Best performing mode: {best_mode[0]}")
            
        print(f"\nDetailed comparison saved to: {comparison_file}")
        
        # 为每个模式生成详细报告
        for mode in ['single_best', 'multi_question']:
            results_file = results_dir / f"evaluation_results_{mode}.jsonl"
            if results_file.exists():
                results = evaluator.load_results(results_file)
                report_file = output_path / f"detailed_report_{mode}.json"
                evaluator.generate_report(results, report_file, mode)
    
    elif args.results_file:
        # 单文件评估
        results = evaluator.load_results(args.results_file)
        report_file = output_path / f"evaluation_report_{args.search_mode}.json"
        
        report = evaluator.generate_report(results, report_file, args.search_mode)
        print(f"\nDetailed report saved to: {report_file}")
        
        # 额外分析
        print("\n" + "="*80)
        print("Additional Multi-Question Graph Analysis")
        print("="*80)
        
        # 响应长度分布
        response_lengths = [len(r['response']) for r in results]
        print(f"Response length distribution:")
        print(f"  Min: {min(response_lengths)} chars")
        print(f"  Max: {max(response_lengths)} chars")
        print(f"  Mean: {np.mean(response_lengths):.1f} chars")
        print(f"  Median: {np.median(response_lengths):.1f} chars")
        
        # 多问题利用率分析
        multi_q_scores = [evaluator.multi_question_utilization_score(r.get('context', '')) for r in results]
        print(f"\nMulti-question utilization:")
        print(f"  Mean utilization: {np.mean(multi_q_scores):.3f}")
        print(f"  High utilization (>0.7): {sum(1 for s in multi_q_scores if s > 0.7)}/{len(multi_q_scores)}")
        print(f"  No multi-question usage (=0): {sum(1 for s in multi_q_scores if s == 0)}/{len(multi_q_scores)}")
        
        # 检索效果分析
        retrieval_scores = [evaluator.retrieval_effectiveness_score(r.get('context', ''), r['query']) for r in results]
        print(f"\nRetrieval effectiveness:")
        print(f"  Mean effectiveness: {np.mean(retrieval_scores):.3f}")
        print(f"  High effectiveness (>0.7): {sum(1 for s in retrieval_scores if s > 0.7)}/{len(retrieval_scores)}")
        print(f"  No retrieval (=0): {sum(1 for s in retrieval_scores if s == 0)}/{len(retrieval_scores)}")
    
    else:
        print("Error: Please provide either --results_file or --results_dir with --compare_modes")
        return

if __name__ == "__main__":
    main()