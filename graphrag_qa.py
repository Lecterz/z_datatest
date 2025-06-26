import json
import argparse
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from graphrag_retriever import QuestionGraphRetriever

# 基于lecterz的QA prompt模板
RAG_RESPONSE = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.

---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

FAIL_RESPONSE = "Sorry, I'm not able to provide an answer to that question."

class QuestionGraphQA:
    def __init__(self, 
                 graphs_dir: str,
                 model_name: str,
                 world_size: int = 1,
                 max_new_tokens: int = 1024,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        # 初始化检索器
        self.retriever = QuestionGraphRetriever(
            graphs_dir=graphs_dir,
            embedding_model=embedding_model
        )
        
        # 初始化生成模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=world_size,
            trust_remote_code=False,
            gpu_memory_utilization=0.9,
            max_num_batched_tokens=8192,
            max_num_seqs=256,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_new_tokens,
        )
    
    def generate_response(self, query: str, search_mode: str = "multi_question", max_questions: int = 3) -> Dict:
        """生成回答，使用lecterz的RAG响应格式"""
        # 检索相关上下文
        context = self.retriever.search_and_format(query, search_mode, max_questions)
        
        # 构建prompt，基于lecterz的RAG_RESPONSE模板
        prompt = self._build_qa_prompt(query, context)
        
        # 生成回答
        generation = self.llm.generate([prompt], self.sampling_params)[0]
        response = generation.outputs[0].text.strip()
        
        # 如果响应为空或过短，返回失败响应
        if not response or len(response.strip()) < 10:
            response = FAIL_RESPONSE
        
        return {
            "query": query,
            "context": context,
            "response": response,
            "search_mode": search_mode,
            "max_questions": max_questions
        }
    
    def _build_qa_prompt(self, query: str, context: str) -> str:
        """构建QA prompt，基于lecterz的RAG_RESPONSE模板"""
        # 设置响应类型
        response_type = "Multiple paragraphs"
        
        # 格式化上下文数据
        if not context or context.strip() == "No relevant context found.":
            context_data = "No relevant context found."
        else:
            context_data = context
        
        # 使用lecterz的prompt模板
        system_prompt = RAG_RESPONSE.format(
            response_type=response_type,
            context_data=context_data
        )
        
        # 构建完整的prompt
        full_prompt = f"{system_prompt}\n\nQuestion: {query}\n\nAnswer:"
        
        return full_prompt
    
    def evaluate_on_dataset(self, dataset_file: str, output_file: str, search_mode: str = "multi_question", max_questions: int = 3):
        """在数据集上评估"""
        # 加载数据集
        with open(dataset_file, 'r', encoding='utf-8') as f:
            dataset = []
            for line in f:
                data = json.loads(line)
                dataset.append(data)
        
        results = []
        
        print(f"Evaluating on {len(dataset)} examples...")
        for i, example in enumerate(dataset):
            if i % 10 == 0:
                print(f"Processing {i}/{len(dataset)}")
            
            question = example["question"]
            ground_truth = example.get("answers", [])
            
            # 生成回答
            result = self.generate_response(question, search_mode, max_questions)
            result["ground_truth"] = ground_truth
            result["example_id"] = i
            
            results.append(result)
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"Evaluation results saved to {output_file}")
        return results
    
    def batch_generate(self, questions: List[str], search_mode: str = "multi_question", max_questions: int = 3) -> List[Dict]:
        """批量生成回答"""
        results = []
        
        print(f"Processing {len(questions)} questions in batch mode...")
        
        # 为了效率，我们可以并行处理检索，但这里简化为串行
        for i, question in enumerate(questions):
            if i % 10 == 0:
                print(f"Processing {i}/{len(questions)}")
            
            result = self.generate_response(question, search_mode, max_questions)
            results.append(result)
        
        return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_dir", required=True, help="Directory containing question graph subdirectories")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct", help="Model name for generation")
    parser.add_argument("--world_size", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum new tokens")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Embedding model")
    
    # 运行模式
    subparsers = parser.add_subparsers(dest="mode", help="Running mode")
    
    # 单个问题模式
    single_parser = subparsers.add_parser("single", help="Answer a single question")
    single_parser.add_argument("--query", required=True, help="Question to answer")
    single_parser.add_argument("--search_mode", choices=["single_best", "multi_question"], default="multi_question")
    single_parser.add_argument("--max_questions", type=int, default=3, help="Maximum number of question graphs to search")
    
    # 数据集评估模式
    eval_parser = subparsers.add_parser("eval", help="Evaluate on dataset")
    eval_parser.add_argument("--dataset_file", required=True, help="Dataset file (JSONL)")
    eval_parser.add_argument("--output_file", required=True, help="Output file for results")
    eval_parser.add_argument("--search_mode", choices=["single_best", "multi_question"], default="multi_question")
    eval_parser.add_argument("--max_questions", type=int, default=3, help="Maximum number of question graphs to search")
    
    # 批量处理模式
    batch_parser = subparsers.add_parser("batch", help="Batch process questions")
    batch_parser.add_argument("--questions_file", required=True, help="File containing questions (one per line)")
    batch_parser.add_argument("--output_file", required=True, help="Output file for results")
    batch_parser.add_argument("--search_mode", choices=["single_best", "multi_question"], default="multi_question")
    batch_parser.add_argument("--max_questions", type=int, default=3, help="Maximum number of question graphs to search")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not args.mode:
        print("Please specify a mode: single, eval, or batch")
        return
    
    # 初始化QA系统
    qa_system = QuestionGraphQA(
        graphs_dir=args.graphs_dir,
        model_name=args.model_name,
        world_size=args.world_size,
        max_new_tokens=args.max_new_tokens,
        embedding_model=args.embedding_model
    )
    
    if args.mode == "single":
        # 单个问题模式
        result = qa_system.generate_response(args.query, args.search_mode, args.max_questions)
        
        print("="*80)
        print(f"Question: {result['query']}")
        print("="*80)
        print("Context:")
        print(result['context'])
        print("="*80)
        print("Answer:")
        print(result['response'])
        print("="*80)
        
    elif args.mode == "eval":
        # 数据集评估模式
        results = qa_system.evaluate_on_dataset(
            args.dataset_file, 
            args.output_file, 
            args.search_mode,
            args.max_questions
        )
        
        # 简单的准确性统计
        total = len(results)
        print(f"\nEvaluation completed on {total} examples")
        print(f"Results saved to {args.output_file}")
        
        # 计算一些基本统计
        non_fail_responses = [r for r in results if r['response'] != FAIL_RESPONSE]
        print(f"Non-fail responses: {len(non_fail_responses)}/{total} ({len(non_fail_responses)/total*100:.1f}%)")
        
        if non_fail_responses:
            avg_response_length = sum(len(r['response']) for r in non_fail_responses) / len(non_fail_responses)
            print(f"Average response length: {avg_response_length:.1f} characters")
        
    elif args.mode == "batch":
        # 批量处理模式
        # 读取问题文件
        with open(args.questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        # 批量处理
        results = qa_system.batch_generate(questions, args.search_mode, args.max_questions)
        
        # 保存结果
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        print(f"Batch processing completed. Results saved to {args.output_file}")
        
        # 统计信息
        total = len(results)
        non_fail_responses = [r for r in results if r['response'] != FAIL_RESPONSE]
        print(f"Processed {total} questions")
        print(f"Successful responses: {len(non_fail_responses)}/{total} ({len(non_fail_responses)/total*100:.1f}%)")

if __name__ == "__main__":
    main()