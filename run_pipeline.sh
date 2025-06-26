#!/bin/bash

# Enhanced GraphRAG Pipeline for feihm project
# 基于lecterz架构完善的feihm GraphRAG流水线

set -e

# 配置参数
export CUDA_VISIBLE_DEVICES=1,4,5,6
WORLD_SIZE=4
export OMP_NUM_THREADS=64
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

# 路径配置
DATASET_PATH=~/llm-fei/Data/NQ/contriever_nq_all_train/train.json
MODEL_PATH=/home/feihm/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
OUTPUT_DIR=./enhanced_graphrag_output
EXTRACTION_DIR=${OUTPUT_DIR}/extraction
GRAPH_DIR=${OUTPUT_DIR}/graph
QA_DIR=${OUTPUT_DIR}/qa

# 创建输出目录
mkdir -p ${EXTRACTION_DIR}
mkdir -p ${GRAPH_DIR}
mkdir -p ${QA_DIR}

echo "🚀 Starting Enhanced feihm GraphRAG Pipeline - Individual Question Graphs..."
echo "📝 Architecture: Each question gets its own knowledge graph (N questions = N graphs)"

# 步骤1: 增强实体关系提取（基于lecterz的gleaning机制）
echo "📋 Step 1: Enhanced Entity and Relationship Extraction with Gleaning"
python enhanced_extract.py \
    --data_file ${DATASET_PATH} \
    --model_name ${MODEL_PATH} \
    --world_size ${WORLD_SIZE} \
    --dest_dir ${EXTRACTION_DIR} \
    --num_proc 32 \
    --batch_size 16 \
    --max_new_tokens 2048

# 检查提取是否成功
if [ ! -f "${EXTRACTION_DIR}/enhanced_extraction_results.jsonl" ]; then
    echo "❌ Enhanced extraction failed!"
    exit 1
fi

echo "✅ Step 1 completed: Enhanced extraction with gleaning mechanism"

# 步骤2: 为每个问题构建独立的知识图谱
echo "🔗 Step 2: Building Individual Knowledge Graphs for Each Question"
python graph_builder.py \
    --extraction_file ${EXTRACTION_DIR}/enhanced_extraction_results.jsonl \
    --output_dir ${GRAPH_DIR} \
    --similarity_threshold 0.7 \
    --similarity_top_k 10 \
    --enable_community \
    --embedding_model all-MiniLM-L6-v2

# 检查图构建是否成功
if [ ! -f "${GRAPH_DIR}/overall_statistics.json" ]; then
    echo "❌ Question graph building failed!"
    exit 1
fi

echo "✅ Step 2 completed: Individual question graphs built"

# 显示图统计信息
if [ -f "${GRAPH_DIR}/overall_statistics.json" ]; then
    echo "📊 Question Graphs Statistics:"
    cat ${GRAPH_DIR}/overall_statistics.json | python -m json.tool
fi

# 步骤3: 测试多问题图谱检索系统
echo "🔍 Step 3: Testing Multi-Question Graph Retrieval System"

# 创建测试查询
TEST_QUERIES=(
    "What is the capital of France?"
    "Who invented the telephone?"
    "What causes earthquakes?"
    "How does photosynthesis work?"
    "When was the Internet created?"
    "What is machine learning?"
    "How do vaccines work?"
    "What is climate change?"
)

echo "Testing retrieval with sample queries using different modes..."

# 首先显示问题图谱统计
echo "--- Question Graph Statistics ---"
python graphrag_retriever.py \
    --graphs_dir ${GRAPH_DIR} \
    --query "dummy" \
    --stats \
    --search_mode single_best 2>/dev/null | head -10

# 测试单一最佳问题图谱搜索
echo "--- Testing Single Best Question Graph Search ---"
for query in "${TEST_QUERIES[@]:0:3}"; do
    echo "Single Best Query: $query"
    python graphrag_retriever.py \
        --graphs_dir ${GRAPH_DIR} \
        --query "$query" \
        --search_mode single_best \
        --max_tokens 2000
    echo "---"
done

# 测试多问题图谱搜索
echo "--- Testing Multi-Question Graph Search ---"
for query in "${TEST_QUERIES[@]:3:5}"; do
    echo "Multi-Question Query: $query"
    python graphrag_retriever.py \
        --graphs_dir ${GRAPH_DIR} \
        --query "$query" \
        --search_mode multi_question \
        --max_questions 3 \
        --max_tokens 4000
    echo "---"
done

# 步骤4: QA系统测试（基于多问题图谱检索）
echo "💬 Step 4: Multi-Question Graph QA System Test"

# 测试不同搜索模式的QA效果
echo "Testing QA with single best question graph mode:"
python graphrag_qa.py single \
    --graphs_dir ${GRAPH_DIR} \
    --model_name ${MODEL_PATH} \
    --world_size ${WORLD_SIZE} \
    --query "What is the capital of France?" \
    --search_mode single_best

echo "Testing QA with multi-question graph mode:"
python graphrag_qa.py single \
    --graphs_dir ${GRAPH_DIR} \
    --model_name ${MODEL_PATH} \
    --world_size ${WORLD_SIZE} \
    --query "How do vaccines work?" \
    --search_mode multi_question \
    --max_questions 3

echo "Testing QA with extended multi-question search:"
python graphrag_qa.py single \
    --graphs_dir ${GRAPH_DIR} \
    --model_name ${MODEL_PATH} \
    --world_size ${WORLD_SIZE} \
    --query "What causes climate change?" \
    --search_mode multi_question \
    --max_questions 5

echo "✅ Step 4 completed: QA system tested with multiple search modes"

# 步骤5: 数据集评估（可选）
if [ "$1" = "--full_eval" ]; then
    echo "📊 Step 5: Full Dataset Evaluation"
    
    # 准备测试数据集（取前100个样本）
    echo "Preparing test dataset..."
    head -n 100 ${DATASET_PATH} > ${QA_DIR}/test_dataset.jsonl
    
    # 测试不同搜索模式
    for mode in "local" "global" "hybrid"; do
        echo "Evaluating with ${mode} search mode..."
        python graphrag_qa.py eval \
            --graph_dir ${GRAPH_DIR} \
            --model_name ${MODEL_PATH} \
            --world_size ${WORLD_SIZE} \
            --dataset_file ${QA_DIR}/test_dataset.jsonl \
            --output_file ${QA_DIR}/evaluation_results_${mode}.jsonl \
            --search_mode ${mode}
    done
    
    echo "✅ Step 5 completed: Full evaluation with all search modes"
    
    # 比较不同搜索模式的效果
    echo "📈 Evaluation Results Comparison:"
    for mode in "local" "global" "hybrid"; do
        if [ -f "${QA_DIR}/evaluation_results_${mode}.jsonl" ]; then
            echo "=== ${mode} Search Mode ==="
            python -c "
import json
results = []
with open('${QA_DIR}/evaluation_results_${mode}.jsonl', 'r') as f:
    for line in f:
        results.append(json.loads(line))

total = len(results)
fail_responses = [r for r in results if r['response'] == 'Sorry, I'\''m not able to provide an answer to that question.']
non_fail = total - len(fail_responses)

print(f'Total questions: {total}')
print(f'Successful responses: {non_fail}/{total} ({non_fail/total*100:.1f}%)')

if non_fail > 0:
    successful_results = [r for r in results if r['response'] != 'Sorry, I'\''m not able to provide an answer to that question.']
    avg_length = sum(len(r['response']) for r in successful_results) / len(successful_results)
    print(f'Average response length: {avg_length:.1f} characters')
    
    # 简单的包含答案检查
    contains_answer = 0
    for r in successful_results:
        response_lower = r['response'].lower()
        for answer in r.get('ground_truth', []):
            if answer.lower() in response_lower:
                contains_answer += 1
                break
    
    if successful_results:
        print(f'Responses containing ground truth: {contains_answer}/{len(successful_results)} ({contains_answer/len(successful_results)*100:.1f}%)')
"
            echo ""
        fi
    done
    
else
    echo "ℹ️  Skipping full evaluation. Use '--full_eval' flag to run complete evaluation."
fi

# 步骤6: 创建示例问题文件并测试批量处理
echo "📝 Step 6: Testing Batch Processing"

# 创建示例问题文件
cat > ${QA_DIR}/sample_questions.txt << EOF
What is artificial intelligence?
How do neural networks work?
What is the difference between machine learning and deep learning?
What causes global warming?
How do electric cars work?
EOF

# 测试批量处理
python graphrag_qa.py batch \
    --graph_dir ${GRAPH_DIR} \
    --model_name ${MODEL_PATH} \
    --world_size ${WORLD_SIZE} \
    --questions_file ${QA_DIR}/sample_questions.txt \
    --output_file ${QA_DIR}/batch_results.jsonl \
    --search_mode hybrid

echo "✅ Step 6 completed: Batch processing tested"

# 总结
echo ""
echo "🎉 Enhanced feihm GraphRAG Pipeline Completed!"
echo "📁 Output Structure:"
echo "   ${OUTPUT_DIR}/"
echo "   ├── extraction/"
echo "   │   └── enhanced_extraction_results.jsonl    # 增强的实体关系提取结果"
echo "   ├── graph/"
echo "   │   ├── nx_data.graphml                       # NetworkX图（lecterz兼容格式）"
echo "   │   ├── knowledge_graph.gpickle               # 快速加载格式"
echo "   │   ├── entities.json                         # 实体数据"
echo "   │   ├── relationships.json                    # 关系数据"
echo "   │   ├── entity_vectors.npz                    # 实体向量"
echo "   │   ├── relationship_vectors.npz              # 关系向量"
echo "   │   └── graph_stats.json                      # 图统计信息"
echo "   └── qa/"
echo "       ├── test_dataset.jsonl                    # 测试数据集"
echo "       ├── evaluation_results_*.jsonl           # 评估结果"
echo "       ├── sample_questions.txt                 # 示例问题"
echo "       └── batch_results.jsonl                  # 批量处理结果"

echo ""
echo "🔧 Usage Examples:"
echo ""
echo "# 单个问题（不同搜索模式）:"
echo "python graphrag_qa.py single --graph_dir ${GRAPH_DIR} --query 'Your question here' --search_mode hybrid"
echo "python graphrag_qa.py single --graph_dir ${GRAPH_DIR} --query 'Your question here' --search_mode local"
echo "python graphrag_qa.py single --graph_dir ${GRAPH_DIR} --query 'Your question here' --search_mode global"
echo ""
echo "# 检索测试:"
echo "python graphrag_retriever.py --graph_dir ${GRAPH_DIR} --query 'Your query here' --search_mode hybrid"
echo ""
echo "# 数据集评估:"
echo "python graphrag_qa.py eval --graph_dir ${GRAPH_DIR} --dataset_file your_data.jsonl --output_file results.jsonl"
echo ""
echo "# 批量处理:"
echo "python graphrag_qa.py batch --graph_dir ${GRAPH_DIR} --questions_file questions.txt --output_file results.jsonl"

echo ""
echo "🆚 Key Improvements over Original feihm:"
echo "   ✓ Multi-round gleaning extraction (based on lecterz)"
echo "   ✓ lecterz-style entity/relationship merging"
echo "   ✓ Community detection with Louvain algorithm"
echo "   ✓ Multi-mode retrieval (local/global/hybrid)"
echo "   ✓ Vector-based similarity search"
echo "   ✓ lecterz-compatible graph storage format"
echo "   ✓ Comprehensive evaluation framework"
echo ""
echo "📈 Expected Performance Improvement:"
echo "   Original feihm exact_match: 0.33"
echo "   Enhanced version: Expected significant improvement with multi-mode retrieval"