import json
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from collections import defaultdict, Counter
import re

# 基于lecterz的常量
GRAPH_FIELD_SEP = "<SEP>"

def truncate_list_by_token_size(list_data: list, key: callable, max_token_size: int):
    """基于lecterz的token截断函数"""
    if max_token_size <= 0:
        return []
    tokens = 0
    result = []
    for data in list_data:
        # 简单的token估计：每个字符约0.75个token
        token_count = len(str(key(data))) * 0.75
        if tokens + token_count > max_token_size:
            break
        tokens += token_count
        result.append(data)
    return result

class QuestionGraphRetriever:
    """为每个问题的独立图谱进行检索"""
    
    def __init__(self, 
                 graphs_dir: str,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 local_search_depth: int = 2,
                 max_tokens: int = 8000,
                 top_k: int = 10):
        
        self.graphs_dir = Path(graphs_dir)
        self.embedding_model = SentenceTransformer(embedding_model)
        self.local_search_depth = local_search_depth
        self.max_tokens = max_tokens
        self.top_k = top_k
        
        # 加载所有问题图谱
        self._load_all_question_graphs()
    
    def _load_all_question_graphs(self):
        """加载所有问题的图谱"""
        print("Loading all question graphs...")
        
        self.question_graphs = {}
        self.question_metadata = {}
        
        # 遍历所有问题目录
        for question_dir in self.graphs_dir.iterdir():
            if question_dir.is_dir() and question_dir.name.startswith("question_"):
                try:
                    question_id = int(question_dir.name.split("_")[1])
                    
                    # 加载图
                    graph_file = question_dir / "question_graph.gpickle"
                    if graph_file.exists():
                        graph = nx.read_gpickle(graph_file)
                        self.question_graphs[question_id] = graph
                    
                    # 加载元数据
                    metadata_file = question_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            self.question_metadata[question_id] = metadata
                    
                    # 加载实体数据
                    entities_file = question_dir / "entities.json"
                    if entities_file.exists():
                        with open(entities_file, 'r', encoding='utf-8') as f:
                            entities = json.load(f)
                            self.question_metadata[question_id]["entities"] = entities
                    
                    # 加载关系数据
                    relationships_file = question_dir / "relationships.json"
                    if relationships_file.exists():
                        with open(relationships_file, 'r', encoding='utf-8') as f:
                            relationships_json = json.load(f)
                            # 转换键回元组格式
                            relationships = {}
                            for key, value in relationships_json.items():
                                src, tgt = key.split("___")
                                relationships[(src, tgt)] = value
                            self.question_metadata[question_id]["relationships"] = relationships
                    
                except Exception as e:
                    print(f"Error loading question graph from {question_dir}: {e}")
                    continue
        
        print(f"Loaded {len(self.question_graphs)} question graphs")
    
    def find_relevant_question_graphs(self, query: str, top_k: int = 5) -> List[int]:
        """找到与查询最相关的问题图谱"""
        if not self.question_metadata:
            return []
        
        # 计算查询与每个问题的相似性
        query_embedding = self.embedding_model.encode([query])
        similarities = []
        
        for question_id, metadata in self.question_metadata.items():
            question_text = metadata.get("question", "")
            
            # 构建问题的表示（问题+实体信息）
            question_repr = question_text
            if "entities" in metadata:
                entity_names = list(metadata["entities"].keys())
                if entity_names:
                    question_repr += " " + " ".join(entity_names[:10])  # 限制实体数量
            
            # 计算相似性
            question_embedding = self.embedding_model.encode([question_repr])
            similarity = cosine_similarity(query_embedding, question_embedding)[0][0]
            similarities.append((similarity, question_id))
        
        # 排序并返回top_k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_question_ids = [question_id for _, question_id in similarities[:top_k]]
        
        return top_question_ids
    
    def extract_entities_from_query(self, query: str, question_id: int) -> List[str]:
        """从查询中提取与特定问题图谱相关的实体"""
        if question_id not in self.question_metadata or "entities" not in self.question_metadata[question_id]:
            return []
        
        entities = self.question_metadata[question_id]["entities"]
        query_lower = query.lower()
        found_entities = []
        
        # 直接匹配
        for entity_name in entities.keys():
            if entity_name in query_lower:
                found_entities.append(entity_name)
        
        # 如果没有直接匹配，使用向量相似性查找
        if not found_entities:
            entity_texts = []
            entity_names = []
            
            for name, entity_data in entities.items():
                # 构建实体文本
                content_parts = [entity_data['entity_name']]
                if entity_data.get('entity_type'):
                    content_parts.append(f"{entity_data['entity_type']}")
                if entity_data.get('description'):
                    content_parts.append(f"{entity_data['description']}")
                entity_text = ": ".join(content_parts)
                
                entity_texts.append(entity_text)
                entity_names.append(name)
            
            if entity_texts:
                query_embedding = self.embedding_model.encode([query])
                entity_embeddings = self.embedding_model.encode(entity_texts)
                similarities = cosine_similarity(query_embedding, entity_embeddings)[0]
                
                # 获取最相似的实体
                top_indices = np.argsort(similarities)[-3:][::-1]  # 取前3个
                for idx in top_indices:
                    if similarities[idx] > 0.3:  # 相似性阈值
                        found_entities.append(entity_names[idx])
        
        return found_entities
    
    def local_search_in_question_graph(self, query: str, question_id: int) -> List[str]:
        """在特定问题图谱中进行本地搜索"""
        if question_id not in self.question_graphs:
            return []
        
        graph = self.question_graphs[question_id]
        metadata = self.question_metadata.get(question_id, {})
        entities = metadata.get("entities", {})
        relationships = metadata.get("relationships", {})
        
        # 提取种子实体
        seed_entities = self.extract_entities_from_query(query, question_id)
        
        if not seed_entities:
            # 如果没有找到种子实体，返回图中所有实体的简要信息
            context_parts = []
            for node in list(graph.nodes())[:self.top_k]:
                if node in entities:
                    entity = entities[node]
                    entity_text = f"Entity: {entity['entity_name']}"
                    if entity.get('entity_type'):
                        entity_text += f" (Type: {entity['entity_type']})"
                    if entity.get('description'):
                        entity_text += f" - {entity['description'][:100]}..."
                    context_parts.append(entity_text)
            return context_parts
        
        # k跳搜索
        relevant_entities = set(seed_entities)
        relevant_relationships = []
        
        current_entities = set(seed_entities)
        for depth in range(self.local_search_depth):
            next_entities = set()
            
            for entity in current_entities:
                if entity in graph:
                    neighbors = list(graph.neighbors(entity))
                    next_entities.update(neighbors)
                    
                    # 收集相关关系
                    for neighbor in neighbors:
                        edge_key = tuple(sorted([entity, neighbor]))
                        if edge_key in relationships:
                            relevant_relationships.append(relationships[edge_key])
            
            relevant_entities.update(next_entities)
            current_entities = next_entities
            
            if not current_entities:
                break
        
        # 构建上下文
        context_parts = []
        
        # 添加实体信息
        entity_contexts = []
        for entity_name in relevant_entities:
            if entity_name in entities:
                entity = entities[entity_name]
                entity_context = {
                    "entity_name": entity['entity_name'],
                    "entity_type": entity.get('entity_type', ''),
                    "description": entity.get('description', ''),
                    "rank": graph.degree(entity_name) if entity_name in graph else 0
                }
                entity_contexts.append(entity_context)
        
        # 按degree排序
        entity_contexts.sort(key=lambda x: x["rank"], reverse=True)
        
        # 截断到token限制
        entity_contexts = truncate_list_by_token_size(
            entity_contexts,
            key=lambda x: x["description"],
            max_token_size=self.max_tokens // 2
        )
        
        for entity_ctx in entity_contexts:
            entity_text = f"Entity: {entity_ctx['entity_name']}"
            if entity_ctx.get('entity_type'):
                entity_text += f" (Type: {entity_ctx['entity_type']})"
            if entity_ctx.get('description'):
                entity_text += f" - {entity_ctx['description']}"
            context_parts.append(entity_text)
        
        # 添加关系信息
        relationship_contexts = []
        for rel in relevant_relationships:
            rel_context = {
                "src_id": rel['src_id'],
                "tgt_id": rel['tgt_id'],
                "description": rel.get('description', ''),
                "weight": rel.get('weight', 1.0),
                "rank": (graph.degree(rel['src_id']) if rel['src_id'] in graph else 0) + 
                       (graph.degree(rel['tgt_id']) if rel['tgt_id'] in graph else 0)
            }
            relationship_contexts.append(rel_context)
        
        # 按rank和weight排序
        relationship_contexts.sort(key=lambda x: (x["rank"], x["weight"]), reverse=True)
        
        # 截断关系
        relationship_contexts = truncate_list_by_token_size(
            relationship_contexts,
            key=lambda x: x["description"],
            max_token_size=self.max_tokens // 2
        )
        
        for rel_ctx in relationship_contexts:
            rel_text = f"Relationship: {rel_ctx['src_id']} -> {rel_ctx['tgt_id']}"
            if rel_ctx.get('description'):
                rel_text += f" - {rel_ctx['description']}"
            context_parts.append(rel_text)
        
        return context_parts
    
    def multi_question_search(self, query: str, max_questions: int = 3) -> List[str]:
        """跨多个问题图谱进行搜索"""
        # 找到最相关的问题图谱
        relevant_question_ids = self.find_relevant_question_graphs(query, max_questions)
        
        if not relevant_question_ids:
            return ["No relevant question graphs found."]
        
        context_parts = []
        
        for i, question_id in enumerate(relevant_question_ids):
            # 添加问题信息
            question_text = self.question_metadata.get(question_id, {}).get("question", f"Question {question_id}")
            context_parts.append(f"## Relevant Question {i+1}: {question_text}")
            
            # 在该问题图谱中搜索
            question_context = self.local_search_in_question_graph(query, question_id)
            if question_context:
                context_parts.extend(question_context[:5])  # 限制每个问题的上下文数量
            else:
                context_parts.append("No relevant context found in this question graph.")
            
            context_parts.append("")  # 添加空行分隔
        
        return context_parts
    
    def search_and_format(self, query: str, search_mode: str = "multi_question", max_questions: int = 3) -> str:
        """搜索并格式化结果"""
        if search_mode == "single_best":
            # 找到最相关的单个问题图谱
            relevant_question_ids = self.find_relevant_question_graphs(query, 1)
            if not relevant_question_ids:
                return "No relevant question graphs found."
            
            question_id = relevant_question_ids[0]
            question_text = self.question_metadata.get(question_id, {}).get("question", f"Question {question_id}")
            context = self.local_search_in_question_graph(query, question_id)
            
            result = f"# Most Relevant Question: {question_text}\n\n"
            if context:
                for item in context:
                    result += f"{item}\n"
            else:
                result += "No relevant context found in the most relevant question graph."
            
            return result
        
        elif search_mode == "multi_question":
            # 跨多个问题图谱搜索
            context = self.multi_question_search(query, max_questions)
            
            result = f"# Multi-Question Graph Search Results\n\n"
            for item in context:
                result += f"{item}\n"
            
            return result if result.strip() else "No relevant context found across question graphs."
        
        else:
            raise ValueError(f"Unknown search mode: {search_mode}")
    
    def get_question_graph_statistics(self) -> Dict:
        """获取问题图谱统计信息"""
        stats = {
            "total_questions": len(self.question_graphs),
            "questions_with_nodes": 0,
            "questions_with_edges": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "avg_nodes_per_question": 0,
            "avg_edges_per_question": 0
        }
        
        node_counts = []
        edge_counts = []
        
        for question_id, graph in self.question_graphs.items():
            num_nodes = graph.number_of_nodes()
            num_edges = graph.number_of_edges()
            
            if num_nodes > 0:
                stats["questions_with_nodes"] += 1
            if num_edges > 0:
                stats["questions_with_edges"] += 1
            
            stats["total_nodes"] += num_nodes
            stats["total_edges"] += num_edges
            
            node_counts.append(num_nodes)
            edge_counts.append(num_edges)
        
        if stats["total_questions"] > 0:
            stats["avg_nodes_per_question"] = stats["total_nodes"] / stats["total_questions"]
            stats["avg_edges_per_question"] = stats["total_edges"] / stats["total_questions"]
        
        stats["node_distribution"] = {
            "min": min(node_counts) if node_counts else 0,
            "max": max(node_counts) if node_counts else 0,
            "median": np.median(node_counts) if node_counts else 0
        }
        
        stats["edge_distribution"] = {
            "min": min(edge_counts) if edge_counts else 0,
            "max": max(edge_counts) if edge_counts else 0,
            "median": np.median(edge_counts) if edge_counts else 0
        }
        
        return stats

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs_dir", required=True, help="Directory containing question graph subdirectories")
    parser.add_argument("--query", required=True, help="Query to search for")
    parser.add_argument("--search_mode", choices=["single_best", "multi_question"], default="multi_question", help="Search mode")
    parser.add_argument("--max_questions", type=int, default=3, help="Maximum number of question graphs to search")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--max_tokens", type=int, default=8000, help="Maximum tokens in context")
    parser.add_argument("--stats", action="store_true", help="Show question graph statistics")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 初始化检索器
    retriever = QuestionGraphRetriever(
        graphs_dir=args.graphs_dir,
        embedding_model=args.embedding_model,
        max_tokens=args.max_tokens
    )
    
    # 显示统计信息
    if args.stats:
        stats = retriever.get_question_graph_statistics()
        print("="*80)
        print("Question Graph Statistics")
        print("="*80)
        print(f"Total questions: {stats['total_questions']}")
        print(f"Questions with nodes: {stats['questions_with_nodes']}")
        print(f"Questions with edges: {stats['questions_with_edges']}")
        print(f"Average nodes per question: {stats['avg_nodes_per_question']:.2f}")
        print(f"Average edges per question: {stats['avg_edges_per_question']:.2f}")
        print(f"Node count range: {stats['node_distribution']['min']} - {stats['node_distribution']['max']} (median: {stats['node_distribution']['median']:.1f})")
        print(f"Edge count range: {stats['edge_distribution']['min']} - {stats['edge_distribution']['max']} (median: {stats['edge_distribution']['median']:.1f})")
        print("="*80)
    
    # 执行搜索
    result = retriever.search_and_format(args.query, args.search_mode, args.max_questions)
    
    print("="*80)
    print(f"Query: {args.query}")
    print(f"Search Mode: {args.search_mode}")
    print(f"Max Questions: {args.max_questions}")
    print("="*80)
    print(result)

if __name__ == "__main__":
    main()