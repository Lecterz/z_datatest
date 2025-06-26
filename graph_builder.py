import json
import pickle
import asyncio
import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from pathlib import Path
import argparse
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import community as community_louvain
from typing import List, Dict

# 基于lecterz的常量和工具函数
GRAPH_FIELD_SEP = "<SEP>"

class MergeEntity:
    """基于lecterz的实体合并类"""
    
    @staticmethod
    def merge_source_ids(existing_source_ids: List[str], new_source_ids: List[str]):
        merged_source_ids = list(set(new_source_ids) | set(existing_source_ids))
        return GRAPH_FIELD_SEP.join(merged_source_ids)

    @staticmethod
    def merge_types(existing_entity_types: List[str], new_entity_types: List[str]):
        # 使用最频繁的实体类型
        merged_entity_types = existing_entity_types + new_entity_types
        entity_type_counts = Counter(merged_entity_types)
        most_common_type = entity_type_counts.most_common(1)[0][0] if entity_type_counts else ''
        return most_common_type

    @staticmethod
    def merge_descriptions(existing_descriptions: List[str], new_descriptions: List[str]):
        merged_descriptions = list(set(new_descriptions) | set(existing_descriptions))
        description = GRAPH_FIELD_SEP.join(sorted(merged_descriptions))
        return description

class MergeRelationship:
    """基于lecterz的关系合并类"""
    
    @staticmethod
    def merge_weight(existing_weights: List[float], new_weights: List[float]):
        return sum(new_weights + existing_weights)

    @staticmethod
    def merge_descriptions(existing_descriptions: List[str], new_descriptions: List[str]):
        return GRAPH_FIELD_SEP.join(
            sorted(set(new_descriptions + existing_descriptions))
        )

    @staticmethod
    def merge_source_ids(existing_source_ids: List[str], new_source_ids: List[str]):
        return GRAPH_FIELD_SEP.join(
            set(new_source_ids + existing_source_ids)
        )

    @staticmethod
    def merge_keywords(existing_keywords: List[str], new_keywords: List[str]):
        return GRAPH_FIELD_SEP.join(
            set(existing_keywords + new_keywords)
        )

    @staticmethod
    def merge_relation_name(existing_names: List[str], new_names: List[str]):
        return GRAPH_FIELD_SEP.join(
            sorted(set(existing_names + new_names))
        )

class QuestionGraphBuilder:
    """为每个问题单独构建知识图谱"""
    
    def __init__(self, 
                 similarity_threshold: float = 0.7,
                 similarity_top_k: int = 10,
                 enable_community_detection: bool = True,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        
        self.similarity_threshold = similarity_threshold
        self.similarity_top_k = similarity_top_k
        self.enable_community_detection = enable_community_detection
        
        # 嵌入模型
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)
        
    def load_extraction_results(self, file_path: str):
        """加载提取结果"""
        print("Loading extraction results...")
        extraction_data = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading data"):
                data = json.loads(line)
                extraction_data.append(data)
        
        return extraction_data
    
    def build_single_question_graph(self, question_data: dict) -> dict:
        """为单个问题构建知识图谱"""
        extraction_result = question_data["extraction_result"]
        question = question_data["question"]
        
        # 初始化该问题的图
        graph = nx.Graph()
        entities = {}
        relationships = {}
        
        # 处理实体
        entity_groups = defaultdict(list)
        for entity in extraction_result["entities"]:
            entity_name = entity["entity_name"].lower().strip()
            entity_groups[entity_name].append(entity)
        
        # 合并同名实体
        for entity_name, entity_list in entity_groups.items():
            merged_entity = self._merge_entities_lecterz_style(entity_list)
            entities[entity_name] = merged_entity
            graph.add_node(entity_name, **merged_entity)
        
        # 处理关系
        relationship_groups = defaultdict(list)
        for relationship in extraction_result["relationships"]:
            src = relationship["src_id"].lower().strip()
            tgt = relationship["tgt_id"].lower().strip()
            
            # 确保实体存在
            if src in entities and tgt in entities:
                key = tuple(sorted([src, tgt]))
                relationship_groups[key].append(relationship)
        
        # 合并关系
        for (src, tgt), relationship_list in relationship_groups.items():
            merged_relationship = self._merge_relationships_lecterz_style(relationship_list)
            relationships[(src, tgt)] = merged_relationship
            graph.add_edge(src, tgt, **merged_relationship)
        
        # 添加相似性边（在单个问题内部）
        if len(entities) > 1:
            self._add_similarity_edges_to_graph(graph, entities)
        
        # 社区检测
        if self.enable_community_detection and len(graph.nodes()) > 1:
            self._detect_communities_for_graph(graph)
        
        return {
            "question": question,
            "graph": graph,
            "entities": entities,
            "relationships": relationships,
            "question_data": question_data
        }
    
    def _merge_entities_lecterz_style(self, entity_list: list) -> dict:
        """基于lecterz的实体合并逻辑"""
        if len(entity_list) == 1:
            return entity_list[0]
        
        # 构建合并数据
        existing_data = defaultdict(list)
        for entity in entity_list:
            for key, value in entity.items():
                existing_data[key].append(value)
        
        # 使用lecterz的合并方法
        merged_source_ids = MergeEntity.merge_source_ids([], existing_data["source_id"])
        merged_entity_type = MergeEntity.merge_types([], existing_data["entity_type"])
        merged_description = MergeEntity.merge_descriptions([], existing_data["description"])
        
        return {
            "entity_name": entity_list[0]["entity_name"],
            "entity_type": merged_entity_type,
            "description": merged_description,
            "source_id": merged_source_ids,
            "occurrence_count": len(entity_list)
        }
    
    def _merge_relationships_lecterz_style(self, relationship_list: list) -> dict:
        """基于lecterz的关系合并逻辑"""
        if len(relationship_list) == 1:
            rel = relationship_list[0]
            rel["occurrence_count"] = 1
            return rel
        
        # 构建合并数据
        existing_data = defaultdict(list)
        for rel in relationship_list:
            for key, value in rel.items():
                existing_data[key].append(value)
        
        # 使用lecterz的合并方法
        merged_source_ids = MergeRelationship.merge_source_ids([], existing_data["source_id"])
        merged_descriptions = MergeRelationship.merge_descriptions([], existing_data["description"])
        merged_keywords = MergeRelationship.merge_keywords([], existing_data.get("keywords", []))
        merged_weight = MergeRelationship.merge_weight([], existing_data.get("weight", [1.0]))
        
        return {
            "src_id": relationship_list[0]["src_id"],
            "tgt_id": relationship_list[0]["tgt_id"],
            "description": merged_descriptions,
            "keywords": merged_keywords,
            "weight": merged_weight,
            "source_id": merged_source_ids,
            "occurrence_count": len(relationship_list)
        }
    
    def _add_similarity_edges_to_graph(self, graph: nx.Graph, entities: dict):
        """为单个图添加相似性边"""
        entity_names = list(entities.keys())
        if len(entity_names) < 2:
            return
        
        # 准备文本用于计算相似性
        entity_texts = []
        for name in entity_names:
            entity = entities[name]
            text = f"{entity['entity_name']} {entity.get('entity_type', '')} {entity.get('description', '')}"
            entity_texts.append(text)
        
        # 计算嵌入向量
        embeddings = self.embedding_model.encode(entity_texts)
        
        # 计算相似性矩阵
        similarity_matrix = cosine_similarity(embeddings)
        
        # 添加相似性边
        added_edges = 0
        for i in range(len(entity_names)):
            for j in range(i + 1, len(entity_names)):
                similarity = similarity_matrix[i][j]
                
                if similarity > self.similarity_threshold:
                    src = entity_names[i]
                    tgt = entity_names[j]
                    
                    # 检查是否已存在边
                    if not graph.has_edge(src, tgt):
                        edge_data = {
                            "src_id": src,
                            "tgt_id": tgt,
                            "description": f"Similarity-based connection (score: {similarity:.3f})",
                            "keywords": "similarity",
                            "weight": similarity,
                            "source_id": "similarity_detection",
                            "edge_type": "similarity"
                        }
                        
                        graph.add_edge(src, tgt, **edge_data)
                        added_edges += 1
        
        return added_edges
    
    def _detect_communities_for_graph(self, graph: nx.Graph):
        """为单个图进行社区检测"""
        try:
            communities = community_louvain.best_partition(graph, weight='weight')
            
            # 将社区信息添加到节点，格式参考lecterz
            for node, community_id in communities.items():
                cluster_info = [{
                    "cluster": community_id,
                    "level": 0  # 简化为单层
                }]
                graph.nodes[node]['clusters'] = json.dumps(cluster_info)
                
        except Exception as e:
            # 如果社区检测失败，继续处理
            pass
    
    def build_all_question_graphs(self, extraction_data: list, output_dir: str):
        """为所有问题构建独立的知识图谱"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        question_graphs = []
        
        print(f"Building {len(extraction_data)} individual question graphs...")
        
        for i, question_data in enumerate(tqdm(extraction_data, desc="Building question graphs")):
            try:
                # 为单个问题构建图
                graph_result = self.build_single_question_graph(question_data)
                question_graphs.append(graph_result)
                
                # 保存单个问题的图（可选）
                question_graph_dir = output_path / f"question_{i}"
                question_graph_dir.mkdir(exist_ok=True)
                
                # 保存图文件
                graph_file = question_graph_dir / "question_graph.gpickle"
                nx.write_gpickle(graph_result["graph"], graph_file)
                
                # 保存实体和关系
                entities_file = question_graph_dir / "entities.json"
                with open(entities_file, 'w', encoding='utf-8') as f:
                    json.dump(graph_result["entities"], f, indent=2, ensure_ascii=False)
                
                relationships_file = question_graph_dir / "relationships.json"
                relationships_json = {f"{k[0]}___{k[1]}": v for k, v in graph_result["relationships"].items()}
                with open(relationships_file, 'w', encoding='utf-8') as f:
                    json.dump(relationships_json, f, indent=2, ensure_ascii=False)
                
                # 保存问题元数据
                metadata_file = question_graph_dir / "metadata.json"
                metadata = {
                    "question": graph_result["question"],
                    "question_id": i,
                    "num_nodes": graph_result["graph"].number_of_nodes(),
                    "num_edges": graph_result["graph"].number_of_edges(),
                    "num_entities": len(graph_result["entities"]),
                    "num_relationships": len(graph_result["relationships"])
                }
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                    
            except Exception as e:
                print(f"Error building graph for question {i}: {e}")
                continue
        
        # 保存整体统计信息
        self._save_overall_statistics(question_graphs, output_path)
        
        return question_graphs
    
    def _save_overall_statistics(self, question_graphs: list, output_path: Path):
        """保存整体统计信息"""
        total_questions = len(question_graphs)
        total_nodes = sum(g["graph"].number_of_nodes() for g in question_graphs)
        total_edges = sum(g["graph"].number_of_edges() for g in question_graphs)
        
        stats = {
            "total_questions": total_questions,
            "total_nodes_across_all_graphs": total_nodes,
            "total_edges_across_all_graphs": total_edges,
            "avg_nodes_per_graph": total_nodes / total_questions if total_questions > 0 else 0,
            "avg_edges_per_graph": total_edges / total_questions if total_questions > 0 else 0,
            "graphs_with_nodes": sum(1 for g in question_graphs if g["graph"].number_of_nodes() > 0),
            "graphs_with_edges": sum(1 for g in question_graphs if g["graph"].number_of_edges() > 0)
        }
        
        # 节点和边数分布
        node_counts = [g["graph"].number_of_nodes() for g in question_graphs]
        edge_counts = [g["graph"].number_of_edges() for g in question_graphs]
        
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
        
        stats_file = output_path / "overall_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\nOverall Statistics:")
        print(f"Total questions processed: {stats['total_questions']}")
        print(f"Graphs with nodes: {stats['graphs_with_nodes']}")
        print(f"Graphs with edges: {stats['graphs_with_edges']}")
        print(f"Average nodes per graph: {stats['avg_nodes_per_graph']:.2f}")
        print(f"Average edges per graph: {stats['avg_edges_per_graph']:.2f}")
        print(f"Statistics saved to: {stats_file}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction_file", required=True, help="Path to extraction results JSONL file")
    parser.add_argument("--output_dir", required=True, help="Output directory for question graphs")
    parser.add_argument("--similarity_threshold", type=float, default=0.7, help="Similarity threshold for adding edges")
    parser.add_argument("--similarity_top_k", type=int, default=10, help="Top K similar entities")
    parser.add_argument("--enable_community", action="store_true", help="Enable community detection")
    parser.add_argument("--embedding_model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 初始化图构建器
    builder = QuestionGraphBuilder(
        similarity_threshold=args.similarity_threshold,
        similarity_top_k=args.similarity_top_k,
        enable_community_detection=args.enable_community,
        embedding_model=args.embedding_model
    )
    
    # 加载提取结果
    extraction_data = builder.load_extraction_results(args.extraction_file)
    
    # 为每个问题构建独立的图
    question_graphs = builder.build_all_question_graphs(extraction_data, args.output_dir)
    
    print(f"\n✅ Successfully built {len(question_graphs)} individual question graphs!")
    print(f"📁 Each question graph saved in separate subdirectories under: {args.output_dir}")

if __name__ == "__main__":
    main()