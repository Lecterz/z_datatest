import re
import json
import time
from collections import defaultdict
from typing import Union, List, Any
import numpy as np
from datasets import load_dataset
import argparse
from pathlib import Path
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 基于lecterz的常量定义
GRAPH_FIELD_SEP = "<SEP>"
DEFAULT_ENTITY_TYPES = ["organization", "person", "geo", "event"]
DEFAULT_TUPLE_DELIMITER = "<|>"
DEFAULT_RECORD_DELIMITER = "##"
DEFAULT_COMPLETION_DELIMITER = "<|COMPLETE|>"

# 修复后的prompt模板 - 使用参考项目的标准格式
ENTITY_EXTRACTION = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. "If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us."

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
#############################
Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
#############################
Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

# Gleaning prompts基于lecterz
ENTITY_CONTINUE_EXTRACTION = """MANY entities were missed in the last extraction. Add them below using the same format:"""
ENTITY_IF_LOOP_EXTRACTION = """It appears some entities may have still been missed. Answer YES | NO if there are still entities that need to be added."""

def clean_str(input_str: str) -> str:
    """基于lecterz的字符串清理函数"""
    if not isinstance(input_str, str):
        return input_str
    
    import html
    result = html.unescape(input_str.strip())
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
    return re.sub('[^A-Za-z0-9 ]', ' ', result.lower()).strip()

def split_string_by_multi_markers(text: str, delimiters: List[str]) -> List[str]:
    """基于lecterz的多分隔符分割函数"""
    if not delimiters:
        return [text]
    split_pattern = "|".join(re.escape(delimiter) for delimiter in delimiters)
    segments = re.split(split_pattern, text)
    return [segment.strip() for segment in segments if segment.strip()]

def is_float_regex(value: str) -> bool:
    """基于lecterz的浮点数检测函数"""
    return bool(re.match(r"^[-+]?[0-9]*\.?[0-9]+$", value))

def parse_triples_enhanced(raw_text, chunk_key):
    """修复后的三元组解析，匹配标准prompt格式"""
    triples = []
    raw_text = raw_text.strip()

    if DEFAULT_COMPLETION_DELIMITER in raw_text:
        raw_text = raw_text.split(DEFAULT_COMPLETION_DELIMITER)[0]

    # 分割记录 - 支持多种分隔符
    records = split_string_by_multi_markers(raw_text, [
        DEFAULT_RECORD_DELIMITER,  # ##
        DEFAULT_COMPLETION_DELIMITER,  # <|COMPLETE|>
        "\n"  # 换行符作为备用分隔符
    ])
    
    print(f"Debug: Found {len(records)} records")
    
    for i, record in enumerate(records):
        record = record.strip()
        if not record:
            continue
            
        print(f"Debug record {i}: {record[:100]}...")
        
        # 方法1: 标准格式 ("entity"<|>name<|>type<|>desc)
        match = re.search(r'\("([^"]+)"[<|>]+([^<|>]+)[<|>]+([^<|>]+)[<|>]+([^<|>)]+)\)', record)
        if match:
            record_type = clean_str(match.group(1))
            if record_type == "entity":
                entity_name = clean_str(match.group(2))
                if entity_name:
                    entity = {
                        "type": "entity",
                        "name": entity_name,
                        "entity_type": clean_str(match.group(3)),
                        "desc": clean_str(match.group(4)),
                    }
                    triples.append(entity)
                    print(f"  -> Parsed entity (method 1): {entity_name}")
            continue
        
        # 方法2: 简单格式 entity|name|type|desc
        if record.startswith(("entity", "Entity")):
            parts = re.split(r'[|<>]+', record)
            if len(parts) >= 4:
                entity_name = clean_str(parts[1])
                if entity_name:
                    entity = {
                        "type": "entity",
                        "name": entity_name,
                        "entity_type": clean_str(parts[2]),
                        "desc": clean_str(" ".join(parts[3:])),  # 合并剩余部分作为描述
                    }
                    triples.append(entity)
                    print(f"  -> Parsed entity (method 2): {entity_name}")
                continue
        
        # 方法3: 标准格式但没有括号 "entity"<|>name<|>type<|>desc
        if '"entity"' in record or "'entity'" in record:
            # 移除引号
            clean_record = record.replace('"', '').replace("'", "")
            parts = re.split(r'[<|>]+', clean_record)
            if len(parts) >= 4:
                entity_name = clean_str(parts[1])
                if entity_name:
                    entity = {
                        "type": "entity",
                        "name": entity_name,
                        "entity_type": clean_str(parts[2]),
                        "desc": clean_str(" ".join(parts[3:])),
                    }
                    triples.append(entity)
                    print(f"  -> Parsed entity (method 3): {entity_name}")
                continue
        
        # 方法4: 关系解析 - 标准5字段格式
        # 标准格式: ("relationship"<|>src<|>tgt<|>desc<|>strength)
        match = re.search(r'\("relationship"[<|>]+([^<|>]+)[<|>]+([^<|>]+)[<|>]+([^<|>]+)[<|>]+([^<|>)]+)\)', record)
        if match:
            src_id = clean_str(match.group(1))
            tgt_id = clean_str(match.group(2))
            if src_id and tgt_id:
                weight = float(match.group(4)) if is_float_regex(match.group(4)) else 1.0
                relationship = {
                    "type": "relation",
                    "head": src_id,
                    "tail": tgt_id,
                    "desc": clean_str(match.group(3)),
                    "score": weight,
                }
                triples.append(relationship)
                print(f"  -> Parsed relationship (standard): {src_id} -> {tgt_id}")
            continue
            
        # 简单关系格式: relationship|src|tgt|desc|strength
        if record.startswith(("relationship", "Relationship")):
            parts = re.split(r'[|<>]+', record)
            if len(parts) >= 4:
                src_id = clean_str(parts[1])
                tgt_id = clean_str(parts[2])
                if src_id and tgt_id:
                    weight = 1.0
                    if len(parts) >= 5 and is_float_regex(parts[4]):
                        weight = float(parts[4])
                    
                    relationship = {
                        "type": "relation",
                        "head": src_id,
                        "tail": tgt_id,
                        "desc": clean_str(parts[3]) if len(parts) > 3 else "",
                        "score": weight,
                    }
                    triples.append(relationship)
                    print(f"  -> Parsed relationship (simple): {src_id} -> {tgt_id}")
                continue
        
        # 方法5: 模糊匹配 - 尝试从任何包含实体信息的行中提取
        if any(keyword in record.lower() for keyword in ["person", "organization", "location", "event", "geo"]):
            # 尝试提取实体信息
            parts = re.split(r'[|<>\-\s]+', record)
            if len(parts) >= 2:
                potential_name = clean_str(parts[0]) or clean_str(parts[1])
                if potential_name and len(potential_name) > 1:
                    # 猜测实体类型
                    entity_type = "unknown"
                    for keyword in ["person", "organization", "location", "event", "geo"]:
                        if keyword in record.lower():
                            entity_type = keyword
                            break
                    
                    entity = {
                        "type": "entity",
                        "name": potential_name,
                        "entity_type": entity_type,
                        "desc": clean_str(" ".join(parts[2:]) if len(parts) > 2 else record[:100]),
                    }
                    triples.append(entity)
                    print(f"  -> Parsed entity (fuzzy): {potential_name}")
    
    print(f"Debug: Total parsed triples: {len(triples)}")
    return triples

def enhanced_extraction_with_gleaning(llm, sampling_params, input_text, max_gleaning=2):
    """增强提取，包含gleaning机制"""
    # 构建初始prompt
    context = {
        "tuple_delimiter": DEFAULT_TUPLE_DELIMITER,
        "record_delimiter": DEFAULT_RECORD_DELIMITER,
        "completion_delimiter": DEFAULT_COMPLETION_DELIMITER,
        "entity_types": ",".join(DEFAULT_ENTITY_TYPES),
        "input_text": input_text
    }
    
    initial_prompt = ENTITY_EXTRACTION.format(**context)
    
    # 初始提取
    generation = llm.generate([initial_prompt], sampling_params)[0]
    result_text = generation.outputs[0].text
    
    # Gleaning过程 - 基于lecterz的多轮提取
    for glean_idx in range(max_gleaning):
        continue_prompt = initial_prompt + result_text + "\n\n" + ENTITY_CONTINUE_EXTRACTION
        glean_generation = llm.generate([continue_prompt], sampling_params)[0]
        glean_result = glean_generation.outputs[0].text
        result_text += glean_result
        
        if glean_idx == max_gleaning - 1:
            break
            
        # 检查是否需要继续 - 基于lecterz的循环检测
        loop_prompt = continue_prompt + glean_result + "\n\n" + ENTITY_IF_LOOP_EXTRACTION
        loop_generation = llm.generate([loop_prompt], sampling_params)[0]
        loop_result = loop_generation.outputs[0].text
        
        if "no" in loop_result.lower() or "n" in loop_result.lower():
            break
    
    return result_text

# 完全保持feihm原版的build_prompt函数
def build_prompt(example, tokenizer):
    """把 NQ 单条样本转成 chat-prompt 字符串 - 完全按照feihm原版"""
    ctxs = example["ctxs"]
    snippet_tpl = "# Title\n{title}\n\n## Text\n{text}"
    # 拼接 passage 文本
    passages = "\n\n".join(
        snippet_tpl.format(title=c.get("title", "<title>"), text=c.get("text", "<text>"))
        for c in ctxs
    )
    return passages

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True, help="Path to nq.json")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--world_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--dest_dir", required=True)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_gleaning", type=int, default=0, help="Maximum gleaning rounds")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 完全按照feihm原版的数据处理流程
    print("[+] Loading dataset ...")
    ds = load_dataset("json", data_files=args.data_file, split="train").select(range(10))
    print(f"    Dataset size = {len(ds)}")

    print("[+] Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=False)

    # 完全按照feihm原版的prompt预处理
    print("[+] Building prompts in parallel ...")
    ds = ds.map(
        lambda ex: {
            "prompt": build_prompt(ex, tokenizer),
            "question": ex["question"],
            "ctxs": ex["ctxs"],
            "answers": ex["answers"]
        },
        num_proc=args.num_proc,
        desc="Generate prompt",
    )
    print(ds)

    # 完全按照feihm原版的LLM初始化
    print("[+] Loading vLLM model ...")
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.world_size,
        trust_remote_code=False,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=8192,
        max_num_seqs=256,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
    )

    # 完全按照feihm原版的输出准备
    out_path = Path(args.dest_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_path / "enhanced_extraction_results.jsonl"

    print("[+] Start enhanced generation with gleaning...")
    generation_start = time.time()
    
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(ds), args.batch_size), desc="Enhanced extraction batches"):
            batch = ds[i : i + args.batch_size]
            prompts = batch["prompt"]
            questions = batch["question"]
            ctxs_list = batch["ctxs"] 
            answers = batch["answers"]

            # 对每个prompt进行增强提取（包含gleaning）
            for j, (question, ctxs, prompt, answer) in enumerate(zip(questions, ctxs_list, prompts, answers)):
                print(f"\n🔍 Processing Question {i + j + 1}/{len(ds)}")
                print(f"   Question: {question[:80]}...")
                
                # 使用增强提取方法
                enhanced_output = enhanced_extraction_with_gleaning(
                    llm, sampling, prompt, args.max_gleaning
                )
                
                print(f"Raw LLM output for question {i+j}:")
                print(enhanced_output)  # 打印前500字符
                print("="*50)
                
                # 解析三元组 - 使用修复后的解析
                chunk_key = f"chunk_{i + j}"
                triples = parse_triples_enhanced(enhanced_output, chunk_key)
                
                print(f"   ✅ Extracted {len([t for t in triples if t['type'] == 'entity'])} entities, {len([t for t in triples if t['type'] == 'relation'])} relations")

                # 完全按照feihm原版的输出格式
                item = {
                    "question": question,
                    "passages": ctxs,
                    "triples": triples,
                    "answers": answer,
                }
                fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    generation_end = time.time()
    print(f"[+] Enhanced generation completed and spent {generation_end-generation_start:.2f} seconds!")
    print(f"[✓] Saved enhanced structured triples to {jsonl_path}")

if __name__ == "__main__":
    main()