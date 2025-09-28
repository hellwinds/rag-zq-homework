# 01-RAGAS-offline.py
import os
import logging
import json
import numpy as np
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# 自定义本地 LLM 和 Embedding（适配 RAGAS）
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from milvus_model.dense import SentenceTransformerEmbeddingFunction
from sqlalchemy import create_engine, text
from typing import ClassVar  # ← 新增导入
from ragas.run_config import RunConfig

run_config = RunConfig(
    max_workers=1,        # 禁用多线程
    timeout=300,          # 延长超时时间（秒）
)

# 日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==============================
# 1. 本地嵌入模型（BGE）
# ==============================
class LocalBGEEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        self.model = SentenceTransformerEmbeddingFunction(model_name=model_path, device="cpu")
    
    def embed_documents(self, texts):
        return self.model(texts)
    
    def embed_query(self, text):
        return self.model([text])[0]

# ==============================
# 2. 本地 LLM（Qwen2.5-0.5B）
# ==============================
class LocalQwenLLM(LLM):
    tokenizer: ClassVar = None
    model: ClassVar = None

    def __init__(self, model_path: str):
        super().__init__()
        if LocalQwenLLM.tokenizer is None:
            logging.info("[LLM] 首次加载 Qwen2.5-0.5B 模型...")
            LocalQwenLLM.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            LocalQwenLLM.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu",
                torch_dtype="auto",
                trust_remote_code=True
            )
    
    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        """
        必须支持 stop, run_manager, **kwargs，否则 RAGAS/LangChain 会报错
        """
        messages = [{"role": "user", "content": prompt}]
        text = LocalQwenLLM.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = LocalQwenLLM.tokenizer([text], return_tensors="pt").to("cpu")
        generated_ids = LocalQwenLLM.model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=LocalQwenLLM.tokenizer.eos_token_id
        )
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = LocalQwenLLM.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()

    @property
    def _llm_type(self):
        return "qwen2.5-0.5b"
# ==============================
# 3. 初始化组件
# ==============================
logging.info("[初始化] 加载本地模型...")
embedding_model = LocalBGEEmbeddings(r".\model")
llm_model = LocalQwenLLM(r".\qwen_1.5b")

# 包装为 RAGAS 兼容格式
llm = LangchainLLMWrapper(llm_model)
embeddings = LangchainEmbeddingsWrapper(embedding_model)

# ==============================
# 4. 数据库连接（用于生成 ground truth）
# ==============================
DB_URL = "mysql+pymysql://root:Rzxgh2025@10.128.12.214:3306/sakila"
engine = create_engine(DB_URL)

# ==============================
# 5. 检索函数（Chroma）
# ==============================
import chromadb
chroma_client = chromadb.PersistentClient(path="./chroma_ddl_db")
ddl_collection = chroma_client.get_collection("ddl_knowledge")

q2sql_client = chromadb.PersistentClient(path="./chroma_q2sql_db")
q2sql_collection = q2sql_client.get_collection("q2sql_knowledge")

dbdesc_client = chromadb.PersistentClient(path="./chroma_dbdesc_db")
dbdesc_collection = dbdesc_client.get_collection("dbdesc_knowledge")

def retrieve_context(question: str):
    # 检索三类上下文
    ddl_res = ddl_collection.query(query_texts=[question], n_results=3, include=["documents"])
    q2sql_res = q2sql_collection.query(query_texts=[question], n_results=3, include=["documents", "metadatas"])
    desc_res = dbdesc_collection.query(query_texts=[question], n_results=5, include=["documents", "metadatas"])
    
    ddl_ctx = "\n".join(ddl_res["documents"][0]) if ddl_res["documents"][0] else ""
    q2sql_ctx = "\n".join([
        f"NL: \"{doc}\"\nSQL: \"{meta.get('sql_text', '')}\""
        for doc, meta in zip(q2sql_res["documents"][0], q2sql_res["metadatas"][0])
    ]) if q2sql_res["documents"][0] else ""
    desc_ctx = "\n".join([
        f"{meta.get('table_name', '')}.{meta.get('column_name', '')}: {doc}"
        for doc, meta in zip(desc_res["documents"][0], desc_res["metadatas"][0])
    ]) if desc_res["documents"][0] else ""
    
    full_context = f"{ddl_ctx}\n{desc_ctx}\n{q2sql_ctx}".strip()
    return full_context.split("\n")  # RAGAS 要求 list[str]

# ==============================
# 6. 生成答案（调用本地 LLM）
# ==============================
def generate_answer(question: str, contexts: list) -> str:
    context_str = "\n".join(contexts)
    prompt = (
        f"### Schema Definitions:\n{context_str}\n"
        f"### Query:\n\"{question}\"\n"
        "请只返回SQL语句，不要包含任何解释或说明。"
    )
    return llm_model._call(prompt)

# ==============================
# 7. 构建评估数据集
# ==============================
# 从 q2sql_pairs.json 加载测试问题
with open("q2sql_pairs.json", "r", encoding="utf-8") as f:
    pairs = json.load(f)

questions = []
ground_truths = []
answers = []
contexts_list = []

for pair in pairs[:5]:  # 可调整数量，避免评估太慢
    q = pair["question"]
    gt_sql = pair["sql"]
    
    # 获取上下文
    ctx = retrieve_context(q)
    
    # 生成答案
    ans = generate_answer(q, ctx)
    
    questions.append(q)
    ground_truths.append(gt_sql)
    answers.append(ans)
    contexts_list.append(ctx)

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts_list,
    "ground_truth": ground_truths  # RAGAS 可选，用于 answer_correctness（本脚本未用）
})

# ==============================
# 8. 执行评估
# ==============================
print("\n=== RAGAS 离线评估（本地 BGE + 本地 Qwen） ===")

# Faithfulness（忠实度）
print("\n[1/2] 评估 Faithfulness（忠实度）...")
faithfulness_metric = [Faithfulness(llm=llm)]
faithfulness_result = evaluate(dataset, faithfulness_metric, run_config=run_config)
faith_score = np.mean(faithfulness_result['faithfulness'])
print(f"✅ Faithfulness: {faith_score:.4f}")

# AnswerRelevancy（答案相关性）
print("\n[2/2] 评估 AnswerRelevancy（答案相关性）...")
relevancy_metric = [AnswerRelevancy(llm=llm, embeddings=embeddings)]
relevancy_result = evaluate(dataset, relevancy_metric, run_config=run_config)
rel_score = np.mean(relevancy_result['answer_relevancy'])
print(f"✅ AnswerRelevancy: {rel_score:.4f}")

print("\n=== 评估完成 ===")
print(f"综合得分: Faithfulness={faith_score:.4f}, Relevancy={rel_score:.4f}")

# 可选：保存详细结果
import pandas as pd
df = dataset.to_pandas()
df["faithfulness"] = faithfulness_result["faithfulness"]
df["answer_relevancy"] = relevancy_result["answer_relevancy"]
df.to_csv("ragas_evaluation_offline_results.csv", index=False, encoding="utf-8")
print("\n📊 详细结果已保存至: ragas_evaluation_offline_results.csv")
