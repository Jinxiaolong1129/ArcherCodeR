import re
from habanero import Crossref

# 从文档中提取的论文标题和arXiv ID
papers = [
    ("The unreasonable effectiveness of entropy minimization in llm reasoning", "2505.15134"),
    ("Matharena: Evaluating llms on uncontaminated math competitions", "2505.23281"),
    ("Language models are few-shot learners", None),
    ("Weak-to-strong generalization: Eliciting strong capabilities with weak supervision", "2312.09390"),
    ("Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities", "2507.06261"),
    ("Reinforcement pre-training", "2506.08007"),
    ("Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars", "2503.01307"),
    ("Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning", "2501.12948"),
    ("Measuring mathematical problem solving with the math dataset", "2103.03874"),
    ("Open-reasoner-zero: An open source approach to scaling up reinforcement learning on the base model", "2503.24290"),
    ("Openai o1 system card", "2412.16720"),
    ("History-aware cross-attention reinforcement: Self-supervised multi turn and chain-of-thought fine-tuning with vllm", "2506.11108"),
    ("A self-supervised reinforcement learning approach for fine-tuning large language models using cross-attention signals", "2502.10482"),
    ("Tulu 3: Pushing frontiers in open language model post-training", "2411.15124"),
    ("Numinamath: The largest public dataset in ai4maths with 860k pairs of competition math problems and solutions", None),
    ("Confidence is all you need: Few-shot rl fine-tuning of language models", "2506.06395"),
    ("Ettrl: Balancing exploration and exploitation in llm test-time reinforcement learning via entropy mechanism", "2508.11356"),
    ("Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models", "2505.24864"),
    ("Deepscaler: Surpassing o1-preview with a 1.5 b model by scaling rl", None),
    ("Maximizing confidence alone improves reasoning", "2505.22660"),
    ("Exploring the limits of transfer learning with a unified text-to-text transformer", None),
    ("Can large reasoning models self-train?", "2505.21444"),
    ("Hybridflow: A flexible and efficient rlhf framework", None),
    ("Welcome to the era of experience", None),
    ("Ladder: Self-improving llms through recursive problem decomposition", "2503.00735"),
    ("Post-training large language models via reinforcement learning from self-feedback", "2507.21931"),
    ("Mirage or method? how model-task alignment induces divergent rl conclusions", "2508.21188"),
    ("Reasoning or memorization? unreliable results of reinforcement learning due to data contamination", "2507.10532"),
    ("Qwen3 technical report", "2505.09388"),
    ("Dapo: An open-source llm reinforcement learning system at scale", "2503.14476"),
    ("Wisdom of the crowd: Reinforcement learning from coevolutionary collective feedback", "2508.12338"),
    ("Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model?", "2504.13837"),
    ("Consistent paths lead to truth: Self-rewarding reinforcement learning for llm reasoning", "2506.08745"),
    ("Right question is already half the answer: Fully unsupervised llm reasoning incentivization", "2504.05812"),
    ("No free lunch: Rethinking internal feedback for llm reasoning", "2506.17219"),
    ("Co-reward: Self-supervised reinforcement learning for large language model reasoning via contrastive agreement", "2508.00410"),
    ("Absolute zero: Reinforced self-play reasoning with zero data", "2505.03335"),
    ("Learning to reason without external rewards", "2505.19590"),
    ("Ttrl: Test-time reinforcement learning", "2504.16084"),
    ("Self-adapting language models", "2506.10943"),
]

cr = Crossref()

print("=" * 80)
print("Paper Links")
print("=" * 80)

zotero_lines = []

for title, arxiv_id in papers:
    print(f"\nTitle: {title}")
    
    # 如果有arXiv ID，直接生成链接
    if arxiv_id:
        print(f"  arXiv: https://arxiv.org/abs/{arxiv_id}")
    
    # 尝试从Crossref获取DOI
    try:
        result = cr.works(query_title=title, limit=1)
        if result['message']['items']:
            item = result['message']['items'][0]
            doi = item.get('DOI')
            url = item.get('URL')
            if doi:
                print(f"  DOI: {doi}")
                zotero_lines.append(f"{doi} https://doi.org/{doi}")
            if url:
                print(f"  URL: {url}")
    except Exception as e:
        print(f"  Crossref error: {e}")

print("\n")
print("=" * 80)
print("Zotero Import (copy below)")
print("=" * 80)
for line in zotero_lines:
    print(line)