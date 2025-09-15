import streamlit as st
import json
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict, Counter

# 思考与推理 (Thinking and Reasoning)：主动认知过程，如分析、推断、假设。Token数量：25个
thinking_and_reasoning = ["analyze", "analyzing", "analysis", "cogitate", "cogitation", "conclude", "conclusion", "deduce", "deduction", "determine", "determining", "determination", "hypothesize", "infer", "inference", "logic", "logical", "reason", "reasoning", "speculate", "speculation", "think", "thinking", "thought", "theorize"]

# 计划与策略 (Planning and Strategy)：制定计划、方法、预测或策略。Token数量：26个
planning_and_strategy = ["algorithm", "algorithmic", "approach", "approaches", "deliberate", "deliberation", "forecast", "forecasting", "method", "methods", "plan", "planning", "plans", "predict", "prediction", "process", "processing", "scheme", "scheming", "solution", "solve", "solving", "strategy", "strategize", "tactic", "tactics"]

# 评估与验证 (Evaluation and Verification)：判断、评估或验证信息。Token数量：12个
evaluation_and_verification = ["assess", "assessment", "evaluate", "evaluation", "judge", "judgment", "rationalize", "rationalization", "validate", "validation", "verify", "verification"]

# 决策与问题解决 (Decision Making and Problem Solving)：做选择、解决难题、处理疑问。Token数量：18个
decision_making = ["choose", "choosing", "decide", "deciding", "decision", "dilemma", "doubt", "issue", "option", "options", "problem", "query", "queries", "question", "resolve", "resolution", "select", "selecting"]

# 反思与回顾 (Reflection and Contemplation)：回顾、深思或权衡经验。Token数量：14个
reflection_and_contemplation = ["contemplate", "contemplation", "muse", "musing", "ponder", "pondering", "reflect", "reflection", "retrospect", "retrospection", "review", "reviewing", "weigh", "weighing"]

# 概念与理论 (Concepts and Theories)：抽象概念、想法、模型或原则。Token数量：16个
concepts_and_theories = ["concept", "concepts", "hypothesis", "hypotheses", "idea", "ideas", "model", "models", "notion", "notions", "paradox", "paradoxes", "principle", "principles", "theory", "theories"]

# 逻辑连接与可能性 (Logical Connectives and Possibility)：逻辑连接词、推理过渡词，或表示可能性的副词。Token数量：29个
logical_connectives = ["alternatively", "although", "and", "because", "but", "either", "even", "hence", "however", "if", "just", "maybe", "nevertheless", "neither", "nor", "only", "or", "perhaps", "possibly", "probably", "since", "so", "still", "then", "therefore", "though", "thus", "while", "yet"]

# Dictionary mapping category names to their word lists
reasoning_categories = {
    "Thinking & Reasoning": thinking_and_reasoning,
    "Planning & Strategy": planning_and_strategy,
    "Evaluation & Verification": evaluation_and_verification,
    "Decision Making": decision_making,
    "Reflection & Contemplation": reflection_and_contemplation,
    "Concepts & Theories": concepts_and_theories,
    "Logical Connectives": logical_connectives
}

def normalize_and_tokenize(text: str) -> list:
    """
    Convert text to lowercase, remove punctuation, and tokenize.
    Returns a list of words.
    """
    # Convert to lowercase
    text = text.lower()
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    # Tokenize by splitting on whitespace
    return text.split()

def analyze_reasoning_vocabulary(text: str) -> dict:
    """
    Analyze reasoning vocabulary in the given text.
    Returns counts for each category and total reasoning words.
    """
    tokens = normalize_and_tokenize(text)
    token_counts = Counter(tokens)
    
    results = {}
    total_reasoning_words = 0
    
    for category, word_list in reasoning_categories.items():
        count = sum(token_counts[word] for word in word_list)
        results[category] = count
        total_reasoning_words += count
    
    results["Total Reasoning Words"] = total_reasoning_words
    results["Total Words"] = len(tokens)
    results["Reasoning Ratio"] = total_reasoning_words / len(tokens) if len(tokens) > 0 else 0
    
    return results

def has_reasoning_vocabulary(text: str, category: str = None, min_count: int = 1) -> bool:
    """
    Check if text contains reasoning vocabulary.
    If category is specified, check only that category.
    """
    analysis = analyze_reasoning_vocabulary(text)
    
    if category and category in reasoning_categories:
        return analysis[category] >= min_count
    else:
        return analysis["Total Reasoning Words"] >= min_count

def load_jsonl_data(file_path):
    """Load data from JSONL file"""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []

def extract_question_from_input(input_text):
    """Extract the main question/problem from the input text"""
    # Look for the problem specification after "user\n"
    if "user\n" in input_text:
        user_part = input_text.split("user\n", 1)[1]
        # Split by "assistant" to get only the user part
        if "assistant" in user_part:
            user_part = user_part.split("assistant")[0]
        
        # Extract the main problem description (usually the first substantial paragraph)
        lines = user_part.strip().split('\n')
        question_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('###') and not line.startswith('```'):
                question_lines.append(line)
                if len(question_lines) >= 3:  # Get first few lines of the problem
                    break
        
        return '\n'.join(question_lines)
    return input_text[:200] + "..." if len(input_text) > 200 else input_text



def main():
    st.set_page_config(
        page_title="JSONL Data Viewer", 
        page_icon="📊", 
        layout="wide"
    )
    
    st.title("📊 JSONL Data Viewer")
    st.markdown("View and analyze evaluation results from JSONL files")
    
    # File path input
    default_path = "/data/xuandong_zhao/mnt/xiaolong/ArcherCodeR/output/ArcherCodeR/Archer-Intuitor-Qwen2.5-3B-2k-8k-batch64/eval/60.jsonl"
    file_path = st.text_input("File Path:", value=default_path, help="Enter the path to your JSONL file")
    
    if st.button("Load Data") or file_path:
        data = load_jsonl_data(file_path)
        
        if data:
            st.success(f"Loaded {len(data)} records")
            
            # Create DataFrame for easier manipulation
            df_data = []
            for i, item in enumerate(data):
                question = extract_question_from_input(item.get('input', ''))
                output_text = item.get('output', '')
                
                # Analyze reasoning vocabulary in the output
                reasoning_analysis = analyze_reasoning_vocabulary(output_text)
                
                df_data.append({
                    'Index': i + 1,
                    'Question': question,
                    'Score': item.get('score', 0),
                    'Step': item.get('step', 0),
                    'Reward': item.get('reward', 0),
                    'Accuracy': item.get('acc', 0),
                    'Full Input': item.get('input', ''),
                    'Full Output': output_text,
                    'Total Reasoning Words': reasoning_analysis['Total Reasoning Words'],
                    'Has Reasoning': reasoning_analysis['Total Reasoning Words'] > 0
                })
            
            df = pd.DataFrame(df_data)
            
            # Sidebar filters
            st.sidebar.header("Filters")
            
            # Reasoning vocabulary filter
            st.sidebar.subheader("🧠 Reasoning Vocabulary Filter")
            reasoning_filter = st.sidebar.selectbox(
                "Show records:",
                ["All", "With Reasoning Vocabulary", "Without Reasoning Vocabulary"]
            )
            
            # Search functionality
            search_term = st.sidebar.text_input("Search in questions:", "")
            
            # Apply reasoning vocabulary filter
            if reasoning_filter == "With Reasoning Vocabulary":
                filtered_df = df[df['Has Reasoning'] == True]
            elif reasoning_filter == "Without Reasoning Vocabulary":
                filtered_df = df[df['Has Reasoning'] == False]
            else:
                filtered_df = df.copy()
            
            if search_term:
                filtered_df = filtered_df[
                    filtered_df['Question'].str.contains(search_term, case=False, na=False)
                ]
            
            # Statistics
            total_records = len(df)
            records_with_reasoning = len(df[df['Has Reasoning'] == True])
            records_without_reasoning = total_records - records_with_reasoning
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", total_records)
            with col2:
                st.metric("With Reasoning", records_with_reasoning)
            with col3:
                st.metric("Without Reasoning", records_without_reasoning)
            with col4:
                st.metric("Filtered Records", len(filtered_df))
            
            # Display mode selection
            display_mode = st.radio(
                "Display Mode:", 
                ["Table View", "Detailed View"], 
                horizontal=True
            )
            
            if display_mode == "Table View":
                # Table view
                st.subheader("Data Table")
                display_df = filtered_df[['Index', 'Question', 'Score', 'Accuracy', 'Step']].copy()
                display_df['Question'] = display_df['Question'].str[:100] + "..."
                st.dataframe(display_df, width='stretch')
                
            else:
                # Detailed view
                st.subheader("Detailed View")
                
                # Record selection
                if len(filtered_df) > 0:
                    selected_idx = st.selectbox(
                        "Select Record:", 
                        options=filtered_df.index,
                        format_func=lambda x: f"Record {filtered_df.loc[x, 'Index']} (Score: {filtered_df.loc[x, 'Score']})"
                    )
                    
                    selected_record = filtered_df.loc[selected_idx]
                    
                    # Display selected record details
                    col1, col2 = st.columns([2, 1])
                    
                    with col2:
                        st.subheader("Metrics")
                        st.metric("Score", selected_record['Score'])
                        st.metric("Accuracy", selected_record['Accuracy'])
                        st.metric("Reward", selected_record['Reward'])
                        st.metric("Step", selected_record['Step'])
                    
                    with col1:
                        st.subheader("Question/Problem")
                        st.text_area("Question Content", selected_record['Question'], height=200, disabled=True, label_visibility="collapsed")
                    
                    st.subheader("Full Output")
                    st.text_area("Output Content", selected_record['Full Output'], height=400, disabled=True, label_visibility="collapsed")
                    
                    # Expandable sections for full content
                    with st.expander("Full Input"):
                        st.text(selected_record['Full Input'])
                    
                    with st.expander("Full Output"):
                        st.text(selected_record['Full Output'])
                else:
                    st.info("No records match the current filters.")
            
            # Download filtered data
            if len(filtered_df) > 0:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download filtered data as CSV",
                    data=csv,
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
