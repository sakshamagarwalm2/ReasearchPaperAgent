import streamlit as st
import os
from dotenv import load_dotenv
import openai
from groq import Groq
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
import arxiv
import requests
from bs4 import BeautifulSoup
import re
from typing import List, Dict
import json
import time
from datetime import datetime

# Load environment variables
load_dotenv()

# Custom CSS for cleaner UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 10px 10px;
    }
    .agent-status {
        background: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    .agent-working {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        animation: pulse 2s infinite;
    }
    .agent-complete {
        background: #d4edda;
        border-left: 4px solid #28a745;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .config-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .analysis-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Custom tool for paper search
class ArxivSearchTool(BaseTool):
    name: str = "Arxiv Paper Search"
    description: str = "Search for academic papers on ArXiv"
    
    def _run(self, query: str) -> str:
        try:
            search = arxiv.Search(
                query=query,
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for paper in search.results():
                results.append({
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "summary": paper.summary[:500] + "..." if len(paper.summary) > 500 else paper.summary,
                    "url": paper.entry_id,
                    "published": paper.published.strftime("%Y-%m-%d")
                })
            
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Error searching papers: {str(e)}"

# Custom tool for paper analysis
class PaperAnalysisTool(BaseTool):
    name: str = "Paper Analysis"
    description: str = "Analyze research paper content and extract key information"
    
    def _run(self, paper_text: str) -> str:
        try:
            words = paper_text.split()
            word_count = len(words)
            key_terms = [word for word in words if len(word) > 8 and word.isalpha()]
            key_terms = list(set(key_terms[:10]))
            
            analysis = {
                "word_count": word_count,
                "key_terms": key_terms,
                "summary_length": len(paper_text),
                "complexity_score": min(10, word_count / 1000)
            }
            
            return json.dumps(analysis, indent=2)
        except Exception as e:
            return f"Error analyzing paper: {str(e)}"

# Initialize tools
arxiv_tool = ArxivSearchTool()
analysis_tool = PaperAnalysisTool()

# Function to verify API keys
def verify_openai_api_key(api_key: str) -> bool:
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception:
        return False

def verify_groq_api_key(api_key: str) -> bool:
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Test"}],
            model="llama-3.1-8b-instant",
            max_tokens=5
        )
        return True
    except Exception:
        return False

# Define Agents
def create_agents(model: str, api_key: str, groq_model: str = None):
    os.environ["OPENAI_API_KEY"] = api_key if model == "OpenAI" else os.getenv("OPENAI_API_KEY", "")
    os.environ["GROQ_API_KEY"] = api_key if model == "Groq" else os.getenv("GROQ_API_KEY", "")
    
    if model == "OpenAI":
        model_name = "gpt-3.5-turbo"
    elif model == "Groq":
        model_name = f"groq/{groq_model or 'llama-3.1-8b-instant'}"
    
    topic_explainer = Agent(
        role='Topic Explainer',
        goal='Explain complex research topics in simple, understandable terms',
        backstory="""You are an expert academic researcher with a gift for making 
        complex topics accessible. You excel at breaking down difficult concepts 
        into digestible explanations while maintaining accuracy.""",
        verbose=True,
        allow_delegation=False,
        tools=[analysis_tool],
        llm=model_name
    )
    
    literature_finder = Agent(
        role='Literature Finder',
        goal='Find and recommend relevant research papers based on given topics',
        backstory="""You are a skilled research librarian with deep knowledge of 
        academic databases and search strategies. You excel at finding the most 
        relevant and high-quality papers for any research topic.""",
        verbose=True,
        allow_delegation=False,
        tools=[arxiv_tool],
        llm=model_name
    )
    
    gap_analyzer = Agent(
        role='Gap Analyzer',
        goal='Identify research gaps and suggest future research directions',
        backstory="""You are a strategic research analyst who specializes in 
        identifying gaps in current literature and suggesting innovative research 
        directions. You have a keen eye for spotting opportunities for new research.""",
        verbose=True,
        allow_delegation=False,
        tools=[analysis_tool],
        llm=model_name
    )
    
    return topic_explainer, literature_finder, gap_analyzer

# Define Tasks
def create_tasks(topic, agents, include_explanation, include_literature, include_gaps):
    topic_explainer, literature_finder, gap_analyzer = agents
    
    tasks = []
    
    if include_explanation:
        explain_task = Task(
            description=f"""Explain the research topic '{topic}' in simple terms. 
            Break down key concepts, methodologies, and current understanding. 
            Make it accessible to someone new to the field while maintaining accuracy.""",
            agent=topic_explainer,
            expected_output="A clear, comprehensive explanation of the topic with key concepts defined"
        )
        tasks.append(explain_task)
    
    if include_literature:
        literature_task = Task(
            description=f"""Find and summarize the most relevant recent papers about '{topic}'. 
            Focus on high-impact papers from the last 3-5 years. Provide summaries of 
            key findings and methodologies.""",
            agent=literature_finder,
            expected_output="A list of relevant papers with summaries and key findings"
        )
        tasks.append(literature_task)
    
    if include_gaps:
        gap_task = Task(
            description=f"""Based on the topic explanation and literature review, 
            identify key research gaps in '{topic}'. Suggest specific research questions 
            and methodologies that could address these gaps.""",
            agent=gap_analyzer,
            expected_output="Identified research gaps with specific research questions and suggested methodologies"
        )
        tasks.append(gap_task)
    
    return tasks

def update_agent_status(container, agent_name: str, status: str, details: str = ""):
    """Update agent status in the CrewAI monitoring section"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if status == "working":
        icon = "🔄"
    elif status == "complete":
        icon = "✅"
    else:
        icon = "⏳"
    
    container.markdown(f"""
    <div>
        <strong>{icon} {agent_name}</strong> - {status.title()} <small>({timestamp})</small><br>
        <small>{details}</small>
    </div>
    """, unsafe_allow_html=True)

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Research Paper Companion",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AI Research Paper Companion</h1>
        <p>Powered by CrewAI Multi-Agent Framework by Saksham HKRM</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown('<div>', unsafe_allow_html=True)
        st.header("⚙️ Configuration")
        
        # Model selection
        model = st.selectbox(
            "🧠 AI Model",
            options=["OpenAI", "Groq"],
            help="Choose your AI provider"
        )
        
        # Groq model selection
        if model == "Groq":
            groq_model = st.selectbox(
                "🚀 Groq Model",
                options=[
                    "llama-3.1-8b-instant",
                    "llama-3.3-70b-versatile"
                ],
                help="Fast vs Capable trade-off"
            )
        else:
            groq_model = None
        
        # API key input
        api_key = st.text_input(
            f"🔑 {model} API Key",
            type="password",
            value=os.getenv(f"{model.upper()}_API_KEY", ""),
            help=f"Enter your {model} API key"
        )
        
        # Verify API key
        api_key_valid = False
        if api_key:
            with st.spinner("Verifying API key..."):
                if model == "OpenAI" and verify_openai_api_key(api_key):
                    st.success("✅ API Key Valid")
                    api_key_valid = True
                elif model == "Groq" and verify_groq_api_key(api_key):
                    st.success("✅ API Key Valid")
                    api_key_valid = True
                else:
                    st.error("❌ Invalid API Key")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Agent Info
        st.markdown("### 👥 Available Agents")
        agents_info = {
            "📖 Topic Explainer": "Simplifies complex research topics",
            "📚 Literature Finder": "Discovers relevant papers",
            "🔍 Gap Analyzer": "Identifies research opportunities"
        }
        
        for agent, desc in agents_info.items():
            st.markdown(f"**{agent}**")
            st.caption(desc)
    
    # Main Content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div>', unsafe_allow_html=True)
        st.subheader("📝 Research Query")
        
        research_topic = st.text_area(
            "What would you like to research?",
            placeholder="Enter your research topic (e.g., 'Machine Learning in Healthcare', 'Quantum Computing Applications')",
            height=120,
            help="Be specific for better results"
        )
        
        st.subheader("🎯 Analysis Options")
        # col1_1, col1_2, col1_3 = st.columns(3)
        
        # with col1_1:
        include_explanation = st.checkbox("📖 Explain Topic", value=True)
        # with col1_2:
        include_literature = st.checkbox("📚 Find Papers", value=True)
        # with col1_3:
        include_gaps = st.checkbox("🔍 Find Gaps", value=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        with st.expander("💡 Quick Start Examples", expanded=False):
            example_topics = [
                "Natural Language Processing in Legal Documents",
                "Blockchain Applications in Supply Chain",
                "Deep Learning for Medical Image Analysis",
                "Sustainable Energy Storage Solutions",
                "Quantum Machine Learning Algorithms"
            ]
            
            cols = st.columns(2)
            for i, topic in enumerate(example_topics):
                with cols[i % 2]:
                    if st.button(f"📌 {topic}", key=f"example_{i}", help="Click to use this topic"):
                        st.session_state.research_topic = topic
                        st.rerun()
        
        # Start Analysis Button
        st.markdown("---")
        start_analysis = st.button(
            "🚀 Start Analysis",
            disabled=not api_key_valid or not research_topic.strip(),
            help="Begin the multi-agent research analysis"
        )
        
        if start_analysis:
            if not (include_explanation or include_literature or include_gaps):
                st.error("Please select at least one analysis option!")
            else:
                st.session_state.groq_model = groq_model
                st.session_state.analysis_requested = True
                st.session_state.research_topic = research_topic
                st.session_state.include_explanation = include_explanation
                st.session_state.include_literature = include_literature
                st.session_state.include_gaps = include_gaps
                st.session_state.model = model
                st.session_state.api_key = api_key
                st.rerun()
    
    with col2:
        st.subheader("📊 Analysis Results")
        
        # CrewAI Activity Monitor
        st.markdown("### 🤖 CrewAI Activity Monitor")
        activity_container = st.container()
        
        if hasattr(st.session_state, 'analysis_requested') and st.session_state.analysis_requested:
            
            # Initialize activity monitor
            with activity_container:
                st.markdown('<div class="agent-status" >', unsafe_allow_html=True)
                st.markdown("**🚀 Initializing CrewAI Framework...**")
                st.markdown("</div>", unsafe_allow_html=True)
            
            try:
                # Update activity monitor
                activity_container.empty()
                with activity_container:
                    update_agent_status(st, "System", "working", "Creating AI agents...")
                
                agents = create_agents(st.session_state.model, st.session_state.api_key, st.session_state.get('groq_model'))
                
                activity_container.empty()
                with activity_container:
                    update_agent_status(st, "System", "working", "Preparing analysis tasks...")
                
                tasks = create_tasks(
                    st.session_state.research_topic, 
                    agents,
                    st.session_state.include_explanation,
                    st.session_state.include_literature,
                    st.session_state.include_gaps
                )
                
                if not tasks:
                    st.error("No tasks created! Please select at least one analysis option.")
                    st.session_state.analysis_requested = False
                    return
                
                activity_container.empty()
                with activity_container:
                    update_agent_status(st, "CrewAI Team", "working", "Agents are collaborating on your research...")
                
                crew = Crew(
                    agents=list(agents),
                    tasks=tasks,
                    verbose=True,
                    process=Process.sequential
                )
                
                if st.session_state.model == "Groq":
                    time.sleep(2)  # Rate limiting for Groq
                
                # Show active agents
                agent_names = ["Topic Explainer", "Literature Finder", "Gap Analyzer"]
                selected_agents = []
                if st.session_state.include_explanation:
                    selected_agents.append(agent_names[0])
                if st.session_state.include_literature:
                    selected_agents.append(agent_names[1])
                if st.session_state.include_gaps:
                    selected_agents.append(agent_names[2])
                
                activity_container.empty()
                with activity_container:
                    for agent in selected_agents:
                        update_agent_status(st, agent, "working", f"Processing {st.session_state.research_topic}...")
                
                result = crew.kickoff()
                
                # Show completion
                activity_container.empty()
                with activity_container:
                    for agent in selected_agents:
                        update_agent_status(st, agent, "complete", "Task completed successfully!")
                    
                    st.markdown("""
                    <div>
                        <strong>🎉 Analysis Complete!</strong><br>
                        <small>All agents have finished their tasks successfully.</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.success("✅ Research analysis completed!")
                
                # Display Results
                st.markdown("---")
                st.subheader("📋 Research Findings")
                
                # Results tabs for cleaner organization
                if st.session_state.include_explanation:
                    with st.expander("📖 Topic Explanation", expanded=True):
                        if hasattr(result, 'tasks_output') and len(result.tasks_output) >= 1:
                            st.markdown(result.tasks_output[0].raw)
                        else:
                            st.write(str(result))
                
                if st.session_state.include_literature:
                    with st.expander("📚 Literature Review", expanded=True):
                        task_index = 1 if st.session_state.include_explanation else 0
                        if hasattr(result, 'tasks_output') and len(result.tasks_output) > task_index:
                            st.markdown(result.tasks_output[task_index].raw)
                        else:
                            st.write("Literature review completed.")
                
                if st.session_state.include_gaps:
                    with st.expander("🔍 Research Gaps & Opportunities", expanded=True):
                        task_index = len(tasks) - 1
                        if hasattr(result, 'tasks_output') and len(result.tasks_output) > task_index:
                            st.markdown(result.tasks_output[task_index].raw)
                        else:
                            st.write("Gap analysis completed.")
                
                # Debug section (collapsed by default)
                with st.expander("🔧 Technical Details", expanded=False):
                    st.code(str(result))
                
                st.session_state.analysis_requested = False
                
            except Exception as e:
                error_msg = str(e)
                activity_container.empty()
                with activity_container:
                    st.markdown("""
                    <div class="agent-status" style="border-left-color: #dc3545; background: #f8d7da;">
                        <strong>❌ Analysis Failed</strong><br>
                        <small>An error occurred during processing</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                if "rate_limit" in error_msg.lower():
                    st.error("🚫 Rate Limit Exceeded")
                    st.info("Wait a moment and try again, or use fewer analysis options.")
                else:
                    st.error(f"Error: {error_msg}")
                
                st.session_state.analysis_requested = False
        
        else:
            with activity_container:
                st.info("👈 Configure your settings and enter a research topic to begin!")
                st.markdown("""
                <div>
                    <strong>🤖 CrewAI Agents Ready</strong><br>
                    <small>Waiting for your research query...</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>🤖 AI Research Paper Companion</strong> | Powered by CrewAI Multi-Agent Framework</p>
        <p><small>This tool uses collaborative AI agents to analyze research topics and identify opportunities.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()