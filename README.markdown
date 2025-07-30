# AI Research Paper Companion

Welcome to the **AI Research Paper Companion**, a powerful tool designed to assist researchers, students, and academics in exploring and analyzing research topics using a multi-agent AI framework. Powered by the CrewAI Multi-Agent Framework, this application leverages advanced AI models to explain complex topics, find relevant literature, and identify research gaps.

![Video](https://github.com/sakshamagarwalm2/ReasearchPaperAgent/blob/main/AI%20Research%20Paper%20Companion.mp4)

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Available Agents](#available-agents)
- [Analysis Options](#analysis-options)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
The AI Research Paper Companion is a Streamlit-based web application that integrates with AI models (OpenAI and Groq) to provide a collaborative research experience. It uses multiple AI agents to break down research topics, search for academic papers, and suggest future research directions, making it an invaluable tool for academic and professional research.

## Features
- **Multi-Agent Collaboration**: Utilizes the CrewAI framework to coordinate specialized AI agents for different tasks.
- **Topic Explanation**: Simplifies complex research topics into accessible explanations.
- **Literature Search**: Finds and summarizes relevant papers from arXiv.
- **Research Gap Analysis**: Identifies gaps in current literature and suggests new research questions.
- **Customizable Analysis**: Allows users to select specific analysis options (explanation, literature, gaps).
- **Real-Time Monitoring**: Provides a live activity monitor to track agent progress.
- **Support for Multiple AI Models**: Compatible with OpenAI and Groq models, including fast and capable variants.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/sakshamagarwalm2/ReasearchPaperAgent
   ```

2. **Install Dependencies**
   Create a virtual environment and install the required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set Up Environment Variables**
   Create a `.env` file in the project root and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

   - Obtain an API key from [OpenAI](https://platform.openai.com/) or [Groq](https://console.groq.com/).
   - Install the `python-dotenv` package if not already included (`pip install python-dotenv`).

4. **Run the Application**
   Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   Open your browser and navigate to `http://localhost:8501`.

## Usage
1. **Configure the App**
   - In the sidebar, select an AI model (OpenAI or Groq).
   - Enter the corresponding API key.
   - Choose a Groq model variant if using Groq (e.g., `llama-3.1-8b-instant`).

2. **Enter a Research Topic**
   - Input your research topic in the "Research Query" section (e.g., "Machine Learning in Healthcare").
   - Use the "Quick Start Examples" expander for pre-defined topics.

3. **Select Analysis Options**
   - Check the boxes for "Explain Topic," "Find Papers," and/or "Find Gaps" based on your needs.

4. **Start Analysis**
   - Click "Start Analysis" to initiate the multi-agent process.
   - Monitor agent activity in the "CrewAI Activity Monitor" section.
   - View results in the "Research Findings" section once completed.

## Configuration
- **AI Model**: Switch between OpenAI and Groq models.
- **API Key**: Input your API key for authentication.
- **Groq Model**: Select a model variant for Groq (e.g., `llama-3.1-8b-instant` or `llama-3.3-70b-versatile`).
- Ensure your API key is valid before starting the analysis.

## Available Agents
- **Topic Explainer**: Breaks down complex topics into simple explanations.
- **Literature Finder**: Searches and summarizes relevant papers from arXiv.
- **Gap Analyzer**: Identifies research gaps and suggests future directions.

## Analysis Options
- **Explain Topic**: Generates a clear explanation of the research topic.
- **Find Papers**: Retrieves and summarizes recent, relevant papers.
- **Find Gaps**: Highlights gaps in the literature and proposes research questions.

## Contributing
We welcome contributions to enhance the AI Research Paper Companion! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Submit a pull request.

Please ensure your code follows the project's coding standards and includes appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions, support, or feedback, please open an issue on the [GitHub repository](https://github.com/sakshamagarwalm2/ReasearchPaperAgent) or contact the maintainers at `sakshamagarwalm2@gmail.com`.

---

*Last updated: July 30, 2025*
