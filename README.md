# REMAS - Real Estate Multi-Agent System

An autonomous multi-agent system for intelligent real estate matching and scheduling, demonstrating collaborative AI agents working together to solve complex real-world problems.

## 🎬 Quick Demo

![REMAS Demo](docs/demo.gif)
_Watch REMAS in action: Natural language input → Multi-agent reasoning → Autonomous scheduling_

1. **As a Renter**: Describe your ideal apartment in natural language
2. **As an Owner**: Get AI-powered candidate recommendations
3. **As a Realtor**: Access automated scheduling and analytics tools

_Demo shows actual system responses with live data_

## 🌟 Why REMAS?

### Unique Advantages

- **🤖 Fully Autonomous**: Agents make intelligent decisions without human intervention
- **🧠 Continuous Learning**: System improves from every interaction and feedback
- **⚖️ Built-in Fairness**: Prevents discrimination and ensures equal opportunities
- **💬 Natural Language**: No forms - just describe what you want conversationally

## 💡 Real-World Use Cases

### Scenario 1: The Busy Professional

**Challenge**: Sarah works 60-hour weeks and can't spend time browsing listings  
**REMAS Solution**: "I need a quiet 1-bedroom near downtown with gym access under $2000"  
**Result**: System autonomously finds 5 perfect matches and schedules 3 showings for her weekend

### Scenario 2: The Frustrated Owner

**Challenge**: Mark's property has been listed for 3 months with no qualified interest  
**REMAS Solution**: Audit agent identifies pricing issue, learning agent suggests improvements  
**Result**: Adjusted recommendations lead to 4 showings and a lease within 2 weeks

### Scenario 3: The Overwhelmed Realtor

**Challenge**: Managing 50+ properties and 200+ potential renters manually  
**REMAS Solution**: Automated scheduling agent makes intelligent decisions, handles communications  
**Result**: time savings, improved client satisfaction, data-driven insights

## 🎯 System Features

### For Renters

- **Natural Language Input**: Describe preferences in conversational language
- **Intelligent Matching**: Receive personalized property recommendations
- **Smart Filtering**: System understands context and priorities

### For Property Owners

- **Property Description Processing**: Convert property details into searchable formats
- **Candidate Matching**: Find qualified renters based on compatibility
- **Market Insights**: Understand property positioning and appeal

### For Realtors

- **Automated Scheduling**: AI-driven showing decision management
- **Performance Analytics**: Comprehensive system audits and insights
- **Feedback Integration**: Continuous system improvement through feedback
- **Fairness Monitoring**: Ensure equitable treatment for all users

## 🤖 Multi-Agent Architecture

REMAS showcases autonomous agents that collaborate to provide comprehensive real estate services:

### Core Agents

**🏠 User Parser Agent** (`agents/user_parser_agent.py`)

- Processes natural language renter preferences
- Extracts structured data from user descriptions
- Handles preference disambiguation and validation

**🏢 Owner Parser Agent** (`agents/owner_parser_agent.py`)

- Interprets property descriptions and listing details
- Converts unstructured property information into searchable formats
- Standardizes property attributes and features

**🔍 Matching Agent** (`agents/matching_agent.py`)

- Performs intelligent property-renter matching using vector similarity
- Generates compatibility scores and recommendations
- Provides summarized matching insights for both parties

**📅 Manage Showings Agent** (`agents/manage_showings_agent.py`)

- Makes autonomous decisions about property showing scheduling
- Balances multiple factors: match quality, candidate availability, fairness
- Generates automated email notifications for all parties

**🔍 Audit Starved Agent** (`agents/audit_starved_agent.py`)

- Monitors system fairness and identifies underserved users/owners
- Provides intelligent recommendations for improving match opportunities
- Generates actionable insights for system optimization

**🎯 Apply Learning Agent** (`agents/apply_learning_agent.py`)

- Processes feedback to extract property-specific insights
- Continuously learns from realtor feedback and showing outcomes
- Enhances future decision-making with accumulated knowledge

## 🔧 Technical Architecture

### Vector Similarity Matching

- Uses cosine similarity for semantic property-renter matching
- Embedding-based approach captures nuanced preferences
- Scalable vector database for real-time searches

### Agent Communication

- Agents share data through structured interfaces
- Asynchronous processing for optimal performance
- Event-driven architecture for responsive system behavior

### Data Management

- Qdrant vector database for similarity searches
- Structured data storage for profiles and preferences
- JSON-based configuration and insights storage

## 📊 System Outputs

### Automated Reports

- Daily showing decisions with detailed reasoning
- Starvation audits identifying improvement opportunities
- Performance metrics and system health indicators

### Email Automation

- Personalized notifications for owners and renters
- Context-aware messaging based on match quality
- Professional communication templates

### Learning Insights

- Property-specific improvement recommendations
- System-wide optimization suggestions
- Continuous feedback loop for agent enhancement

## 📁 Project Structure

```
REMAS/
├── agents/                 # Multi-agent system components
│   ├── user_parser_agent.py
│   ├── owner_parser_agent.py
│   ├── matching_agent.py
│   ├── manage_showings_agent.py
│   ├── audit_starved_agent.py
│   └── apply_learning_agent.py
├── feedback/               # Learning and feedback system
├── utils/                  # Shared utilities and database connections
├── config/                 # System configuration
├── data/                   # Data storage and insights
├── logs/                   # System logs and outputs
└── app.py                  # Streamlit web interface
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Required dependencies (install via pip)
- Vector database (Qdrant) access
- OpenAI API key for LLM agents

### Installation

1. Clone the repository:

```bash
git clone [repository-url]
cd REMAS
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
# Add your API keys and database configurations
cp .env.example .env
# Edit .env with your credentials
```

### Running the System

Start the Streamlit web interface:

```bash
python -m streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 📄 License
