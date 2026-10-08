---
title: "Building AI Agents with MCP, ADK, and A2A: A Comprehensive Guide"
date: 2025-11-13
permalink: /posts/2025/11/ai-agents-mcp-adk-a2a/
excerpt: "How the Model Context Protocol, Google's Agent Development Kit and the Agent2Agent protocol fit together, with a currency agent built step by step."
tags:
  - Agents
  - MCP
  - Google ADK
  - A2A
---

![](/images/posts/agents-mcp-adk-a2a/1_hNos347FZrJyOWVVAVD_Hg.png)

The landscape of AI agent development is rapidly evolving, and three powerful technologies are emerging as foundational building blocks: Model Context Protocol (MCP), Agent Development Kit (ADK), and Agent2Agent Protocol (A2A). Together, these tools are transforming how developers build, deploy, and orchestrate intelligent AI systems that can communicate with external data sources, perform complex tasks, and collaborate with other agents.

## Understanding the Foundation: What Are MCP, ADK, and A2A?

### Model Context Protocol (MCP): The Universal Bridge for AI Context

Model Context Protocol is an open standard introduced by Anthropic in November 2024 that revolutionizes how AI systems connect with external resources. Think of MCP as a “universal remote” for AI — it provides a standardized way for language models to access tools, data sources, and services without requiring custom integrations for each combination.

**Key capabilities of MCP include:**

- **Resources**: Providing context and data for AI models to use
- **Tools**: Functions that AI models can execute to perform actions
- **Prompts**: Templated messages and workflows for users
- **Sampling**: Server-initiated agentic behaviors for recursive LLM interactions

The protocol operates on a client-server architecture using JSON-RPC 2.0 messaging, similar to the Language Server Protocol (LSP) that transformed developer tooling. MCP supports both standard input/output (stdio) and HTTP transports with Server-Sent Events (SSE) for streaming capabilities. Major AI providers including OpenAI, Google DeepMind, and Microsoft have adopted MCP, signaling its emergence as an industry standard. The protocol addresses the fundamental challenge of information silos by eliminating the need for N×M custom connectors between AI models and data sources.

### Agent Development Kit (ADK): Google’s Framework for Production-Ready Agents

Agent Development Kit is Google’s open-source framework designed to make agent development feel more like software development. Released with a stable v1.0.0 version, ADK is the same framework powering agents within Google products like Agentspace and the Google Customer Engagement Suite.

**ADK’s core strengths include:**

- **Model-agnostic design**: Works with different LLMs while optimized for Gemini
- **Deployment flexibility**: Run locally, on Vertex AI Agent Engine, or custom infrastructure
- **Multi-agent orchestration**: Compose specialized agents in hierarchical structures
- **Built-in evaluation**: Systematically assess both response quality and execution.

ADK provides three primary agent types for building sophisticated multi-agent systems:

1.  **LLM Agents**: Leverage large language models for natural language understanding and reasoning
2.  **Sequential/Parallel/Loop Agents**: Handle workflow orchestration with predictable patterns
3.  **Hierarchical Agents**: Enable parent-child relationships for complex coordination

The framework seamlessly integrates with over 100 pre-built connectors to enterprise systems, supports workflows through Application Integration, and can access data from AlloyDB, BigQuery, and other Google Cloud services.

### Agent2Agent (A2A) Protocol: Enabling Multi-Agent Collaboration

The Agent2Agent Protocol, introduced by Google in April 2025, is an open communication standard that enables AI agents to discover, communicate, and collaborate with each other regardless of their underlying frameworks or vendors. With support from over 150 organizations including major hyperscalers, technology providers, and enterprise customers, A2A is rapidly gaining momentum.

**A2A’s defining features include:**

- **Universal interoperability**: Agents work together seamlessly across platforms
- **Enterprise-grade security**: Supports OAuth 2.0, OpenID Connect, and API keys aligned with OpenAPI
- **Multi-modal support**: Handles text, audio, and video streaming
- **Long-running tasks**: Designed for both quick responses and deep research that may take hours or days
- **Real-time updates**: Provides continuous feedback throughout task lifecycle

The protocol follows a three-step workflow:

1.  **Discovery**: Client agents fetch Agent Cards from remote agents to identify capabilities
2.  **Authentication**: Secure connection establishment using enterprise-grade schemes
3.  **Communication**: Task delegation and information exchange via JSON-RPC 2.0 over HTTPS

## Building a Currency Agent: A Practical Example

The Google Codelabs tutorial demonstrates how these three technologies work together by building a currency conversion agent. This hands-on example walks through the complete development lifecycle from creating an MCP server to exposing an agent via A2A.

### Step 1: Creating a Local MCP Server with FastMCP

The first step involves building an MCP server using FastMCP, a Python package that simplifies MCP server creation. The server exposes a single tool called get_exchange_rate that fetches real-time currency data from the Frankfurter API:

```python
import httpx
from fastmcp import FastMCP
mcp = FastMCP("Currency MCP Server 💵")
@mcp.tool()
def get_exchange_rate(
    currency_from: str = 'USD',
    currency_to: str = 'EUR',
    currency_date: str = 'latest',
):
    """Use this to get current exchange rate."""
    response = httpx.get(
        f'https://api.frankfurter.app/{currency_date}',
        params={'from': currency_from, 'to': currency_to},
    )
    return response.json()
```

FastMCP automatically handles protocol details through decorators, making it the fastest path from idea to production. The framework supports both synchronous and asynchronous functions, with automatic type hint conversion to tool definitions.

### Step 2: Deploying the MCP Server to Cloud Run

Running MCP servers remotely on Cloud Run provides significant benefits:

- **Scalability**: Automatic scaling based on demand
- **Centralized access**: Team members can share a single server via IAM privileges
- **Security**: Built-in authentication through Cloud Run Invoker role

The deployment process is straightforward using the gcloud CLI:

```bash
gcloud run deploy mcp-server --no-allow-unauthenticated --region=us-central1 --source .
```

For local development, the Cloud Run proxy creates an authenticated tunnel to the remote server, ensuring all traffic is properly authorized.

### Step 3: Building the Agent with ADK

With the MCP server deployed, the next step creates a currency agent using ADK that connects to the MCP tools. The agent leverages ADK’s MCPToolset class for seamless integration:

```python
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool import MCPToolset, StreamableHTTPConnectionParams
def create_agent() -> LlmAgent:
    return LlmAgent(
        model="gemini-2.5-flash",
        name="currency_agent",
        description="An agent that can help with currency conversions",
        instruction=SYSTEM_INSTRUCTION,
        tools=[
            MCPToolset(
                connection_params=StreamableHTTPConnectionParams(
                    url=os.getenv("MCP_SERVER_URL", "http://localhost:8080/mcp")
                )
            )
        ],
    )
```

ADK makes agent creation extremely lightweight while providing sophisticated orchestration capabilities. The framework handles agent lifecycle management, memory, tool integration, and supports sequential, parallel, or loop-based multi-agent workflows.

### Step 4: Exposing the Agent as an A2A Server

The final step exposes the currency agent using the A2A protocol, enabling it to communicate with other agents. This involves creating an Agent Card that advertises capabilities:

```python
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
skill = AgentSkill(
    id='get_exchange_rate',
    name='Currency Exchange Rates Tool',
    description='Helps with exchange values between various currencies',
    tags=['currency conversion', 'currency exchange'],
    examples=['What is exchange rate between USD and GBP?'],
)
agent_card = AgentCard(
    name='Currency Agent',
    description='Helps with exchange rates for currencies',
    url=f'http://{host}:{port}/',
    version='1.0.0',
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[skill],
)
```

The A2A Python SDK provides an A2AFastAPIApplication class that simplifies running A2A-compliant HTTP servers using FastAPI and Uvicorn. The AgentExecutor interface handles core request processing logic, converting between ADK's google.genai.types and A2A's a2a.types.

## Enterprise Benefits and Strategic Implications

### Breaking Down Silos and Avoiding Vendor Lock-In

The combination of MCP, ADK, and A2A addresses a critical enterprise challenge: interoperability. Organizations can now build specialized agents using different frameworks and vendors while ensuring they can communicate through standardized protocols. This prevents vendor lock-in and provides the flexibility to choose best-in-class solutions for specific use cases.

### Accelerating Time to Production

By standardizing connectivity, these protocols dramatically reduce integration complexity. What previously required weeks of custom development can now be accomplished in days. ADK’s integration with Vertex AI Agent Engine provides a direct path from prototype to production-ready deployment.

### Enabling Complex Multi-Agent Workflows

The true power emerges when multiple specialized agents collaborate to solve complex problems. ADK’s hierarchical agent structures support sophisticated patterns like:

- **Coordinator/Dispatcher Pattern**: A central LLM agent routes requests to specialized sub-agents
- **Sequential Workflows**: Step-by-step processing pipelines
- **Parallel Execution**: Concurrent task processing for independent operations
- **Long-Running Research**: Tasks spanning hours or days with context preservation

For example, Tyson Foods and Gordon Food Service are pioneering collaborative A2A systems to drive sales and reduce supply chain friction, creating real-time channels for agents to share product data and leads.

### Enterprise-Grade Security and Compliance

A2A integrates seamlessly with existing enterprise infrastructure, supporting OAuth 2.0, OpenID Connect, and API keys as authentication mechanisms. The protocol treats agents as opaque, allowing collaboration without revealing proprietary logic or internal implementations — essential for preserving intellectual property.

### Advanced Capabilities with Gemini 2.5 Flash

The currency agent example leverages Gemini 2.5 Flash, Google’s best model in terms of price-performance. Released with enhanced capabilities, Gemini 2.5 Flash excels at:

- **Large-scale processing**: Handles high-volume, low-latency tasks efficiently
- **Thinking capabilities**: First Flash model with visible reasoning processes
- **Advanced tool use**: 25% faster response times with potentially 85% lower cost per query
- **Enhanced security**: Significantly increased protection against indirect prompt injection attacks

Gemini 2.5 Flash also features “thought summaries” that organize the model’s reasoning into clear formats, enabling customers to validate complex AI tasks and ensure alignment with business logic.

## Best Practices and Considerations

### Designing Effective MCP Servers

When building MCP servers, focus on modularity and reusability:

- **Expose focused capabilities**: Each server should handle specific domains
- **Implement proper validation**: Check file sizes, types, and permissions before processing
- **Add comprehensive logging**: Enable production monitoring and debugging
- **Use type hints**: FastMCP automatically converts them to tool definitions

### Architecting Multi-Agent Systems

Effective multi-agent design requires careful orchestration:

- **Define clear roles**: Each agent should have specialized expertise
- **Establish communication patterns**: Use shared state, delegation, or explicit invocation
- **Balance LLM-driven vs. deterministic flows**: LLM orchestration provides flexibility; deterministic workflows offer precision
- **Plan for long-running operations**: Design for tasks that may require hours or days

### Security and Access Control

Enterprise deployments demand robust security:

- **Require authentication**: Use --no-allow-unauthenticated for Cloud Run deployments
- **Implement IAM-based access**: Leverage Cloud Run Invoker role for team access
- **Validate inputs thoroughly**: Prevent path traversal and injection attacks
- **Use service accounts**: Assign minimum necessary permissions

## The Future of Agent Ecosystems

The convergence of MCP, ADK, and A2A represents a significant shift toward commoditized multi-agent connectivity. Much like TCP/IP and HTTP became foundational internet protocols, these standards are positioning themselves as the backbone of collaborative AI systems.

Early adopters report impressive results: 66% productivity gains, 57% cost savings, and 54% improvement in customer experience. As the ecosystem matures, we can expect:

- **Expanded framework support**: Integration with LangChain, CrewAI, and other orchestration tools
- **Richer tool ecosystems**: Growing libraries of pre-built MCP servers and ADK agents
- **Industry-specific solutions**: Vertical-focused agents leveraging standard protocols
- **Cross-organizational collaboration**: Secure agent-to-agent communication between companies

## Conclusion

MCP, ADK, and A2A together provide a comprehensive stack for building modern AI agent systems. MCP solves the data connectivity problem, ADK provides production-ready orchestration, and A2A enables multi-agent collaboration at scale. This trinity of technologies is transforming AI development from isolated experiments to enterprise-grade, interoperable systems.

The currency agent tutorial demonstrates that getting started is remarkably straightforward — developers can go from zero to a deployed, collaborative agent system in just a few hours. As these standards gain adoption and the ecosystem matures, we’re entering an era where AI agents can truly work together across boundaries, unlocking unprecedented automation capabilities and business value.

For developers looking to stay ahead in the rapidly evolving AI landscape, investing time in understanding and implementing these protocols is no longer optional — it’s essential for building the next generation of intelligent, collaborative applications.

---

*Also published on [Medium](https://medium.com/@rbinsafi/building-ai-agents-with-mcp-adk-and-a2a-a-comprehensive-guide-29cbc641eb50).*
