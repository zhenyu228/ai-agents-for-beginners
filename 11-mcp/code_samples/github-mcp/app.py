import os
import json
from dotenv import load_dotenv


import chainlit as cl
from mcp import ClientSession

from semantic_kernel.kernel import Kernel
from azure.core.credentials import AzureKeyCredential


from semantic_kernel.functions import KernelFunction, kernel_function
from semantic_kernel.contents import ChatHistory, AuthorRole, ChatMessageContent
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents.function_call_content import FunctionCallContent
from semantic_kernel.contents.function_result_content import FunctionResultContent
from semantic_kernel.connectors.mcp import MCPStdioPlugin
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread, AgentGroupChat
from semantic_kernel.agents.strategies import (
    SequentialSelectionStrategy,
    DefaultTerminationStrategy
)

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchFieldDataType, SearchableField


# Load environment variables
load_dotenv()


# Example Weather Plugin (Tool)

        

class RAGPlugin:
    def __init__(self, search_client):
        self.search_client = search_client

    @kernel_function(name="search_events", description="Searches for relevant events based on a query")
    def search_events(self, query: str) -> str:
        """Retrieves relevant events from Azure Search based on the query."""
        try:
            results = self.search_client.search(query, top=5)
            context_strings = []
            for result in results:
                if 'content' in result:
                    context_strings.append(f"Event: {result['content']}")

            if context_strings:
                return "\n\n".join(context_strings)
            else:
                return "No relevant events found."
        except Exception as e:
            return f"Error searching for events: {str(e)}"


# Initialize Azure AI Search with persistent storage
search_service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
index_name = "event-descriptions"

search_client = SearchClient(
    endpoint=search_service_endpoint,
    index_name=index_name,
    credential=AzureKeyCredential(search_api_key)
)

index_client = SearchIndexClient(
    endpoint=search_service_endpoint,
    credential=AzureKeyCredential(search_api_key)
)

# Define the index schema
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="content", type=SearchFieldDataType.String)
]

index = SearchIndex(name=index_name, fields=fields)

# Check if index already exists if not, create it
try:
    existing_index = index_client.get_index(index_name)
    print(f"Index '{index_name}' already exists, using the existing index.")
except Exception as e:
    # Create the index if it doesn't exist
    print(f"Creating new index '{index_name}'...")
    index_client.create_index(index)

# Always read event descriptions from markdown file
with open("event-descriptions.md", "r") as f:
    markdown_content = f.read()

# Split the markdown content into individual event descriptions
event_descriptions = markdown_content.split("---")  # You can change the delimiter

# Create documents for Azure Search
documents = []
for i, description in enumerate(event_descriptions):
    description = description.strip()  # Remove leading/trailing whitespace
    if description:  # Avoid empty descriptions
        documents.append({"id": str(i + 1), "content": description})

# Add documents to the index (only if we have documents)
if documents:
    # Delete existing documents first to avoid duplicates
    try:
        search_client.delete_documents(documents=[{"id": doc["id"]} for doc in documents])
        print("Cleared existing documents")
    except Exception as e:
        print(f"Warning: Failed to clear existing documents: {str(e)}")
    
    # Upload new documents
    search_client.upload_documents(documents)
    print(f"Uploaded {len(documents)} documents to index")

def flatten(xss):
    return [x for xs in xss for x in xs]


@cl.on_mcp_connect
async def on_mcp(connection, session: ClientSession):
    result = await session.list_tools()
    tools = [{
        "name": t.name,
        "description": t.description,
        "input_schema": t.inputSchema,
    } for t in result.tools]

    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_tools[connection.name] = tools
    cl.user_session.set("mcp_tools", mcp_tools)


@cl.step(type="tool")
async def call_tool(tool_use):
    tool_name = tool_use.name
    tool_input = tool_use.input

    current_step = cl.context.current_step
    current_step.name = tool_name

    # Identify which mcp is used
    mcp_tools = cl.user_session.get("mcp_tools", {})
    mcp_name = None

    for connection_name, tools in mcp_tools.items():
        if any(tool.get("name") == tool_name for tool in tools):
            mcp_name = connection_name
            break

    if not mcp_name:
        current_step.output = json.dumps(
            {"error": f"Tool {tool_name} not found in any MCP connection"})
        return current_step.output

    mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)

    if not mcp_session:
        current_step.output = json.dumps(
            {"error": f"MCP {mcp_name} not found in any MCP connection"})
        return current_step.output

    try:
        current_step.output = await mcp_session.call_tool(tool_name, tool_input)
    except Exception as e:
        current_step.output = json.dumps({"error": str(e)})

    return current_step.output


@cl.on_chat_start
async def on_chat_start():
 
    # Create kernel
    kernel = Kernel()

    # Define service ID
    service_id = "agent"

    # Create and add chat completion service
    # chat_completion_service = OpenAIChatCompletion(
    #     ai_model_id="gpt-4o-mini",
    #     async_client=client,
    #     service_id=service_id
    # )

    sk_filter = cl.SemanticKernelFilter(kernel=kernel)

    kernel.add_service(AzureChatCompletion(service_id=service_id))
    settings = kernel.get_prompt_execution_settings_from_service_id(
        service_id=service_id)
    settings.function_choice_behavior = FunctionChoiceBehavior.Auto()

 

    # Create a properly instantiated RAGPlugin
    rag_plugin = RAGPlugin(search_client)

    # Add to kernel
    kernel.add_plugin(rag_plugin, plugin_name="RAG")

    # Store in session
    cl.user_session.set("rag_plugin", rag_plugin)

    # Add GitHub MCP plugin
    try:
        # Create GitHub MCP plugin using MCPStdioPlugin
        github_plugin = MCPStdioPlugin(
            name="Github",
            description="Github Plugin",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"]
        )

        # Connect to the GitHub MCP server
        await github_plugin.connect()

        # Add the plugin to the kernel
        kernel.add_plugin(github_plugin)

        # Store the plugin in user session for cleanup later
        cl.user_session.set("github_plugin", github_plugin)

        print("GitHub plugin added successfully")
    except Exception as e:
        print(f"Error adding GitHub plugin: {str(e)}")

    GITHUB_INSTRUCTIONS = """
You are an expert on GitHub repositories. When answering questions, you **must** use the provided GitHub username to find specific information about that user's repositories, including:

*   Who created the repositories
*   The programming languages used
*   Information found in files and README.md files within those repositories
*   Provide links to each repository referenfced in your answers

**Important:** Never perform general searches for repositories. Always use the given GitHub username to find the relevant information. If a GitHub username is not provided, state that you need a username to proceed.
    """

    HACKATHON_AGENT = """
You are an AI Agent Hackathon Strategist specializing in recommending winning project ideas.

Your task:
1. Analyze the GitHub activity of users to understand their technical skills
2. Suggest creative AI Agent projects tailored to their expertise. 
3. Focus on projects that align with Microsoft's AI Agent Hackathon prize categories

When making recommendations:
- Base your ideas strictly on the user's GitHub repositories, languages, and tools
- Give suggestions on tools, languaghes and framweworks to use to build it. 
- Provide detailed project descriptions including architecture and implementation approach
- Explain why the project has potential to win in specific prize categories
- Highlight technical feasibility given the user's demonstrated skills by referencing the specific repositories or languages used.

Formatting your response:
- Provide a clear and structured response that includes:
    - Suggested Project Name
    - Project Description 
    - Potential languages and tools to use
    - Link to each relevant GitHub repository you based your recommendation on

Hackathon prize categories:
- Best Overall Agent ($20,000)
- Best Agent in Python ($5,000)
- Best Agent in C# ($5,000)
- Best Agent in Java ($5,000)
- Best Agent in JavaScript/TypeScript ($5,000)
- Best Copilot Agent using Microsoft Copilot Studio or Microsoft 365 Agents SDK ($5,000)
- Best Azure AI Agent Service Usage ($5,000)
        
"""

    EVENTS_AGENT = """
You are an Event Recommendation Agent specializing in suggesting relevant tech events.

Your task:
1. Review the project idea recommended by the Hackathon Agent
2. Use the search_events function to find relevant events based on the technologies mentioned.
3. NEVER suggest and event that the where there is not a relevant technology that the user has used.
3. ONLY recommend events that were returned by the search_events functionf

When making recommendations:
- IMPORTANT: You must first call the search_events function with appropriate technology keywords from the project
- Only recommend events that were explicitly returned by the search_events function
- Do not make up or suggest events that weren't in the search results
- Construct search queries using specific technologies mentioned (e.g., "Python AI workshop" or "JavaScript hackathon")
- Try multiple search queries if needed to find the most relevant events


For each recommended event:
- Only include events found in the search_events results
- Explain the direct connection between the event and the specific project requirements
- Highlight relevant workshops, sessions, or networking opportunities

Formatting your response:
- Start with "Based on the hackathon project idea, here are relevant events that I found:"
- Only list events that were returned by the search_events function
- For each event, include the exact event details as returned by search_events
- Explain specifically how each event relates to the project technologies

If no relevant events are found, acknowledge this and suggest trying different search terms instead of making up events.
"""

    github_agent = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="GithubAgent",
        instructions=GITHUB_INSTRUCTIONS,
        plugins=[github_plugin]
    )

    hackathon_agent = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="HackathonAgent",
        instructions=HACKATHON_AGENT
    )

    events_agent = ChatCompletionAgent(
        service=AzureChatCompletion(),
        name="EventsAgent",
        instructions=EVENTS_AGENT,
        plugins=[rag_plugin]  # Add the plugin here
    )

    # Create the agent group chat
    agent_group_chat = AgentGroupChat(
        agents=[github_agent, hackathon_agent, events_agent],
        selection_strategy=SequentialSelectionStrategy(
            initial_agent=github_agent),
        termination_strategy=DefaultTerminationStrategy(maximum_iterations=3)
    )

    # Create a new chat history
    chat_history = ChatHistory()

    # Store in user session
    cl.user_session.set("kernel", kernel)
    cl.user_session.set("settings", settings)  # Store settings in session
    cl.user_session.set("chat_completion_service", AzureChatCompletion())
    cl.user_session.set("chat_history", chat_history)
    cl.user_session.set("mcp_tools", {})
    # Store the agent group chat
    cl.user_session.set("agent_group_chat", agent_group_chat)


# Add a cleanup handler for when the session ends
@cl.on_chat_end
async def on_chat_end():
    # Get the GitHub plugin if it exists
    github_plugin = cl.user_session.get("github_plugin")
    if github_plugin:
        try:
            await github_plugin.close()
            print("GitHub plugin closed successfully")
        except Exception as e:
            print(f"Error closing GitHub plugin: {str(e)}")


@cl.on_message
async def on_message(message: cl.Message):
    kernel = cl.user_session.get("kernel")
    chat_completion_service = cl.user_session.get("chat_completion_service")
    chat_history = cl.user_session.get("chat_history")
    settings = cl.user_session.get("settings")
    agent_group_chat = cl.user_session.get("agent_group_chat")
    sk_filter = cl.SemanticKernelFilter(kernel=kernel)


    # Check if the message is requesting a hackathon project recommendation
    user_input = message.content.lower()
    if "recommend" and "github" in user_input:
        sk_filter = cl.SemanticKernelFilter(kernel=kernel)

        # Add user message to chat history
        chat_history.add_user_message(message.content)

        # Add user message to the agent group chat's channel
        await agent_group_chat.add_chat_message(message.content)

        # Create message for response stream - USE ONLY ONE MESSAGE OBJECT
        answer = cl.Message(content="Processing your request using GitHub, Hackathon and Events agents...\n\n")
        await answer.send()

        agent_responses = []
        async for content in agent_group_chat.invoke():
            agent_name = content.name or "Agent"
            response = f"**{agent_name}**: {content.content}"
            agent_responses.append(response)
            await answer.stream_token(f"{response}\n\n")

        # Add the full agent responses to chat history
        full_response = "\n\n".join(agent_responses)
        chat_history.add_assistant_message(full_response)

        # Update the message with all responses
        answer.content = full_response
        await answer.update()
    else:
        # Regular processing for other messages
        # Add user message to history
        chat_history.add_user_message(message.content)

        # Create a Chainlit message for the response stream
        answer = cl.Message(content="")

        async for msg in chat_completion_service.get_streaming_chat_message_content(
            chat_history=chat_history,
            user_input=message.content,
            settings=settings,
            kernel=kernel,
        ):
            if msg.content:
                await answer.stream_token(msg.content)
            # Handle function calls if they occur
            if isinstance(msg, FunctionCallContent):
                function_name = msg.function_name
                function_arguments = msg.arguments
                await answer.stream_token(f"\n\nCalling function: {function_name} with arguments: {function_arguments}\n\n")
            # Handle function results
            if isinstance(msg, FunctionResultContent):
                await answer.stream_token(f"Function result: {msg.content}\n\n")

        # Add the full assistant response to history
        chat_history.add_assistant_message(answer.content)

        # Send the final message
        await answer.send()
