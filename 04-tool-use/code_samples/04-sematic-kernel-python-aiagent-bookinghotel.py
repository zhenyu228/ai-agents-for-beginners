import asyncio
from typing import Annotated
import requests

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents.azure_ai import AzureAIAgent, AzureAIAgentSettings
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole
from semantic_kernel.functions.kernel_function_decorator import kernel_function

ai_agent_settings = AzureAIAgentSettings.create()

# 从环境变量中获取SERP API密钥
SERP_API_KEY = os.getenv('SERPAPI_SEARCH_API_KEY')
BASE_URL = 'https://serpapi.com/search'  # 修正了URL，去掉了默认的引擎参数

# Define Booking Plugin
class BookingPlugin:
    """Booking Plugin for customers"""

    @kernel_function(description="booking hotel")
    def booking_hotel(self, query: Annotated[str, "The name of the city"], check_in_date: Annotated[str, "Hotel Check-in Time"], check_out_date: Annotated[str, "Hotel Check-out Time"]) -> Annotated[str, "Return the result of booking hotel information"]:
        """
        Function to book a hotel.
        Parameters:
        - query: The name of the city
        - check_in_date: Hotel Check-in Time
        - check_out_date: Hotel Check-out Time
        Returns:
        - The result of booking hotel information
        """

        # Define the parameters for the hotel booking request
        params = {
            "engine": "google_hotels",
            "q": query,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": "1",
            "currency": "GBP",
            "gl": "uk",
            "hl": "en",
            "api_key": SERP_API_KEY
        }

        # Send the GET request to the SERP API
        response = requests.get(BASE_URL, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response content as JSON
            response_data = response.json()
            # Return the properties from the response
            if "properties" in response_data:
                return response_data["properties"]
            else:
                return "No hotel properties found in the response."
        else:
            # Return error message if the request failed
            return f"Error: Failed to retrieve hotel information. Status code: {response.status_code}"

    @kernel_function(description="booking flight")
    def booking_flight(self, origin: Annotated[str, "The name of Departure"], destination: Annotated[str, "The name of Destination"], outbound_date: Annotated[str, "The date of outbound"], return_date: Annotated[str, "The date of Return_date"]) -> Annotated[str, "Return the result of booking flight information"]:
        """
        Function to book a flight.
        Parameters:
        - origin: The name of Departure
        - destination: The name of Destination
        - outbound_date: The date of outbound
        - return_date: The date of Return_date
        Returns:
        - The result of booking flight information
        """
        
        # Define the parameters for the outbound flight request
        go_params = {
            "engine": "google_flights",
            "departure_id": origin,  # 修正了参数，使用了函数参数
            "arrival_id": destination,  # 修正了参数，使用了函数参数
            "outbound_date": outbound_date,  # 修正了参数，使用了函数参数
            "return_date": return_date,  # 修正了参数，使用了函数参数
            "currency": "GBP",
            "hl": "en",
            "api_key": SERP_API_KEY  # 使用环境变量
        }

        # Send the GET request for the outbound flight
        go_response = requests.get(BASE_URL, params=go_params)

        # Initialize the result string
        result = ''

        # Check if the outbound flight request was successful
        if go_response.status_code == 200:
            # Parse the response content as JSON
            response_data = go_response.json()
            # Append the outbound flight information to the result
            result += "# Outbound Flight \n " + str(response_data)
        else:
            # Add error message to result
            result += f"Error retrieving outbound flight: Status code {go_response.status_code}\n"

        # Define the parameters for the return flight request
        back_params = {
            "engine": "google_flights",  # 添加了引擎参数
            "departure_id": destination,
            "arrival_id": origin,
            "outbound_date": return_date,  # 修正了参数，对于返程航班这应该是出发日期
            "currency": "GBP",
            "hl": "en",
            "api_key": SERP_API_KEY
        }

        # Send the GET request for the return flight
        back_response = requests.get(BASE_URL, params=back_params)

        # Check if the return flight request was successful
        if back_response.status_code == 200:
            # Parse the response content as JSON
            response_data = back_response.json()
            # Append the return flight information to the result
            result += "\n# Return Flight \n" + str(response_data)
        else:
            # Add error message to result
            result += f"\nError retrieving return flight: Status code {back_response.status_code}"

        # Return the result
        return result

# 创建一个异步函数来执行主要的代码逻辑
async def main():
    async with DefaultAzureCredential() as creds:
        async with AzureAIAgent.create_client(
            credential=creds,
            conn_str=ai_agent_settings.project_connection_string.get_secret_value(),
        ) as client:
            
            # Define the agent's name and instructions
            AGENT_NAME = "BookingAgent"
            AGENT_INSTRUCTIONS = """
            You are a booking agent, help me to book flights or hotels.

            Thought: Understand the user's intention and confirm whether to use the reservation system to complete the task.

            Action:
            - If booking a flight, convert the departure name and destination name into airport codes.
            - If booking a hotel or flight, use the corresponding API to call. Ensure that the necessary parameters are available. If any parameters are missing, use default values or assumptions to proceed.
            - If it is not a hotel or flight booking, respond with the final answer only.
            - Output the results using a markdown table:
            - For flight bookings, separate the outbound and return contents and list them in the order of Departure_airport Name | Airline | Flight Number | Departure Time | Arrival_airport Name | Arrival Time | Duration | Airplane | Travel Class | Price (USD) | Legroom | Extensions | Carbon Emissions (kg).
            - For hotel bookings, list them in the order of Properties Name | Properties description | check_in_time | check_out_time | prices | nearby_places | hotel_class | gps_coordinates.
            """

            try:
                # Create agent definition with the specified model, name, and instructions
                agent_definition = await client.agents.create_agent(
                    model=ai_agent_settings.model_deployment_name,
                    name=AGENT_NAME,
                    instructions=AGENT_INSTRUCTIONS,
                )

                # Create the AzureAI Agent using the client and agent definition
                agent = AzureAIAgent(
                    client=client,
                    definition=agent_definition,
                    # Optionally configure polling options
                    # polling_options=RunPollingOptions(run_polling_interval=timedelta(seconds=1)),
                )

                # Add the BookingPlugin to the agent's kernel
                agent.kernel.add_plugin(BookingPlugin(), plugin_name="booking")

                # Create a new thread for the agent
                thread = await client.agents.create_thread()

                # Define user inputs for the agent to process
                user_inputs = [
                    "Help me book flight tickets and hotel for the following trip London Heathrow LHR Feb 20th 2025 to New York JFK returning Feb 27th 2025 flying economy with British Airways only. I want a stay in a Hilton hotel in New York please provide costs for the flight and hotel"
                ]

                # Process each user input
                for user_input in user_inputs:
                    # Add the user input as a chat message
                    await agent.add_chat_message(
                        thread_id=thread.id, message=ChatMessageContent(role=AuthorRole.USER, content=user_input)
                    )
                    print(f"# User: '{user_input}'")
                    # Invoke the agent for the specified thread
                    async for content in agent.invoke(
                        thread_id=thread.id,
                        temperature=0.2,  # Override the agent-level temperature setting with a run-time value
                    ):
                        if content.role != AuthorRole.TOOL:
                            print(f"# Agent: {content.content}")
            finally:
                # 确保资源清理，即使发生异常
                if 'thread' in locals():
                    await client.agents.delete_thread(thread.id)
                if 'agent' in locals() and hasattr(agent, 'id'):
                    await client.agents.delete_agent(agent.id)

# 在普通Python脚本中运行异步函数
if __name__ == "__main__":
    asyncio.run(main())