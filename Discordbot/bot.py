import os
import discord
import openai
from discord.ext import commands
from dotenv import load_dotenv
import asyncio

load_dotenv()
TOKEN = os.getenv('DISCORD_BOT_TOKEN')
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

def setup_openai():
    openai.api_key = OPENAI_API_KEY

setup_openai()

# Create a dictionary to store messages in threads
thread_messages = {}

# Create a queue to manage incoming messages
message_queue = asyncio.Queue()

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name}')
    
    # Create and start the background processing task
    bot.loop.create_task(process_queue())

async def process_queue():
    while True:
        # Get the next message from the queue
        channel, author, user_message = await message_queue.get()

        try:
            if user_message:
                # Check if the message is in a thread or the bot is mentioned
                if isinstance(channel, discord.Thread) or bot.user.mentioned_in(user_message):
                    async with channel.typing():
                        # Save the message to the thread_messages dictionary
                        thread_id = channel.id
                        if thread_id not in thread_messages:
                            thread_messages[thread_id] = []

                        thread_messages[thread_id].append(user_message.content)

                        # Use the stored messages as context
                        context_messages = [{"role": "user", "content": msg} for msg in thread_messages[thread_id]]

                        response = openai.ChatCompletion.create(
                            model="gpt-4",
                            messages=[
                                {"role": "system", "content": "you are a helpful assistant."},
                                *context_messages,  # Include previous messages as context
                                {"role": "user", "content": user_message.content}
                            ],
                        )

                        response_content = response['choices'][0]['message']['content'].strip()

                        await channel.send(f"{author.mention}, {response_content}")
                else:
                    # If not responding, do not display typing status
                    await asyncio.sleep(3)  # Sleep to simulate bot processing

        except Exception as e:
            print(f"An error occurred: {e}")

@bot.event
async def on_message(message):
    try:
        if not message.author.bot:
            # Add the message to the processing queue
            await message_queue.put((message.channel, message.author, message))
            
    except Exception as e:
        print(f"An error occurred: {e}")

bot.run(TOKEN)
