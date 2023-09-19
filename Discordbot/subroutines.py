import os
from dotenv import load_dotenv
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

def uberthink (messages, responses = [], context = {}, depth = 0):
    depth += 1
    if depth > 5: return responses
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "you are a helpful assistant."},
            *context_messages,  # Include previous messages as context
            {"role": "user", "content": user_message.content}
            ],
        functions=subroutines.functions_spec(),
        function_call="auto",
    )
    message = response["choices"][0]["message"]
    if (message.get("function_call")):
        name = message["function_call"]["name"]
        args = json.loads(message["function_call"]["arguments"])
        result = apply_function_call(name, args, context)
        messages.append(message) # append message
        messages.append({ "role": "function", "name": name, "content": result})
    else:
        await context.channel.send(responses)

def apply_function_call(name, args, context):
    print(f"Function call invoked {name} {args}")

def functions_spec(select="all"):
    # TODO: The gpt-plus plugin "ask the code" leaked their callspec earlier;
    # Remember it was something like: https://askthecode.ai/.well-known/plugin.yaml
    # but i can't remember the subdomain.
    list_repo = {
        "name": "list_repo",
        "description": """
            Lists files in a public Github repository.
        """.strip(),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": """
                        The path to a repository, example:
                        given 'https://github.com/telamon/picofeed' the path is 'telamon/picofeed'
                    """.strip()
                }
            },
            "required": ["path"]
        }
    }
    repo_fetch_files = {
        "name": "repo_fetch_files",
        "description": """
            Fetches content for given files, don't fetch more than 5 files at a time.
            Read them carefully before fetching a new.
        """.strip(),
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": """
                        Select paths returned by list_repo; example:
                        ["telamon/picofeed/README.md", "telamon/picofeed/index.js"]
                    """.strip()
                }
            },
            "required": ["paths"]
        }
    }

    discord_add_reaction = {
        "name": "repo_fetch_files",
        "description": """
            Sometimes the only right answer is a reaction,
            use to add an emoji to an existing message that you feel for.
        """.strip(),
        "parameters": {
            "type": "object",
            "properties": {
                "reaction": {
                    "type": "string",
                    "description": "A single Emoji"
                }
            },
            "required": ["reaction"]
        }
    }
    return [
        discord_add_reaction
    ]
