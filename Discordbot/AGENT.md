```
   _
||/ \   _ .__|_|_  _     _ |__|_
oo\_/\/(/_|  |_| |(_)|_|(_|| ||_
                         _|
```

## Install

```bash
pip3 install -r requirements.in
```

## Create Agent

An agent that implements OpenAI

```python
from overthink import AIAgent

class PrintAgent(AIAgent):
    # The agent generate multiple messages
    async def output(self, messages, context):
        for msg in messages:
            print(f'Output: {msg}')

# Initialize
agent = PrintAgent()

# Configure system
agent.system = """
You're a forgotten typewriter left behind in an attic.
"""

# Start generation messages
messages = [
    { 'role': 'user', content: 'Hello, what year is this?' }
]
await agent.overthink(messages)
```

Any passed kwargs to `overthink` will be available as `context` in both `output()` func
and actions during the generation session:

```python
from overthink import Context

def pm_user(ctx: Context, messages):
    user_id = ctx.get('user_id') # 125

agent.add_action(pm_user)
agent.overthink(messages, user_id = 125)
```

## Actions

An action is a function with typed arguments and documentation/description
to hint the model when to call the action.

Example:

```python
from datetime import datetime

def look_at_clock():
    """It shows you the current time"""
    return datetime.now().isoformat() # strings are preferred

agent.add_action(look_at_clock)
```
Ask the agent "what time is it?" and it'll call the function then tell you the time.

Example with custom arguments:

```python
from overthink import describe

@describe(
    message="The message to broadcast",
    song="Title - Artist of the song",
    tweet="Post to twitter?",
    fb="Post to facebook?"
)
def notify_everyone_and_play_music(message: str, song:str, tweet:bool, fb:bool):
    """This function broadcats a message to social media and
       then starts playing the song of your choice,
       be careful!
    """,
    if (tweet): twitter.post(message)
    if (fb): fb.create_post(message)

    song = media_player.search(song)
    song.play()
    return "Done"

agent.add_action(notify_everyone_and_play_music)

await agent.overthink([
    { "role": "user", "content": "Whoa! We just released 1.0!! Spread the word and put on an appropriate tune! :party:" }
])
```

### Action return types

- `str` Strings are fed back to the model to generate additional messages, the model can read complex data as json
- `False` or `None` stops generation causing overthink to return.
- `True` signals "done" back to the model and continues generation

---

License: AGPLv3
