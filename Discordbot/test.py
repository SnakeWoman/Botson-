import unittest
import overthink
from pdb import set_trace

# --------------------------------
# ----------- DEFINE AGENTS -----------
# --------------------------------

# Agent that has a hardcoded reply
class DummyAgent(overthink.Agent):
    async def think(self, messages):
        [print(f"in< {m}") for m in messages]
        return { "role": "assistant", "content": "I'm hardcoded, i don't know" }

    async def output(self, messages):
        [print(f"out> {m}") for m in messages]

# Agent that always reacts with thumbs-up
class ActionAgent(DummyAgent):
    async def think(self, messages):
        return {
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": "emoji_reaction",
                "arguments": "{\"emoji\":\"👍\", \"stop\":true}"
            }
        }

# OpenAI Agent that prints results
class TermAIAgent(overthink.AIAgent):
    async def output(self, messages):
        [print(f"out> {m}") for m in messages]

# Define a test action
@describe(
    """
        Sometimes the only right answer is a reaction,
        call to add an emoji to the user's message.
    """,
    emoji = """
      A single emoji representing emotional reaction,
      avoid using the flags an transport emojis.
    """,
    stop = "Signals that the reaction is the end of response"
)
async def emoji_reaction(emoji: str, stop: bool = False):
    s = f"AI is feeling {emoji}"
    print(s) # pretend to be reaction
    if not stop:
        return s

# --------------------------------
# ----------- DEFINE TESTS -----------
# --------------------------------

class TestOverthinkAgent(unittest.IsolatedAsyncioTestCase):
    async def test_dummy(self):
        messages = [
            { "role": "user", "content": "@nombo> Yeah, i think so" },
            { "role": "user", "content": "@molly> What is it that you don't like about blueberries?" }
        ]
        agent = DummyAgent()
        depth, generated = await agent.overthink(messages)
        self.assertEqual(depth, 0)

    async def test_ai(self):
        messages = [
            { "role": "user", "content": "@nombo> Yeah, i think so" },
            { "role": "user", "content": "@molly> What is it that you don't like about blueberries?" },
            { "role": "user", "content": "@molly> @assistant, please emoji react?" }
        ]
        agent = ActionAgent()
        # Attach dummy action
        agent.add_action(overthink.emoji_reaction)
        depth, generated = await agent.overthink(messages)
        self.assertEqual(depth, 0)

class TestJSONSchemaGenerator(unittest.TestCase):
    def test_pedant_doc(self):
        @overthink.describe(a = "CoolString", b = "BoldBool")
        def test_func(a: str, b: bool = False):
            """A func"""
            print(a, b)
        spec = overthink.to_json_schema(test_func)
        self.assertEqual(spec['name'], 'test_func')
        self.assertEqual(spec['description'], 'A func')
        self.assertTrue('a' in spec['parameters']['required'])
        self.assertTrue('a' in spec['parameters']['properties'])
        x = spec['parameters']['properties']['a']
        self.assertEqual(x['type'], 'string')
        self.assertEqual(x['description'],'CoolString')
        self.assertTrue('b' in spec['parameters']['properties'])
        y = spec['parameters']['properties']['b']
        self.assertEqual(y['type'], 'boolean')
        self.assertEqual(y['description'], 'BoldBool')

    def test_desc2(self):
        @overthink.describe("noop", n = "hops")
        def test_fn2(n: int):
            pass
        spec = overthink.to_json_schema(test_fn2)
        self.assertEqual(spec['description'], 'noop')
        self.assertEqual(spec['parameters']['properties']['n']['description'], 'hops')

if __name__ == '__main__':
  unittest.main()


