import unittest
import overthink
from pdb import set_trace
from overthink import Agent, AIAgent, describe, Context

# Fixes much text in-between runs
print("""
   _
||/ \   _ .__|_|_  _     _ |__|_
oo\_/\/(/_|  |_| |(_)|_|(_|| ||_
                         _|
""")

# --------------------------------
# ----------- DEFINE AGENTS -----------
# --------------------------------

# Agent that has a hardcoded reply
class DummyAgent(Agent):
    async def think(self, messages):
        [print(f"in< {m}") for m in messages]
        return { "role": "assistant", "content": "I'm hardcoded, i don't know" }

    async def output(self, messages, ctx):
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
class TermAIAgent(AIAgent):
    async def output(self, messages, ctx):
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
        result = await agent.overthink(messages)
        self.assertEqual(result['depth'], 0)

    async def test_ai(self):
        messages = [
            { "role": "user", "content": "@nombo> Yeah, i think so" },
            { "role": "user", "content": "@molly> What is it that you don't like about blueberries?" },
            { "role": "user", "content": "@molly> @assistant, please emoji react?" }
        ]
        agent = ActionAgent()
        # Attach dummy action
        agent.add_action(emoji_reaction)
        result = await agent.overthink(messages)
        self.assertEqual(result['depth'], 0)

    async def test_context(self):
        messages = [
            { "role": "user", "content": "@nombo> Yeah, i think so" },
        ]
        class ContextAgent(Agent):
            async def think(self, messages):
                return {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "action_with_ctx",
                        "arguments": "{\"a\":0}"
                    }
                }
            async def output(self, messages, ctx):
                # Ensure our var is availabe in output
                if ctx.get('channel', 0) != 999: raise RuntimeError('output() context missing key')

        # This feels brittle AF
        @describe("An action wants to know disco-channel-id", a = "some number")
        def action_with_ctx(random_name: Context, a: int):
            if a != 0: raise RuntimeError('Expected var `a` to be 0')
            if random_name.get('channel', 0) != 999: raise RuntimeError('action() context missing key')
            return None

        # Ensure context is not included in json-schema
        spec = overthink.to_json_schema(action_with_ctx)
        self.assertFalse('random_name' in spec['parameters']['properties'])
        self.assertFalse('random_name' in spec['parameters']['required'])

        # Boot the agent
        agent = ContextAgent()
        agent.add_action(action_with_ctx)


        # Pretending we wanna keep track of discord channel-id
        result = await agent.overthink(messages, channel=999)
        self.assertEqual(result["channel"], 999) # included in result


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


