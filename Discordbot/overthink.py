import os
import inspect
import re
import json
from dotenv import load_dotenv
load_dotenv()

class Agent():
    actions = {}
    system = "You're a helpful agent"
    disable_actions = 'none'
    max_depth = 5

    def __init__(self, **kwargs):
        if 'max_depth' in kwargs: self.max_depth = kwargs['max_depth']
        if 'disable_actions' in kwargs: self.disable_actions = kwargs['disable_actions']
        if 'system' in kwargs: self.max_depth = kwargs['system']
        if 'actions' in kwargs:
            for action in actions: self.add_action(action)

    def add_action(self, action: callable):
        spec = to_json_schema(action)
        name = spec['name']
        self.actions[name] = action

    def dbg (self, *args):
        print("DBG>", *args)

    async def overthink(self, messages = [], generated = [], depth = 0):
        self.dbg(f"overthink(d:{depth})")
        if (depth >= self.max_depth):
            await self.output(generated)
            return depth, generated

        message = await self.think([
            { "role": "system", "content": self.system },
            *messages,
            *generated
        ])
        generated.append(message)

        if message.get('function_call'):
            name = message["function_call"]["name"]
            args = json.loads(message["function_call"]["arguments"])
            self.dbg(f"ACTION[{name}]({args})", message)
            if (self.actions.get(name)):
                result = await self.actions[name](**args)
                if result is None:
                    await self.output(generated)
                    return (depth, generated)

                # TODO: handle binary/attachments results, images, audio, video what not.
                generated.append(message) # append message
                generated.append({ "role": "function", "name": name, "content": result})
                # Recurse with added info
                return await self.overthink(messages, generated, depth + 1)

        # Terminal thought
        await self.output(generated)
        return depth, generated

    # overidden by impl; OpenAI/GPT4All/Vicuna
    async def think(self, messages):
        raise NotImplementedError()

    # override this method with and do channel.send()
    async def output(self, generated):
        raise NotImplementedError()

    # Returns a registered actions as json-type-spec thingy
    async def functions_spec(self):
        return [to_json_schema(f) for _, f in self.actions.items()]

# Agent that uses OpenAI
class AIAgent(Agent):
    model ='gpt-4'
    function_call='auto'
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'model' in kwargs: self.model = kwargs['model']
        if 'function_call' in kwargs: self.function_call = kwargs['function_call']
        import openai
        openai.api_key = kwargs['api_key'] if 'api_key' in kwargs else os.getenv("OPENAI_API_KEY")
        self.openai = openai

    async def think(self, messages):
        functions = await self.functions_spec()
        response = self.openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            functions=functions,
            function_call=self.function_call,
        )
        message = response["choices"][0]["message"]
        term_stop = response["choices"][0]["finish_reason"]
        print(f"finish_reason: {term_stop}")
        return message

def describe(description = None, **annotations):
    def inner(func):
        if not hasattr(func, '__action__'):
            func.__action__ = {}
        if description is not None:
            func.__action__['_desc'] = description
        for param, annotation in annotations.items():
            func.__action__[param] = annotation.strip()
        return func
    return inner

def to_json_schema(fn: callable):
    """Reads a functions name/arguments and generates JSON-schema using inspect"""
    sig = inspect.signature(fn)
    doc = inspect.getdoc(fn)
    if doc is not None: doc = doc.strip()
    attrs = fn.__action__ if hasattr(fn, '__action__') else {}
    fn_description = attrs['_desc'] if '_desc' in attrs else doc

    # Extracting parameter details
    properties = {}
    for param, details in sig.parameters.items():
        # Extract type annotations from signature
        param_type = details.annotation
        if param_type == str:
            param_type = "string"
        elif param_type == bool:
            param_type = "boolean"

        # Extract parameter descriptions from docstring
        description = attrs[param] if param in attrs else ''
        properties[param] = {
            "type": param_type,
            "description": description
        }

    # Required parameters (those without default values)
    required = [k for k, v in sig.parameters.items() if v.default == v.empty]

    # Format everything into a JSON schema
    spec = {
        "name": fn.__name__,
        "description": fn_description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }
    return spec
