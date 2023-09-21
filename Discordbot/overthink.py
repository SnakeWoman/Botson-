import os
import inspect
import re
import json
from asyncio import get_event_loop
from functools import partial
from typing import NewType
from dotenv import load_dotenv
load_dotenv()

Context = NewType('Context', dict)

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

    async def overthink(self, messages: list, **context):
        """Perform recursive generation
        This method loops back on when function_call is requested.

        Args:
            messages (list): The initial list of messages
            context (kwargs): passed through to output() # TODO: also actions?
        """
        _depth = 0
        _generated = []
        if '_depth' in context:
            _depth = context.pop('_depth')
        if '_generated' in context:
            _generated = context.pop('_generated')

        self.dbg(f"Overthink(ROUND{_depth}, {context})")
        if _depth < self.max_depth:
            message = await self.think([
                { "role": "system", "content": self.system },
                *messages,
                *_generated
            ])
            _generated.append(message)

            if message.get('function_call'):
                name = message["function_call"]["name"]
                arguments = json.loads(message["function_call"]["arguments"])
                self.dbg(f"RUNNING[{name}]({arguments})", message)

                if name in self.actions:
                    result = await self._invoke_action(name, arguments, context)
                    if result is True: result = "Done!"
                    if result is not None and result is not False:
                        if not isinstance(result, str):
                            result = json.dumps(result)
                        # TODO: handle binary/attachments results, images, audio, video what not.
                        _generated.append(message) # append message
                        # TODO: JSON stringify result
                        _generated.append({ "role": "function", "name": name, "content": result})

                        # Recurse with new generated output
                        _depth+=1
                        return await self.overthink(messages, **{
                            **context,
                            "_depth": _depth,
                            "_generated": _generated
                        })

        # End of thought
        await self.output(_generated, context)
        return {**context, "depth": _depth, "messages": messages, "generated": _generated }

    async def _invoke_action(self, name, arguments, context):
        action = self.actions[name]
        sig = inspect.signature(action)

        has_context_arg = len([k for k, v in sig.parameters.items() if v.annotation is Context])
        is_async = inspect.iscoroutinefunction(action)

        prepped_args = (context, ) if has_context_arg else ()
        curry_action = partial(action, *prepped_args, **arguments)
        if is_async:
            return await curry_action()
        else:
            loop = get_event_loop()
            return await loop.run_in_executor(None, curry_action)

    # overidden by impl; OpenAI/GPT4All/Vicuna
    async def think(self, messages: list):
        raise NotImplementedError()

    # override this method with and do channel.send()
    async def output(self, generated: list, ctx):
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
        self.model = kwargs.get('model', 'gpt-4')
        self.function_call = kwargs.get('function_call', 'auto') # Set to 'never' to disable function calling
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

    # Filter out all context-vars
    args = [(k, v) for k, v in sig.parameters.items() if v.annotation is not Context]

    for param, details in args:
        # Extract type annotations from signature
        param_type = details.annotation
        if param_type is Context: continue # skip Ctx args
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
    required = [k for k, v in args if v.default == v.empty]

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
