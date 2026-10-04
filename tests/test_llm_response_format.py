import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from agents.llm import OpenAIChatClient
from agents.ocm_agent.prediction import PREDICTION_RESPONSE_FORMAT


class ResponseFormatTest(unittest.TestCase):
    def test_schema_is_forwarded_and_omitted_when_not_requested(self):
        client = OpenAIChatClient(api_key="test", base_url="http://localhost:1/v1", timeout=1)
        client._create_with_retry = Mock(return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}", tool_calls=None))],
            usage=None,
        ))
        for response_format in (PREDICTION_RESPONSE_FORMAT, None):
            client.create_chat_completion(model="test", messages=[], temperature=0,
                                          response_format=response_format)
            kwargs = client._create_with_retry.call_args.kwargs
            if response_format is None:
                self.assertNotIn("response_format", kwargs)
            else:
                self.assertEqual(kwargs["response_format"], response_format)
