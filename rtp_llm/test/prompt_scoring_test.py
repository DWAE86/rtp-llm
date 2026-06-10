import logging
import os
import shlex
import sys
import time
from unittest import TestCase, main
from unittest.mock import patch

import requests

from rtp_llm.test.utils.maga_server_manager import MagaServerManager

logging.basicConfig(level=logging.INFO)


class PromptScoringTest(TestCase):
    server_manager = None

    @classmethod
    def setUpClass(cls):
        import unittest

        checkpoint_path = os.environ.get("CHECKPOINT_PATH")
        if not checkpoint_path:
            raise unittest.SkipTest("CHECKPOINT_PATH not set")
        model_type = os.environ.get("MODEL_TYPE", "qwen_3")

        env_args = {
            "TP_SIZE": "1",
            "DP_SIZE": "1",
            "EP_SIZE": "1",
            "WORLD_SIZE": "1",
            "MAX_SEQ_LEN": "5120",
            "WARM_UP": "0",
            "DEVICE_RESERVE_MEMORY_BYTES": "-2048000000",
            "RESERVER_RUNTIME_MEM_MB": "2048",
            "NCCL_DISABLE_ABORT": "1",
            "FT_DISABLE_CUSTOM_AR": "1",
        }

        cls.server_manager = MagaServerManager(env_args=env_args)
        ret = cls.server_manager.start_server(
            model_path=checkpoint_path,
            model_type=model_type,
            tokenizer_path=checkpoint_path,
        )
        if not ret:
            raise RuntimeError("Server failed to start")
        logging.info("Server is ready")

    @classmethod
    def tearDownClass(cls):
        if cls.server_manager:
            cls.server_manager.stop_server()

    def _post(self, endpoint, payload):
        import json

        ret, response = self.server_manager.visit(
            payload, retry_times=3, endpoint=endpoint
        )
        self.assertTrue(ret, f"Request to {endpoint} failed: {response}")
        if isinstance(response, str):
            response = json.loads(response)
        return response

    def test_01_normal_generation(self):
        """Regression: normal generation still works."""
        data = self._post(
            "/",
            {
                "prompt": "Hello, world!",
                "generate_config": {"max_new_tokens": 5},
            },
        )
        self.assertIn("response", data)
        self.assertTrue(data.get("finished"))
        self.assertTrue(len(data["response"]) > 0)

    def test_02_general_protocol_prompt_logits(self):
        """General protocol returns prompt_logits with correct shape."""
        data = self._post(
            "/",
            {
                "prompt": "The quick brown fox jumps over the lazy dog",
                "generate_config": {
                    "return_prompt_logits": True,
                    "prompt_logits_top_k": 16,
                    "max_new_tokens": 1,
                    "return_target_logprob": True,
                },
            },
        )
        self.assertIn("prompt_logits", data)
        pl = data["prompt_logits"]
        self.assertIn("topk_logprobs", pl)
        self.assertIn("topk_token_ids", pl)
        self.assertIn("target_logprobs", pl)
        self.assertIn("start_pos", pl)
        self.assertIn("end_pos", pl)

        num_positions = len(pl["topk_logprobs"])
        self.assertGreater(num_positions, 0)
        self.assertEqual(len(pl["topk_logprobs"][0]), 16)
        self.assertEqual(len(pl["topk_token_ids"][0]), 16)
        self.assertEqual(len(pl["topk_token_ids"]), num_positions)
        self.assertEqual(len(pl["target_logprobs"]), num_positions - 1)

        for row in pl["topk_logprobs"]:
            for v in row:
                self.assertLessEqual(v, 0.0)

        for row in pl["topk_logprobs"]:
            for i in range(len(row) - 1):
                self.assertGreaterEqual(row[i], row[i + 1])

        for row in pl["topk_token_ids"]:
            for t in row:
                self.assertIsInstance(t, int)
                self.assertGreaterEqual(t, 0)

    def test_03_general_protocol_range_clipping(self):
        """Range clipping (start/end) works correctly."""
        data = self._post(
            "/",
            {
                "prompt": "The quick brown fox jumps over the lazy dog and runs away",
                "generate_config": {
                    "return_prompt_logits": True,
                    "prompt_logits_top_k": 8,
                    "prompt_logits_start": 2,
                    "prompt_logits_end": 6,
                    "max_new_tokens": 1,
                    "return_target_logprob": True,
                },
            },
        )
        pl = data["prompt_logits"]
        self.assertEqual(pl["start_pos"], 2)
        self.assertEqual(pl["end_pos"], 6)
        self.assertEqual(len(pl["topk_logprobs"]), 4)
        self.assertEqual(len(pl["topk_logprobs"][0]), 8)

    def test_04_openai_protocol_prompt_logprobs(self):
        """OpenAI protocol returns prompt_logprobs."""
        data = self._post(
            "/v1/chat/completions",
            {
                "model": "qwen",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 2+2? Answer with just the number.",
                    }
                ],
                "prompt_logprobs": 10,
                "max_tokens": 1,
            },
        )
        self.assertIn("prompt_logprobs", data)
        pl = data["prompt_logprobs"]
        self.assertIsNotNone(pl)
        self.assertIn("topk_logprobs", pl)
        self.assertIn("topk_token_ids", pl)
        self.assertIn("target_logprobs", pl)
        self.assertIn("start_pos", pl)
        self.assertIn("end_pos", pl)

        num_positions = len(pl["topk_logprobs"])
        self.assertGreater(num_positions, 0)
        self.assertEqual(len(pl["topk_logprobs"][0]), 10)

    def test_05_no_prompt_logits_when_disabled(self):
        """Normal request does not include prompt_logits."""
        data = self._post(
            "/",
            {
                "prompt": "Hello",
                "generate_config": {"max_new_tokens": 3},
            },
        )
        pl = data.get("prompt_logits")
        self.assertTrue(pl is None or pl == {})


if __name__ == "__main__":
    main()
