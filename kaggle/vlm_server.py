#!/usr/bin/env python3
"""Local OpenAI-compatible chat server for a Qwen vision-language model from Hugging Face.

CoRe's VLM client (RRM/preference_label.py) talks to it exactly as it talks to Gemini:
POST /v1/chat/completions with text and base64 PNG images. Requests that arrive together
are batched on the GPU (CoRe sends a feedback round's queries in parallel).

  CUDA_VISIBLE_DEVICES=1 python kaggle/vlm_server.py --model Qwen/Qwen3-VL-4B-Instruct --port 8008
  # 8B (the model the CoRe paper tested) needs 4-bit on a 16 GB T4 and is ~2x slower:
  CUDA_VISIBLE_DEVICES=1 python kaggle/vlm_server.py --model Qwen/Qwen3-VL-8B-Instruct --load-in-4bit
"""
import argparse
import base64
import io
import json
import queue
import threading
import time
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image


def to_qwen_messages(messages):
    """OpenAI chat messages -> Qwen chat-template messages (images decoded to PIL)."""
    out = []
    for m in messages:
        content = m.get('content')
        parts = []
        if isinstance(content, str):
            parts.append({'type': 'text', 'text': content})
        else:
            for p in content or []:
                if p.get('type') == 'text':
                    parts.append({'type': 'text', 'text': p['text']})
                elif p.get('type') == 'image_url':
                    url = p['image_url']['url']
                    data = base64.b64decode(url.split(',', 1)[1])
                    parts.append({'type': 'image', 'image': Image.open(io.BytesIO(data)).convert('RGB')})
        out.append({'role': m['role'], 'content': parts})
    return out


class Generator:
    def __init__(self, model_name, load_in_4bit, max_batch, max_new_tokens, batch_wait, device='auto'):
        from transformers import AutoModelForImageTextToText, AutoProcessor
        gpu = torch.cuda.is_available() and device in ('auto', 'cuda')
        mps = not gpu and device in ('auto', 'mps') and torch.backends.mps.is_available()  # Mac, testing only
        kwargs = dict(device_map='cuda:0' if gpu else ('mps' if mps else 'cpu'), attn_implementation='sdpa')
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                                bnb_4bit_compute_dtype=torch.float16)
        else:
            # T4 has no bfloat16: use float16 on GPU (float32 on CPU, for testing only)
            kwargs['dtype'] = torch.float16 if (gpu or mps) else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = 'left'
        self.max_batch, self.max_new_tokens, self.batch_wait = max_batch, max_new_tokens, batch_wait
        self.queue = queue.Queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, messages, max_tokens):
        fut = Future()
        self.queue.put((to_qwen_messages(messages), max_tokens, fut))
        return fut

    def _loop(self):
        while True:
            batch = [self.queue.get()]
            deadline = time.time() + self.batch_wait
            while len(batch) < self.max_batch:
                try:
                    batch.append(self.queue.get(timeout=max(0.0, deadline - time.time())))
                except queue.Empty:
                    break
            # requests with images and text-only requests are generated in separate batches
            has_img = lambda conv: any(p['type'] == 'image' for m in conv for p in m['content'])
            for group in ([b for b in batch if has_img(b[0])], [b for b in batch if not has_img(b[0])]):
                if not group:
                    continue
                try:
                    for (_, _, fut), res in zip(group, self._generate(group)):
                        fut.set_result(res)
                except Exception as e:  # e.g. out of memory: answer each request in the group with the error
                    for _, _, fut in group:
                        fut.set_exception(e)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    @torch.inference_mode()
    def _generate(self, batch):
        convs = [b[0] for b in batch]
        max_new = min(max(b[1] or self.max_new_tokens for b in batch), self.max_new_tokens)
        inputs = self.processor.apply_chat_template(convs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                    processor_kwargs=dict(return_tensors='pt', padding=True))
        inputs = inputs.to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        prompt_len = inputs['input_ids'].shape[1]
        texts = self.processor.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
        n_prompt = int(inputs['attention_mask'].sum(dim=1).float().mean().item())
        return [(t.strip(), n_prompt, int((o[prompt_len:] != self.processor.tokenizer.pad_token_id).sum()))
                for t, o in zip(texts, out)]


def make_handler(gen, model_name):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip('/') in ('/health', '/v1/models'):
                self._send(200, {'object': 'list', 'data': [{'id': model_name, 'object': 'model'}]})
            else:
                self._send(404, {'error': {'message': 'not found'}})

        def do_POST(self):
            if not self.path.rstrip('/').endswith('/chat/completions'):
                return self._send(404, {'error': {'message': 'not found'}})
            req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))))
            n = int(req.get('n') or 1)
            try:
                futs = [gen.submit(req['messages'], req.get('max_tokens') or req.get('max_completion_tokens'))
                        for _ in range(n)]
                res = [f.result() for f in futs]
            except Exception as e:
                return self._send(500, {'error': {'message': f'{type(e).__name__}: {e}'}})
            choices = [{'index': i, 'message': {'role': 'assistant', 'content': t}, 'finish_reason': 'stop'}
                       for i, (t, _, _) in enumerate(res)]
            pt, ct = res[0][1], sum(r[2] for r in res)
            self._send(200, {'id': f'local-{time.time_ns()}', 'object': 'chat.completion', 'created': int(time.time()),
                             'model': model_name, 'choices': choices,
                             'usage': {'prompt_tokens': pt, 'completion_tokens': ct, 'total_tokens': pt + ct}})

        def log_message(self, *args):
            pass
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct')
    p.add_argument('--port', type=int, default=8008)
    p.add_argument('--load-in-4bit', action='store_true')
    p.add_argument('--max-batch', type=int, default=4)
    p.add_argument('--max-new-tokens', type=int, default=768)
    p.add_argument('--batch-wait', type=float, default=0.3, help="seconds to wait for more requests to batch")
    p.add_argument('--device', default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
    args = p.parse_args()
    t0 = time.time()
    gen = Generator(args.model, args.load_in_4bit, args.max_batch, args.max_new_tokens, args.batch_wait, args.device)
    print(f"[vlm] {args.model} loaded in {time.time() - t0:.0f}s; serving on http://127.0.0.1:{args.port}/v1", flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(gen, args.model)).serve_forever()


if __name__ == '__main__':
    main()
