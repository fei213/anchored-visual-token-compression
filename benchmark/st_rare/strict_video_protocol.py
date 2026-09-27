"""Controlled single-video path: projector/pooling, exact budget, one newline/frame.

This is a controlled evaluation protocol, not a claim to reproduce the upstream
model's default video layout. The learned compressors (Anchor / Anchor+Div) are
registered at eval time and dispatched through this SAME protocol, so the
generation/scoring path is identical to every baseline -- only the selector
differs.
"""
import math
import torch
from llava.compression.selectors import create_selector
from llava.compression.packing import pack_tokens

# Additive hook: a learned compressor is registered at eval time and dispatched
# through this SAME protocol, so the generation/scoring path is identical to
# every baseline -- only the selector differs.
LEARNED_SELECTORS = {}   # name -> callable(features, newline, content) -> (packed, meta)


def register_learned(name, fn):
    LEARNED_SELECTORS[name] = fn


# Baseline methods implemented in llava/compression/selectors.py, plus the two
# learned compressors registered via register_learned("learned_anchor"/"learned_div").
METHODS = ['controlled_full', 'uniform', 'random', 'kmeans_medoid', 'rare_only',
           'temporal_only', 'st_rare', 'st_rare_balanced']


def select_visual(features, newline, method, ratio=0.25, weight=1.0, seed=0, k=64):
    if method not in METHODS and method not in LEARNED_SELECTORS:
        raise ValueError(method)
    t, n, d = features.shape
    side = math.isqrt(n)
    assert side * side == n and 0 < ratio <= 1
    native = t * (n + 1)
    target = native if method == 'controlled_full' else math.floor(native * ratio)
    content = target - t
    if content < t:
        raise ValueError('Budget must retain at least one content token per frame')

    if method in LEARNED_SELECTORS:
        packed, lmeta = LEARNED_SELECTORS[method](features, newline, content)
        assert packed.shape == (target, d), (packed.shape, target, d)
        meta = dict(protocol='controlled_frame_newline_v1', method=method,
                    baseline_visual_tokens=native, target_visual_tokens=target,
                    actual_visual_tokens=len(packed), actual_retention=len(packed) / native,
                    effective_selector=lmeta, content_tokens=content, separator_tokens=t,
                    per_frame_count=lmeta.get('per_frame_count'))
        return packed, meta

    if method == 'controlled_full':
        indices = [torch.arange(n, device=features.device) for _ in range(t)]
        config = {}
    else:
        minimum = min(4, content // t)
        if method == 'st_rare_balanced':
            minimum = min(k, n, max(minimum, content // (2 * t)))
        config = dict(method='st_rare' if method == 'st_rare_balanced' else method,
                      kmeans_k=min(k, n), seed=seed, novelty_weight=weight,
                      min_tokens_per_frame=minimum)
        _, meta = create_selector(config['method'], config).select(features, ratio, newline, side, side)
        indices = [[] for _ in range(t)]
        for token in meta['token_metadata']:
            if not token['is_separator']:
                indices[token['frame_id']].append(token['token_id'])
        indices = [torch.tensor(x, device=features.device, dtype=torch.long) for x in indices]

    assert sum(x.numel() for x in indices) == content
    assert all(len(x.unique()) == len(x) for x in indices)
    assert all(x.numel() >= 1 for x in indices)
    packed, meta = pack_tokens(features, indices, newline, side, side, add_separators=True)
    assert packed.shape == (target, d)
    meta.update(protocol='controlled_frame_newline_v1', method=method, baseline_visual_tokens=native,
                target_visual_tokens=target, actual_visual_tokens=len(packed),
                actual_retention=len(packed) / native, effective_selector=config,
                selected_indices=[x.tolist() for x in indices])
    return packed, meta


def generate_from_visual(model, tokenizer, question, visual, max_tokens=128, min_tokens=None):
    from transformers import Qwen2ForCausalLM
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    conv = conv_templates['qwen_1_5'].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + '\n' + question)
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').to(model.device)
    positions = (ids == IMAGE_TOKEN_INDEX).nonzero().flatten()
    assert len(positions) == 1
    at = int(positions[0])
    embed = model.get_model().embed_tokens
    inputs = torch.cat([embed(ids[:at]), visual, embed(ids[at + 1:])]).unsqueeze(0)
    expected = len(ids) - 1 + len(visual)
    assert inputs.shape[1] == expected
    mask = torch.ones((1, expected), device=inputs.device, dtype=torch.long)
    if min_tokens is not None and not 0 <= min_tokens <= max_tokens:
        raise ValueError('min_tokens must be in [0, max_tokens]')
    kwargs = dict(do_sample=False, max_new_tokens=max_tokens, use_cache=True)
    if min_tokens is not None:
        kwargs['min_new_tokens'] = min_tokens
    output = Qwen2ForCausalLM.generate(model, inputs_embeds=inputs, attention_mask=mask, **kwargs)
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0], dict(
        final_sequence_tokens=expected, text_tokens=len(ids) - 1, generated_tokens=output.shape[1])
