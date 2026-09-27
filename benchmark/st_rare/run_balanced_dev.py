"""Frozen balanced development screening with reusable visual-feature cache."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=ROOT];sys.path.append(str(ROOT))
import transformers
import torch
import numpy as np
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images
from benchmark.st_rare.video_utils import load_video
from benchmark.st_rare.strict_video_protocol import select_visual,generate_from_visual

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'artifacts/st_rare/balanced_dev_v1/manifest.jsonl')
    p.add_argument('--methods',nargs='+',default=['controlled_full','uniform','st_rare'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    samples=[json.loads(x) for x in a.manifest.read_text().splitlines()]
    assert all(s['split']=='dev' for s in samples)
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    tokenizer,model,processor,_=load_pretrained_model('lmms-lab/LLaVA-Video-7B-Qwen2',None,'llava_qwen',device_map='auto',attn_implementation='sdpa')
    model.eval();model.config.image_aspect_ratio='square';model.config.image_grid_pinpoints=None
    protocol=dict(manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),methods=a.methods,frames=16,max_new_tokens=512,ratio=0.25,seed=0,
                  torch=torch.__version__,transformers=transformers.__version__,model_config=model.config.to_dict(),samples=samples,
                  code_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['benchmark/st_rare/run_balanced_dev.py','benchmark/st_rare/strict_video_protocol.py','benchmark/st_rare/video_utils.py','llava/compression/selectors.py']},
                  sampling='First QA of each named task type, deduplicated by question; no selection by scores',
                  timing_caveat='Shared visual encoder and one selection per video/method; timings are diagnostic, not end-to-end speed claims')
    (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2,default=str))
    cache=a.output/'features';cache.mkdir()
    count=0
    with torch.inference_mode():
        for i,s in enumerate(samples):
            qa=json.loads(Path(s['qa_file']).read_text());questions=[];seen=set()
            for kind,pairs in qa.items():
                if not kind.endswith('_qa_pairs') or not isinstance(pairs,list) or not pairs:continue
                pair=pairs[0]
                if 'Q' not in pair or 'A' not in pair or pair['Q'] in seen:continue
                questions.append((kind,pair));seen.add(pair['Q'])
            assert questions
            frames=load_video(s['video_path'],16);assert frames is not None
            pixels=process_images(frames,processor,model.config)
            if isinstance(pixels,list):pixels=torch.stack(pixels)
            pixels=pixels.to(model.device,dtype=next(model.get_vision_tower().parameters()).dtype)
            torch.cuda.synchronize();start=time.monotonic()
            vision=model.get_vision_tower()(pixels);projector=model.get_model().mm_projector
            features=model.get_2dPool(projector(vision.to(next(projector.parameters()).dtype)))
            torch.cuda.synchronize();encoder_seconds=time.monotonic()-start
            torch.save(dict(features=features.cpu(),newline=model.model.image_newline.detach().cpu(),sample=s,questions=questions),cache/f'{i:03d}.pt')
            for method in a.methods:
                torch.cuda.synchronize();start=time.monotonic()
                visual,meta=select_visual(features,model.model.image_newline,method,0.25)
                torch.cuda.synchronize();selection_seconds=time.monotonic()-start
                for kind,pair in questions:
                    torch.cuda.synchronize();start=time.monotonic()
                    prediction,counts=generate_from_visual(model,tokenizer,pair['Q'],visual,max_tokens=512)
                    torch.cuda.synchronize()
                    row=dict(video_id=s['video_id'],video_path=s['video_path'],source=s['source'],video_class=s['video_class'],parent=s['inferred_parent'],split='dev',qa_type=kind,
                             question=pair['Q'],ground_truth=pair['A'],prediction=prediction,success=True,
                             encoder_seconds=encoder_seconds,selection_seconds=selection_seconds,generation_seconds=time.monotonic()-start,**counts,**meta)
                    with (a.output/'results.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    count+=1
                print(f'DONE video={i+1}/{len(samples)} method={method} rows={count}',flush=True)
    (a.output/'COMPLETE').write_text(str(count))

if __name__=='__main__':main()
