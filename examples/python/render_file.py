import argparse
from spacemit_feathertalk import FeatherTalkService
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--audio',required=True);p.add_argument('--output',required=True);p.add_argument('--max-frames',type=int,default=0);p.add_argument('--worker')
a=p.parse_args()
with FeatherTalkService(a.config,a.worker) as engine:
    result=engine.render_file(a.audio,a.output,a.max_frames)
    print(f'{result.frames} frames saved to {result.output_dir} at 25fps')
