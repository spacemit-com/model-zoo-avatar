#!/usr/bin/env python3
"""Require observed EP execution and no CPU Resize events in an ORT JSON profile."""
import argparse,json
from pathlib import Path

def inspect(path):
    events=json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(events,list):raise ValueError('Expected ORT event list')
    cpu=sum(1 for e in events if e.get('args',{}).get('provider')=='CPUExecutionProvider' and e.get('args',{}).get('op_name')=='Resize')
    ep=sum(1 for e in events if e.get('args',{}).get('provider')=='SpaceMITExecutionProvider')
    return {'ok':cpu==0 and ep>0,'cpu_resize_events':cpu,'ep_events':ep}

def main():
    p=argparse.ArgumentParser();p.add_argument('profile',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();result=inspect(a.profile)
    if a.output:a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result));raise SystemExit(0 if result['ok'] else 2)
if __name__=='__main__':main()
