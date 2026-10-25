"""Validate and summarize all six completed H2-mini no-replay runs."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from continualguard.h2_mini_summary import summarize
if __name__=='__main__':
 result=summarize(ROOT)
 print(json.dumps({'status':result['aggregate']['status'],'checkpoints':result['validation']['checkpoints'],'readiness':result['readiness']['classification'],'total_stage_runtime_seconds':result['aggregate']['total_stage_runtime_seconds']},indent=2))
