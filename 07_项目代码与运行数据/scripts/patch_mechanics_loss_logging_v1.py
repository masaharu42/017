"""Add per-update mechanics loss CSV logging to future training runs."""
from pathlib import Path
from datetime import datetime
import ast
p=Path('train_diffusion.py')
s=p.read_text(encoding='utf-8')
old='''                global_step += 1
                if diagnostics is not None:
                    diagnostics.record'''
new='''                global_step += 1
                if mechanics:
                    append_csv_row(output_dir / "mechanics_loss_per_step.csv", {
                        "global_step": global_step, **mechanics_logs,
                    })
                if diagnostics is not None:
                    diagnostics.record'''
if new in s:
    print('Per-step mechanics logging already installed.')
elif s.count(old)==1:
    updated=s.replace(old,new,1); ast.parse(updated)
    backup=p.with_name('train_diffusion.before_mechanics_loss_log_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.py')
    backup.write_text(s,encoding='utf-8');p.write_text(updated,encoding='utf-8')
    print('Installed; backup:',backup)
else:
    raise SystemExit('Training file differs from expected; no changes made.')
