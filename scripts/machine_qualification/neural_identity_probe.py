"""Trace identical neural inputs and any row-position floating-point differences.

Does not change the tested models, assertions or tolerances. Reports the first
Linear operation where byte-identical input rows acquire unequal output rows.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(os.environ['QUALIFICATION_ROOT'])
sys.path.insert(0, str(root / 'repo-main'))
import torch
from src.holdem.actions import bet_candidates
from src.holdem.multistreet_models import VARIANTS, make_model
from src.holdem.multistreet_reference import build_context, range_support

torch.set_num_threads(int(os.environ.get('QUALIFICATION_TORCH_THREADS', '2')))
support = range_support(json.loads((root / 'repo-main/configs/holdem/representation.json').read_text())['range_templates'])
rows = []
for street, board in [('flop', ('Ac', 'Kd', '7h')), ('turn', ('Ac', 'Kd', '7h', '4s'))]:
    contexts = [build_context(name='same-public-prefix', split='train', street=street,
        board=board, holding=('Qs', 'Jc'), support=support, samples=2,
        deals_per_sample=1, seed=seed) for seed in (13, 29)]
    observations = [world.observe(context.hero_seat) for context in contexts for world in context.worlds]
    candidates = [bet_candidates(view) for view in observations]
    assert all(v == observations[0] for v in observations)
    assert all(v == candidates[0] for v in candidates)
    for variant in VARIANTS:
        model = make_model(variant, seed=31)
        divergences = []
        def hook(name):
            def inspect(module, inputs, outputs):
                data = inputs[0].detach()
                values = outputs.detach()
                if data.ndim != 2 or values.ndim != 2:
                    return
                for i in range(len(data)):
                    for j in range(i + 1, len(data)):
                        if torch.equal(data[i], data[j]) and not torch.equal(values[i], values[j]):
                            divergences.append({'layer': name, 'shape': list(data.shape),
                                'rows': [i, j], 'input_equal': True,
                                'input_sha256': hashlib.sha256(data[i].numpy().tobytes()).hexdigest(),
                                'max_abs_output_delta': float((values[i] - values[j]).abs().max()),
                                'max_float32_ulp_delta': int((values[i].view(torch.int32).to(torch.int64) - values[j].view(torch.int32).to(torch.int64)).abs().max())})
                            return
            return inspect
        handles = [module.register_forward_hook(hook(name)) for name, module in model.named_modules()
                   if isinstance(module, torch.nn.Linear)]
        output = model(candidates)
        for handle in handles:
            handle.remove()
        rows.append({'street': street, 'variant': variant, 'candidates_equal': True,
            'output_identical': all(torch.equal(output[0].regrets, v.regrets) and
                                    torch.equal(output[0].values, v.values) for v in output),
            'first_identical_input_divergence': divergences[0] if divergences else None})
print(json.dumps({'torch': torch.__version__, 'threads': torch.get_num_threads(),
    'capability': torch.backends.cpu.get_cpu_capability(),
    'MKL_CBWR': os.environ.get('MKL_CBWR'), 'mkldnn': torch.backends.mkldnn.enabled,
    'cases': rows}, indent=2), flush=True)
