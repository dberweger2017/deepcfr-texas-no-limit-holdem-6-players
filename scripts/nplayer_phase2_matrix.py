"""The fixed 42 contrasts; roots and counts do not depend on observed outcomes."""
from scripts.run_nplayer_blueprint_seed import SEEDS

BLOCKS = 8192
ROOT_BASE = 202610150801
BOTS = ('random', 'check-call', 'tight-aggressive', 'pot-pressure')


def matrix():
    cases = []
    def add(name, singleton, pair, primary=False, target_role='singleton'):
        cases.append({'name':name,'singleton':singleton,'pair':pair,'primary':primary,
                      'target_role':target_role,'blocks':BLOCKS,'root':ROOT_BASE+len(cases)})
    for seed in SEEDS:
        final = f'{seed}/1000m'
        for prior in ('200m','500m'):
            early = f'{seed}/{prior}'
            add(f'learning-{seed}-{prior}',final,early,prior=='200m')
            add(f'learning-{seed}-{prior}-reverse',early,final,target_role='pair')
    for seed in SEEDS:
        final = f'{seed}/1000m'
        for bot in BOTS:
            add(f'robustness-{seed}-{bot}',final,'bot='+bot)
            add(f'robustness-{seed}-{bot}-reverse','bot='+bot,final,target_role='pair')
    for seed in SEEDS:
        for other in SEEDS:
            if seed != other:
                add(f'consistency-{seed}-{other}',f'{seed}/1000m',f'{other}/1000m')
    return cases
