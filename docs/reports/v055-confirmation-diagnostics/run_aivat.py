"""Spawn-safe entry point for #316's unchanged per-hand wrapper, with progress logging."""
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import runpy


class ProgressPool(ProcessPoolExecutor):
    def map(self, fn, *iterables, **kwargs):
        for count, result in enumerate(super().map(fn, *iterables, **kwargs), 1):
            if count % 1000 == 0:
                print(f'{fn.__name__}: {count} completed', flush=True)
            yield result


if __name__ == '__main__':
    # Keep this guarded module as __main__ when macOS workers are spawned.
    concurrent.futures.ProcessPoolExecutor = ProgressPool
    runpy.run_path('docs/reports/slumbot-aivat-real-runs/dump_aivat.py', run_name='aivat_dump_wrapper')
