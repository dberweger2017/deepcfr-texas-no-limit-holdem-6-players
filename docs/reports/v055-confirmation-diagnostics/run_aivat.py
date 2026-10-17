"""Spawn-safe entry point for the unchanged #316 per-hand wrapper (including macOS)."""
import runpy

if __name__ == '__main__':
    runpy.run_path('docs/reports/slumbot-aivat-real-runs/dump_aivat.py', run_name='aivat_dump_wrapper')
