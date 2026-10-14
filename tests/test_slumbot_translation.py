"""Live client options and journal analysis keep translation identity explicit."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.play_slumbot import load_player
from src.blueprint.translation_cli import add_translation_options, translation_options
from src.blueprint.action_translation import TranslationOptions
from src.blueprint.abstraction import HU200_EQUITY_SCHEMA
from src.blueprint import equity_buckets
from src.blueprint.equity_buckets import EquityCards
from src.arena.external.analysis import analyze
from src.arena.external.journal import PrefixJournal
from src.arena.external.slumbot import SlumbotAdapter, replay_record, terminal_record
from tests.native_equity_tables import write_tables
from tests.test_hu200_evaluation import fixture_spec
from tests.test_external_slumbot_session import LIVE


def test_translation_cli_bounds_are_explicit():
    p=argparse.ArgumentParser();add_translation_options(p)
    assert translation_options(p.parse_args([])) is None
    assert translation_options(p.parse_args(['--action-translation']))==TranslationOptions()
    with pytest.raises(ValueError):translation_options(p.parse_args(['--translation-max-states','1']))
    with pytest.raises(ValueError):translation_options(p.parse_args(['--action-translation','--translation-max-events','0']))


def test_load_player_and_analysis_translation_identity(tmp_path,monkeypatch):
    tables=tmp_path/'tables';tables.mkdir()
    scenes=[(r['hole_cards'],r['board']) for r in LIVE[0] if len(r['board'])>=3]
    write_tables(tables,scenes);cards=EquityCards(tables,pinned=False);cards.directory=tables
    monkeypatch.setattr(equity_buckets,'EQUITY_K50_SHA256',cards.sha256)
    spec=fixture_spec(tmp_path/'model',HU200_EQUITY_SCHEMA,cards=cards)
    args=SimpleNamespace(model=Path(spec['path']),model_sha256=spec['sha256'],card_buckets=tables,
        policy_cache=tmp_path/'cache',missing_key_fallback='uniform',cache_read_only=False,
        seed=381,action_translation=True,translation_max_states=512,translation_max_events=128)
    model,player,identity=load_player(args)
    assert identity['action_translation']==model.description['action_translation']
    assert identity['action_translation']['max_states']==512
    journal=PrefixJournal(tmp_path/'journal.jsonl',identity)
    adapter=SlumbotAdapter()
    for response in LIVE[0][:-1]:
        view=adapter.observation(response,'h0');action=player.choose_action(view)
        record=replay_record(response,'h0',action)
        record['policy_telemetry']=dict(player.last_decision[3],street=view.street.value)
        journal.append(record)
    journal.terminal(terminal_record(LIVE[0][-1],'h0'))
    journal.close()
    result=analyze(tmp_path/'journal.jsonl',model=model,workers=1)
    assert result['action_translation']==identity['action_translation']
    assert sum(sum(v.values()) for v in result['translation_by_street'].values())==len(LIVE[0])-1
    assert result['lookup_seconds']['max']>=0
    model.configure_translation(None)
    with pytest.raises(ValueError,match='translation differs'):
        analyze(tmp_path/'journal.jsonl',model=model,workers=1)
