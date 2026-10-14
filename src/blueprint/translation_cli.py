"""Explicit translation CLI options; an omitted flag preserves legacy identities."""
from src.blueprint.action_translation import TranslationOptions


def add_translation_options(parser):
    parser.add_argument('--action-translation', action='store_true')
    parser.add_argument('--translation-max-states', type=int, default=512)
    parser.add_argument('--translation-max-events', type=int, default=128)


def translation_options(args):
    if not getattr(args, 'action_translation', False):
        if (getattr(args, 'translation_max_states', 512), getattr(args, 'translation_max_events', 128)) != (512, 128):
            raise ValueError('Translation bounds require --action-translation')
        return None
    return TranslationOptions(args.translation_max_states, args.translation_max_events)
