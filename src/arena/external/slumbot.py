"""Official Slumbot wire codec: 200BB, street-local bet-to, public-only parsing.

This boundary deliberately cannot admit the repository's fixed HU100 policy.
Terminal winnings are evaluator data; terminal responses never become policy inputs.
"""

from dataclasses import asdict, dataclass
import re

from src.arena.external.interface import GameContract, JsonTransport
from src.blueprint.action_translation import Betting
from src.game.observation import ActionTaken, BlindPosted, BoardDealt, Decision, HandStarted, replay
from src.game.types import Action, ActionKind, Street

CONTRACT = GameContract(stack=20000)
TOKEN = re.compile(r'b[1-9][0-9]*|[kcf/]')
CARDS = {r + s for r in '23456789TJQKA' for s in 'cdhs'}
STREETS = (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)


@dataclass(frozen=True)
class PublicResponse:
    action: str
    client_pos: int
    hole_cards: tuple[str, str]
    board: tuple[str, ...]


def public_response(response: dict) -> PublicResponse:
    # Allowlist only: tokens, opponent cards, seeds, payoffs and arbitrary
    # evaluator/service additions can never reach policy observations.
    if not isinstance(response, dict) or ('error_msg' in response or 'error' in response) or 'winnings' in response:
        raise ValueError('Error or terminal response cannot be supplied to a policy')
    action = response.get('action'); seat = response.get('client_pos')
    own = response.get('hole_cards'); board = response.get('board')
    if (not isinstance(action, str) or len(action) > 4096
            or type(seat) is not int or seat not in (0, 1)
            or not isinstance(own, (list, tuple)) or len(own) != 2
            or not isinstance(board, (list, tuple)) or len(board) not in (0, 3, 4, 5)):
        raise ValueError('Invalid Slumbot public response')
    cards = [*own, *board]
    if any(not isinstance(c, str) or c not in CARDS for c in cards) or len(set(cards)) != len(cards):
        raise ValueError('Invalid or duplicate public cards')
    parts = TOKEN.findall(action)
    if ''.join(parts) != action or action.count('/') > 3:
        raise ValueError('Invalid Slumbot action grammar')
    return PublicResponse(action, seat, tuple(own), tuple(board))


class SlumbotAdapter:
    contract = CONTRACT

    def observation(self, response: dict, hand_id: str):
        public = public_response(response)
        # Slumbot seat 1 is the button/SB; local physical seats are retained.
        start = HandStarted(hand_id, ('slumbot-seat-0', 'slumbot-seat-1'), (20000, 20000), 1, 50, 100, '0.01')
        events = [start, BlindPosted(1, 50), BlindPosted(0, 100)]
        state = Betting((20000, 20000), actor=1).blind(events[1]).blind(events[2])
        street = 0
        for token in TOKEN.findall(public.action):
            if token == '/':
                if street == 3 or state.actor is not None or any(state.folded) or not all(state.stacks):
                    raise ValueError('Invalid betting street transition or terminal runout')
                street += 1
                end = (0, 3, 4, 5)[street]; begin = (0, 0, 3, 4)[street]
                if len(public.board) < end:
                    raise ValueError('Missing public board')
                event = BoardDealt(STREETS[street], public.board[begin:end])
                state = state.board(event, 1, 100)
                if state is None: raise ValueError('Invalid public betting boundary')
                events.append(event)
            else:
                if state.actor is None:
                    raise ValueError('Action after round/hand closure')
                action = (Action(ActionKind.RAISE, int(token[1:])) if token.startswith('b') else
                          Action({'k': ActionKind.CHECK, 'c': ActionKind.CALL, 'f': ActionKind.FOLD}[token]))
                # The public sample disallows folds when checking is available.
                if action.kind == ActionKind.FOLD and state.legal().call_amount == 0:
                    raise ValueError('Free fold is outside the verified Slumbot protocol')
                seat = state.actor; previous_street = state.street
                events.append(Decision(seat, state.legal()))
                state, paid = state.apply(action)
                events.append(ActionTaken(seat, previous_street, action, paid))
        # Official ParseAction permits omission of the final street separator.
        # Infer only the next completed round when its public board is present.
        if (state.actor is None and not any(state.folded) and all(state.stacks)
                and street < 3 and len(public.board) == (0, 3, 4, 5)[street + 1]):
            street += 1
            end = (0, 3, 4, 5)[street]; begin = (0, 0, 3, 4)[street]
            event = BoardDealt(STREETS[street], public.board[begin:end])
            state = state.board(event, 1, 100)
            if state is None: raise ValueError('Invalid inferred public betting boundary')
            events.append(event)
        if (state.actor != public.client_pos or state.actor is None
                or len(public.board) != (0, 3, 4, 5)[street]):
            raise ValueError('Response is not a client decision with a matching public board')
        events.append(Decision(state.actor, state.legal()))
        return replay(tuple(events), public.client_pos, public.hole_cards)

    def encode_action(self, action, observation):
        if observation.finished or observation.actor != observation.seat:
            raise ValueError('Client is not acting')
        observation.legal_actions.validate(action)
        if action.kind == ActionKind.RAISE: return f'b{action.raise_to}'
        if action.kind == ActionKind.FOLD and observation.legal_actions.call_amount == 0:
            raise ValueError('Free fold is outside the verified Slumbot protocol')
        return {ActionKind.FOLD: 'f', ActionKind.CHECK: 'k', ActionKind.CALL: 'c'}[action.kind]


class SlumbotConnection:
    """Inject transport, keep rotating bearer tokens out of replay evidence."""
    def __init__(self, transport: JsonTransport, policy):
        CONTRACT.admit(policy)  # Before the first request, even in fixture transport.
        self.transport = transport
        self.token = None
        self.failed = False

    def request(self, endpoint, increment=None):
        if self.failed:
            raise ValueError('Connection stopped after an ambiguous or invalid response')
        if increment is not None and (not isinstance(increment, str) or not re.fullmatch(r'b[1-9][0-9]*|[kcf]', increment)):
            raise ValueError('Invalid Slumbot action increment')
        if endpoint not in ('new_hand', 'act') or (endpoint == 'act') != (increment is not None):
            raise ValueError('Invalid Slumbot endpoint/action request')
        if endpoint == 'act' and self.token is None:
            raise ValueError('An action requires a current token')
        body = {'token': self.token} if self.token else {}
        if increment is not None: body['incr'] = increment
        try:
            response = self.transport.post('/slumbot/api/' + endpoint, body)
            if not isinstance(response, dict) or 'error_msg' in response or 'error' in response:
                raise ValueError('Slumbot rejected the request')
            new_token = response.get('token')
            if new_token is not None:
                if not isinstance(new_token, str) or not new_token:
                    raise ValueError('Invalid token update')
                self.token = new_token
            if self.token is None:
                raise ValueError('Slumbot did not establish a token')
        except Exception:
            self.failed = True
            raise
        return response


def replay_record(response, hand_id, action=None):
    adapter = SlumbotAdapter()
    public = public_response(response)
    view = adapter.observation(asdict(public), hand_id)
    encoded = adapter.encode_action(action, view) if action is not None else None
    return {'protocol': 'slumbot-public-prefix-v1', 'contract': asdict(CONTRACT),
            'hand_id': hand_id, 'public': asdict(public), 'observation': asdict(view),
            'selected': encoded}


def verify_record(record):
    public = record['public']
    view = SlumbotAdapter().observation(public, record['hand_id'])
    selected = record['selected']
    if selected is None: action = None
    elif re.fullmatch(r'b[1-9][0-9]*', selected): action = Action(ActionKind.RAISE, int(selected[1:]))
    else: action = Action({'k': ActionKind.CHECK, 'c': ActionKind.CALL, 'f': ActionKind.FOLD}[selected])
    import json
    if json.loads(json.dumps(replay_record(public, record['hand_id'], action))) != json.loads(json.dumps(record)):
        raise ValueError('External replay record differs')
    return view
