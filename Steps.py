# Performing actions which are not relevant to be taught to the AI, such as difficulty selection and screen transitions.

from MAMEToolkit.sf_environment.Actions import Actions

def set_difficulty(frame_ratio, difficulty):
    steps = [
        {'wait': 0, 'actions': [Actions.SERVICE]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_JPUNCH]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_JPUNCH]},
        ]
    if difficulty % 8 < 3:
        steps += [{'wait': int(10 / frame_ratio),
                  'actions': [Actions.P1_LEFT]} for i in range(3 - difficulty % 8)]
    else:
        steps += [{'wait': int(10 / frame_ratio),
                  'actions': [Actions.P1_RIGHT]} for i in range(difficulty % 8 - 3)]
    steps += [
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_JPUNCH]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_JPUNCH]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_JPUNCH]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_DOWN]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.P1_JPUNCH]},
        ]
    return steps

def start_game(frame_ratio):
    return [
        {'wait': int(300 / frame_ratio), 'actions': [Actions.COIN_P1]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.COIN_P1]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_START]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_UP]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_UP]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_UP]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_UP]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_UP]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_UP, Actions.P1_JPUNCH]},
        {'wait': int(80 / frame_ratio), 'actions': [Actions.P1_JPUNCH]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_JPUNCH]},
        ]

def next_stage(frame_ratio):
    return [{'wait': int(60 / frame_ratio),
            'actions': [Actions.P1_JPUNCH]}] + [{'wait': 0,
            'actions': [Actions.P1_JPUNCH]} for _ in range(int(180
            / frame_ratio))] + [{'wait': int(60 / frame_ratio),
                                'actions': [Actions.P1_JPUNCH]}]

def new_game(frame_ratio):
    return [
        {'wait': 0, 'actions': [Actions.SERVICE]},
        {'wait': int(30 / frame_ratio), 'actions': [Actions.P1_UP]},
        {'wait': int(30 / frame_ratio), 'actions': [Actions.P1_JPUNCH]},
        {'wait': int(300 / frame_ratio), 'actions': [Actions.COIN_P1]},
        {'wait': int(10 / frame_ratio), 'actions': [Actions.COIN_P1]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_START]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_UP]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_UP]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_UP]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_UP]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_UP]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_UP, Actions.P1_JPUNCH]},
        {'wait': int(80 / frame_ratio), 'actions': [Actions.P1_JPUNCH]},
        {'wait': int(60 / frame_ratio), 'actions': [Actions.P1_JPUNCH]},
        ]

