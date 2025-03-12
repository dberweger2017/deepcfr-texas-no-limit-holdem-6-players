# DeepCFR Poker GUI

A graphical interface for playing against trained DeepCFR poker models.

## Features

- Play against multiple AI opponents with a visual interface
- Support for both standard DeepCFR models and models with opponent modeling
- Visual representation of cards, player chips, and game state
- Game history log
- Customizable game settings (blinds, stake, player positions)

## Installation

Ensure you have PyQt5 installed:

```bash
pip install PyQt5
```

## Usage

### Basic Usage

Run the application from your project root directory:

```bash
python -m scripts.poker_gui
```

This will open the model selection dialog, allowing you to choose which models to play against.

### Command Line Options

You can also launch the application with specific models or settings using command line arguments:

#### Load specific model files

```bash
python -m scripts.poker_gui --models path/to/model1.pt path/to/model2.pt path/to/model3.pt
```

#### Load models from a folder

```bash
python -m scripts.poker_gui --models_folder models_mixed_om_v2
```

#### Set your position and game parameters

```bash
python -m scripts.poker_gui --models_folder models_om --position 2 --stake 100 --sb 0.5 --bb 1
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--models` | List of specific model paths to use as opponents | None |
| `--models_folder` | Directory containing model checkpoint files | None |
| `--position` | Your position at the table (0-5) | 0 |
| `--stake` | Initial chip stack for all players | 200.0 |
| `--sb` | Small blind amount | 1.0 |
| `--bb` | Big blind amount | 2.0 |

## Gameplay

1. At the start of each hand, cards will be dealt and blinds posted
2. When it's your turn, the action buttons will be enabled
3. Choose from available actions (fold, check/call, raise)
4. For raises, you can set a custom amount or use preset options (half pot, full pot)
5. AI players will take their turns automatically
6. At the end of each hand, results will be displayed
7. Click "New Hand" to start another hand

## Model Types

The application automatically detects whether a model uses opponent modeling based on its filename:
- Models with "om" in the filename are loaded as DeepCFRAgentWithOpponentModeling
- Other models are loaded as standard DeepCFRAgent

## Troubleshooting

- If you encounter errors loading models, check that the paths are correct
- Make sure you're running the script from the project root directory
- If no models are found, the application will use random agents as opponents