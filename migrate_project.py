#!/usr/bin/env python3
import os
import shutil
import fileinput

def create_directory_structure():
    """Create the project directory structure."""
    directories = [
        'src/core',
        'src/opponent_modeling',
        'src/training',
        'scripts',
        'models/base',
        'models/opponent_modeling',
        'logs/base',
        'logs/opponent_modeling',
        'tests',
        'notebooks',
        'configs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create an empty __init__.py in each directory
        open(os.path.join(directory, '__init__.py'), 'a').close()

def update_imports(filename):
    """Update import statements in a file."""
    import_replacements = [
        # Model and encoding imports
        ('import model', 'from src.core import model'),
        ('from model import', 'from src.core.model import'),
        
        # Deep CFR imports
        ('import deep_cfr', 'from src.core import deep_cfr'),
        ('from deep_cfr import', 'from src.core.deep_cfr import'),
        
        # Opponent modeling imports
        ('import opponent_model', 'from src.opponent_modeling import opponent_model'),
        ('from opponent_model import', 'from src.opponent_modeling.opponent_model import'),
        
        # Opponent modeling agent imports
        ('import deep_cfr_with_opponent_modeling', 'from src.opponent_modeling import deep_cfr_with_opponent_modeling'),
        ('from deep_cfr_with_opponent_modeling import', 'from src.opponent_modeling.deep_cfr_with_opponent_modeling import'),
    ]
    
    # Read the file
    with open(filename, 'r') as f:
        content = f.read()
    
    # Apply replacements
    for old, new in import_replacements:
        content = content.replace(old, new)
    
    # Write back to the file
    with open(filename, 'w') as f:
        f.write(content)

def migrate_files():
    """Migrate files to new directory structure."""
    file_mapping = {
        # Core files
        'model.py': 'src/core/model.py',
        'deep_cfr.py': 'src/core/deep_cfr.py',
        
        # Opponent Modeling
        'opponent_model.py': 'src/opponent_modeling/opponent_model.py',
        'deep_cfr_with_opponent_modeling.py': 'src/opponent_modeling/deep_cfr_with_opponent_modeling.py',
        
        # Training Scripts
        'train.py': 'src/training/train.py',
        'train_with_opponent_modeling.py': 'src/training/train_with_opponent_modeling.py',
        'train_mixed_with_opponent_modeling.py': 'src/training/train_mixed_with_opponent_modeling.py',
        
        # Utility Scripts
        'play.py': 'scripts/play.py',
        'visualize_tournament.py': 'scripts/visualize_tournament.py',
        'telegram_notifier.py': 'scripts/telegram_notifier.py'
    }
    
    # Move files
    for source, destination in file_mapping.items():
        if os.path.exists(source):
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            
            # Move the file
            shutil.move(source, destination)
            print(f"Moved {source} to {destination}")
            
            # Update imports in the moved file
            update_imports(destination)
        else:
            print(f"Warning: {source} not found")

def create_package_init_files():
    """Create __init__.py files for each package."""
    init_contents = {
        'src/__init__.py': '''"""
DeepCFR Poker AI Project
"""\n''',
        
        'src/core/__init__.py': '''"""
Core models and utilities for Deep CFR poker AI
"""\n''',
        
        'src/opponent_modeling/__init__.py': '''"""
Opponent modeling components for Deep CFR poker AI
"""\n''',
        
        'src/training/__init__.py': '''"""
Training scripts and utilities for Deep CFR poker AI
"""\n'''
    }
    
    for filepath, content in init_contents.items():
        with open(filepath, 'w') as f:
            f.write(content)

def main():
    """Execute the migration process."""
    print("Starting project migration...")
    
    # Create directory structure
    create_directory_structure()
    
    # Migrate files
    migrate_files()
    
    # Create package __init__ files
    create_package_init_files()
    
    print("\nMigration complete!")
    print("Please review the changes and test your scripts.")

if __name__ == '__main__':
    main()