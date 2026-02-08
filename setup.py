#!/usr/bin/env python3
"""
Setup and quick start script for Credit Risk Explainer.
"""

import subprocess
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is 3.9 or higher."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        logger.error(f"Python 3.9+ required. Current version: {version.major}.{version.minor}")
        return False
    logger.info(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True


def create_directories():
    """Create necessary directories."""
    directories = [
        'models/saved',
        'data/processed',
        'logs',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created directory: {directory}")


def install_dependencies():
    """Install required Python packages."""
    logger.info("Installing dependencies from requirements.txt...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '--quiet'
        ])
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False


def run_training():
    """Run the model training pipeline."""
    logger.info("Running model training pipeline...")
    
    try:
        subprocess.check_call([
            sys.executable, 'scripts/train_models.py'
        ])
        logger.info("✅ Model training completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training failed: {e}")
        return False


def display_next_steps():
    """Display next steps to the user."""
    print("\n" + "="*70)
    print("🎉 SETUP COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("\n1. Launch the Streamlit UI:")
    print("   cd ui")
    print("   streamlit run app.py")
    print("\n2. Open your browser to: http://localhost:8501")
    print("\n3. Explore the features:")
    print("   - Submit credit applications")
    print("   - View model performance")
    print("   - Analyze global explanations")
    print("   - Check fairness metrics")
    print("\n4. Read the documentation:")
    print("   - README.md: Project overview")
    print("   - docs/TECHNICAL_SPEC.md: Technical details")
    print("   - docs/MODEL_CARD.md: Model documentation")
    print("="*70 + "\n")


def main():
    """Main setup flow."""
    print("\n" + "="*70)
    print("Credit Risk Explainer - Setup & Quick Start")
    print("="*70 + "\n")
    
    # Step 1: Check Python version
    logger.info("Step 1: Checking Python version...")
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Create directories
    logger.info("\nStep 2: Creating directories...")
    create_directories()
    
    # Step 3: Install dependencies
    logger.info("\nStep 3: Installing dependencies...")
    if not install_dependencies():
        logger.warning("⚠️ Some dependencies may not have installed correctly")
        logger.info("You can manually install them with: pip install -r requirements.txt")
    
    # Step 4: Run training
    logger.info("\nStep 4: Training models...")
    user_input = input("Would you like to train the models now? (y/n): ").strip().lower()
    
    if user_input == 'y':
        if not run_training():
            logger.warning("⚠️ Training encountered errors. Check logs for details.")
    else:
        logger.info("⏭️  Skipping model training. You can train later with:")
        logger.info("   python scripts/train_models.py")
    
    # Display next steps
    display_next_steps()


if __name__ == "__main__":
    main()
