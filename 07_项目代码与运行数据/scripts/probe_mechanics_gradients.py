"""Always run the train entry point in zero-update mechanics calibration mode."""
import sys
from train_diffusion import main

if __name__ == '__main__':
    if '--mechanics-calibrate' not in sys.argv:
        sys.argv.append('--mechanics-calibrate')
    main()
