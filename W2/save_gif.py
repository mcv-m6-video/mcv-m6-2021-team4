import sys
sys.path.append("W1")
from utils import save_gif

if __name__ == '__main__':
    save_gif(sys.argv[1], sys.argv[2])