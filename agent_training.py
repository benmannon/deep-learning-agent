import level
from draw import Draw


def main():
    lvl = level.square()
    draw = Draw(lvl.grid.shape)
    draw.update(lvl)
    draw.run()

if __name__ == "__main__":
    main()
