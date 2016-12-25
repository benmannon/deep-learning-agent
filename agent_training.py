from draw import Draw
import level


def main():
    lvl = level.square()
    draw = Draw(lvl.grid.shape)
    draw.init()
    draw.update(lvl)
    draw.run()

if __name__ == "__main__":
    main()
