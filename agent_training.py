from draw import Draw
import level


def main():
    draw = Draw()
    draw.init()
    draw.update(level.square())
    draw.run()

if __name__ == "__main__":
    main()
