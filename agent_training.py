import draw
import level


def main():
    draw.init()
    draw.update(level.square())
    draw.run()

if __name__ == "__main__":
    main()
