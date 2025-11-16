import logging


logger = logging.getLogger(__name__)


def main():
    message = "Hello from on-the-go-coding-test!"
    logger.info(message)
    print(message)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
