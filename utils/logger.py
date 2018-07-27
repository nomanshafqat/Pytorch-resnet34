import logging
import os

class Logger:
    """ Class for logging various debug and info messages
    Attributes:
        logger: Object of default python's logging class. Used to log messages
    args:
        :param directory: Logs will be stored as directory/log/log.txt
        :param name: Name of the project (This can be used to get the same logger anywhere without using this class
        :param debug: Show debug messasges in console if set true
    """

    def __init__(self, directory, name, debug=True):

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Create directory to store logs
        assert (os.path.exists(directory))
        if not (os.path.exists(os.path.join(directory, "log"))):
            os.mkdir(os.path.join(directory, "log"))
        self.directory = os.path.join(directory, "log")

        # Text Logger
        file_handler = logging.FileHandler(os.path.join(self.directory, "logs.txt"))
        file_handler.setLevel(logging.DEBUG)

        # Console Logger
        stream_handler = logging.StreamHandler()
        if debug:
            stream_handler.setLevel(logging.DEBUG)
        else:
            stream_handler.setLevel(logging.INFO)

        # Format logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

        self.logger = logger

    def get_logger(self):
        """Returns the default logger object of python
        """
        return self.logger

