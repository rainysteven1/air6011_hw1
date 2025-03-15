from typing import Any, Dict
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
import os
import logging
import sys


class _ColorCodes:
    DEBUG = "\033[36m"
    INFO = "\033[32m"
    WARNING = "\033[33m"
    ERROR = "\033[31m"
    RESET = "\033[0m"


class _ColoredFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: str = None):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        ori_message = super().format(record)

        color = getattr(_ColorCodes, record.levelname, _ColorCodes.RESET)
        levelname = record.levelname
        colored_level = f"{color}{levelname}{_ColorCodes.RESET}"

        return ori_message.replace(levelname, colored_level)


class CustomLogger:
    _instance = None

    LOG_FILE = "run.log"

    def __new__(
        cls, name: str, log_dir: str, log_level: str, enable_color: bool = True
    ):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.__init_logger(name, log_dir, log_level, enable_color)
        return cls._instance

    def __get_console_formatter(self) -> logging.Formatter:
        base_fmt = "[%(asctime)s] [%(levelname)s] [%(module)s] - %(message)s"
        if self.enable_color:
            return _ColoredFormatter(base_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        else:
            return logging.Formatter(base_fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def __ensure_log_dir(self):
        os.makedirs(self.log_dir, exist_ok=True)
        if not os.access(self.log_dir, os.W_OK):
            os.chmod(self.log_dir, 0o755)

    def __init_logger(
        self, name: str, log_dir: str, log_level: str, enable_color: bool
    ):
        self.name = name
        self.log_dir = log_dir
        self.enable_color = enable_color

        self.__ensure_log_dir()

        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level.upper())
        self.logger.propagate = False

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        self.__configure_handlers()

        self.logger = logging.LoggerAdapter(self.logger, extra={})

    def __configure_handlers(self):
        file_formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"levelname": "severity", "asctime": "timestamp"},
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            static_fields={"runtime": "python"},
        )
        file_handler = RotatingFileHandler(
            filename=os.path.join(self.log_dir, self.LOG_FILE), encoding="utf-8"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.__get_console_formatter())
        self.logger.addHandler(console_handler)

    def __log(
        self,
        level: str,
        message: str,
        extra: Dict[str, Any] = None,
        exc_info: bool = False,
    ) -> None:
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        extra = extra or {}
        extra.update({"timestamp": datetime.now().isoformat()})
        log_method(message, extra=extra, exc_info=exc_info)

    def info(self, message: str, **kwargs) -> None:
        self.__log("INFO", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self.__log("ERROR", message, **kwargs)


timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_file = os.path.join(os.getcwd(), "result", timestamp)
logger = CustomLogger("AIR6011_HW1", log_file, "INFO")
