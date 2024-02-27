import logging
import sys
from datetime import datetime, timezone

COLORS = {
    'DEBUG': '\033[94m',
    'INFO': '\033[92m',
    'WARNING': '\033[33m',
    'ERROR': '\033[1;31m',
    'CRITICAL': '\033[4;1;31m',
    'RESET': '\033[0;0;37m'
}


class ColoredFormatter(logging.Formatter):
    def format(self, record):
        colors = {
            'levelname': COLORS.get(record.levelname, COLORS['RESET']),
        }
        colored_attrs = {
            attr: f"{colors[attr]}{getattr(record, attr)}{COLORS['RESET']}"
            for attr in colors
        }

        formatted_record = super().format(record)
        for attr, colored_value in colored_attrs.items():
            formatted_record = formatted_record.replace(
                getattr(record, attr), colored_value
            )
        return formatted_record


logging.Formatter.formatTime = (
    lambda self, record, datefmt=None: datetime.fromtimestamp(
        record.created,
        timezone.utc
    ).isoformat()
)
console_format = ColoredFormatter(
    fmt=' | '.join(
        [
            '%(asctime)s',
            '%(levelname)5s',
            '\033[95mmessage\033[37m: %(message)s'
        ]
    )
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(console_format)

logger = logging.getLogger('main')

logger.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

if __name__ == '__main__':
    logger.debug(msg='Your message')
    logger.info(msg='Your message')
    logger.warning(msg='Your message')
    logger.error(msg='Your message')
    logger.critical(msg='Your message')
