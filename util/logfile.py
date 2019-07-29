import logging
import logging.handlers

#logger instance, log level
logger = logging.getLogger("mylogger")
logger.setLevel(logging.DEBUG)

#formmater
# formatter = logging.Farmatter('[%(Levelname)s%(filename)s:%(lineno)s)]%(asctime)s>%(message)s')
formatter = logging.Formatter('[%(filename)s:%(lineno)s]%(asctime)s>%(message)s')

#filehander, StreamHander
fileHandler = logging.FileHandler('./log/my.log', 'a', 'utf-8')
fileMaxByte = 1024 * 1024 * 100 #100MB
fileHandler = logging.handlers.RotatingFileHandler('./log/my.log', maxBytes=fileMaxByte, backupCount=10, encoding='utf-8')
streamHandler = logging.StreamHandler()

#formmater setting in handler
fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)

#add Handler at logging
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)