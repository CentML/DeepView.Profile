'use babel';

import {Logger, LogLevel} from './logger_impl';
import env from './env.json';

let logger;

if (env.development) {
  logger = new Logger(LogLevel.DEBUG);
} else {
  logger = new Logger(LogLevel.WARN);
}

export default logger;
