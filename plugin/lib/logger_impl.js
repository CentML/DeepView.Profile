'use babel';

const LogLevel = {
  TRACE: 0,
  DEBUG: 1,
  INFO: 2,
  WARN: 3,
  ERROR: 4,
};

class Logger {
  constructor(logLevel) {
    this._logLevel = logLevel;
  }

  error(...args) {
    if (this._logLevel > LogLevel.ERROR) {
      return;
    }
    this._log(console.error, ...args);
  }

  warn(...args) {
    if (this._logLevel > LogLevel.WARN) {
      return;
    }
    this._log(console.warn, ...args);
  }

  info(...args) {
    if (this._logLevel > LogLevel.INFO) {
      return;
    }
    this._log(console.info, ...args);
  }

  debug(...args) {
    if (this._logLevel > LogLevel.DEBUG) {
      return;
    }
    this._log(console.debug, ...args);
  }

  trace(...args) {
    if (this._logLevel > LogLevel.TRACE) {
      return;
    }
    this._log(console.trace, ...args);
  }

  _log(loggingFn, ...args) {
    loggingFn('[Skyline]', ...args);
  }
}

export {Logger, LogLevel};
