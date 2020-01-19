'use babel';

import RunTimeEntryLabel from './RunTimeEntryLabel';
import {processFileReference, toPercentage} from '../utils';
import Logger from '../logger';

class RunTimeEntry {
  constructor({name, runTimeMs, filePath, lineNumber}) {
    this._name = name;
    this._runTimeMs = runTimeMs;
    this._filePath = filePath;
    this._lineNumber = lineNumber;
  }

  get name() {
    return this._name;
  }

  get runTimeMs() {
    return this._runTimeMs;
  }

  get filePath() {
    return this._filePath;
  }

  get lineNumber() {
    return this._lineNumber;
  }
}

class RunTimeBreakdown {
  constructor(iterationRunTimeMs, entryMap) {
    this._iterationRunTimeMs = iterationRunTimeMs;
    this._entryMap = entryMap;
  }

  getIterationRunTimeMs() {
    return this._iterationRunTimeMs;
  }

  getEntriesByLabel(label) {
    return this._entryMap[label].entries;
  }

  getTotalTimeMsByLabel(label) {
    return this._entryMap[label].totalTimeMs;
  }

  static fromRunTimeResponse(runTimeResponse) {
    const iterationRunTimeMs = runTimeResponse.getIterationRunTimeMs();

    const forwardEntries = [];
    const backwardEntries = [];

    for (const entry of runTimeResponse.getRunTimeEntriesList()) {
      forwardEntries.push(new RunTimeEntry({
        name: entry.getOperationName(),
        runTimeMs: entry.getForwardMs(),
        ...processFileReference(entry.getContext()),
      }));

      if (isNaN(entry.getBackwardMs())) {
        continue;
      }

      backwardEntries.push(new RunTimeEntry({
        name: entry.getOperationName(),
        runTimeMs: entry.getBackwardMs(),
        ...processFileReference(entry.getContext()),
      }));
    }

    const sumEntryTimes = (acc, entry) => entry.runTimeMs + acc;
    const totalForwardMs = forwardEntries.reduce(sumEntryTimes, 0);
    const totalBackwardMs = backwardEntries.reduce(sumEntryTimes, 0);

    const entryMap = {
      [RunTimeEntryLabel.Forward]: {
        entries: forwardEntries,
        totalTimeMs: totalForwardMs,
      },
      [RunTimeEntryLabel.Backward]: {
        entries: backwardEntries,
        totalTimeMs: totalBackwardMs,
      },
    };

    const remainingTimeMs = iterationRunTimeMs - totalForwardMs - totalBackwardMs;
    let totalIterationMs = iterationRunTimeMs;
    if (remainingTimeMs > 0) {
      entryMap[RunTimeEntryLabel.Other] = {
        entries: [new RunTimeEntry({
          name: 'Other',
          runTimeMs: remainingTimeMs,
          filePath: null,
          lineNumber: null,
        })],
        totalTimeMs: remainingTimeMs,
      };
    } else {
      Logger.warn('The total measured iteration run time is less than the sum of the run time entries.');
      totalIterationMs = totalForwardMs + totalBackwardMs;
    }

    return new RunTimeBreakdown(totalIterationMs, entryMap);
  }
}

export default RunTimeBreakdown;
