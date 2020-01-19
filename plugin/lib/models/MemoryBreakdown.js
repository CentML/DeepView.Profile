'use babel';

import MemoryEntryLabel from './MemoryEntryLabel';
import {processFileReference, toPercentage} from '../utils';

class MemoryEntry {
  constructor({name, sizeBytes, filePath, lineNumber}) {
    this._name = name;
    this._sizeBytes = sizeBytes;
    this._filePath = filePath;
    this._lineNumber = lineNumber;
  }

  get name() {
    return this._name;
  }

  get sizeBytes() {
    return this._sizeBytes;
  }

  get filePath() {
    return this._filePath;
  }

  get lineNumber() {
    return this._lineNumber;
  }

  static fromWeightEntry(weightEntry) {
    return MemoryEntry._fromHelper(
      weightEntry,
      weightEntry.getWeightName(),
      weightEntry.getSizeBytes() + weightEntry.getGradSizeBytes(),
    );
  }

  static fromActivationEntry(activationEntry) {
    return MemoryEntry._fromHelper(
      activationEntry,
      activationEntry.getOperationName(),
      activationEntry.getSizeBytes(),
    );
  }

  static _fromHelper(entry, name, sizeBytes) {
    return new MemoryEntry({
      name,
      sizeBytes,
      ...processFileReference(entry.getContext()),
    });
  }
}

class MemoryBreakdown {
  constructor(peakUsageBytes, entryMap) {
    this._peakUsageBytes = peakUsageBytes;
    this._entryMap = entryMap;
  }

  getPeakUsageBytes() {
    return this._peakUsageBytes;
  }

  getEntriesByLabel(label) {
    return this._entryMap[label].entries;
  }

  getTotalSizeBytesByLabel(label) {
    return this._entryMap[label].totalSizeBytes;
  }

  static fromMemoryUsageResponse(memoryUsageResponse) {
    const peakUsageBytes = memoryUsageResponse.getPeakUsageBytes();

    const weightEntries = memoryUsageResponse.getWeightEntriesList()
      .map(MemoryEntry.fromWeightEntry);
    const activationEntries = memoryUsageResponse.getActivationEntriesList()
      .map(MemoryEntry.fromActivationEntry);

    const sumEntrySizes = (acc, entry) => entry.sizeBytes + acc;
    const totalWeightSizeBytes = weightEntries.reduce(sumEntrySizes, 0);
    const totalActivationSizeBytes = activationEntries.reduce(sumEntrySizes, 0);

    const untrackedBytes = peakUsageBytes - totalWeightSizeBytes - totalActivationSizeBytes;

    return new MemoryBreakdown(peakUsageBytes, {
      [MemoryEntryLabel.Weights]: {
        entries: weightEntries,
        totalSizeBytes: totalWeightSizeBytes,
      },
      [MemoryEntryLabel.Activations]: {
        entries: activationEntries,
        totalSizeBytes: totalActivationSizeBytes,
      },
      [MemoryEntryLabel.Untracked]: {
        entries: [new MemoryEntry({
          name: 'Untracked',
          sizeBytes: untrackedBytes,
          filePath: null,
          lineNumber: null,
        })],
        totalSizeBytes: untrackedBytes,
      },
    });
  }
}

export default MemoryBreakdown;
