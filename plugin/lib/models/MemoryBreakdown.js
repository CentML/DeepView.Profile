'use babel';

import path from 'path';
import MemoryEntryLabel from './MemoryEntryLabel';
import {processFileReference, toPercentage} from '../utils';

class MemoryEntry {
  constructor({name, sizeBytes, displayPct, filePath, lineNumber}) {
    this._name = name;
    this._sizeBytes = sizeBytes;
    this._displayPct = displayPct;
    this._filePath = filePath;
    this._lineNumber = lineNumber;
  }

  get name() {
    return this._name;
  }

  get sizeBytes() {
    return this._sizeBytes;
  }

  get displayPct() {
    return this._displayPct;
  }

  get filePath() {
    return this._filePath;
  }

  get lineNumber() {
    return this._lineNumber;
  }

  static fromWeightEntry(weightEntry, peakUsageBytes) {
    return MemoryEntry._fromHelper(
      weightEntry,
      weightEntry.getWeightName(),
      weightEntry.getSizeBytes() + weightEntry.getGradSizeBytes(),
      peakUsageBytes,
    );
  }

  static fromActivationEntry(activationEntry, peakUsageBytes) {
    return MemoryEntry._fromHelper(
      activationEntry,
      activationEntry.getOperationName(),
      activationEntry.getSizeBytes(),
      peakUsageBytes,
    );
  }

  static _fromHelper(entry, name, sizeBytes, peakUsageBytes) {
    return new MemoryEntry({
      name,
      sizeBytes,
      displayPct: toPercentage(sizeBytes, peakUsageBytes),
      ...processFileReference(entry.getContext()),
    });
  }
}

class MemoryBreakdown {
  constructor(entryMap) {
    this._entryMap = entryMap;
  }

  getEntriesByLabel(label) {
    return this._entryMap[label].entries;
  }

  getOverallDisplayPctByLabel(label) {
    return this._entryMap[label].displayPct;
  }

  getTotalSizeBytesByLabel(label) {
    return this._entryMap[label].totalSizeBytes;
  }

  static fromMemoryUsageResponse(memoryUsageResponse) {
    const peakUsageBytes = memoryUsageResponse.getPeakUsageBytes();

    const weightEntries = memoryUsageResponse.getWeightEntriesList()
      .map(entry => MemoryEntry.fromWeightEntry(entry, peakUsageBytes));
    const activationEntries = memoryUsageResponse.getActivationEntriesList()
      .map(entry => MemoryEntry.fromActivationEntry(entry, peakUsageBytes));

    const sumEntrySizes = (acc, entry) => entry.sizeBytes + acc;
    const totalWeightSizeBytes = weightEntries.reduce(sumEntrySizes, 0);
    const totalActivationSizeBytes = activationEntries.reduce(sumEntrySizes, 0);

    const untrackedBytes = peakUsageBytes - totalWeightSizeBytes - totalActivationSizeBytes;
    const untrackedDisplayPct = toPercentage(untrackedBytes, peakUsageBytes)

    return new MemoryBreakdown({
      [MemoryEntryLabel.Weights]: {
        entries: weightEntries,
        displayPct: toPercentage(totalWeightSizeBytes, peakUsageBytes),
        totalSizeBytes: totalWeightSizeBytes,
      },
      [MemoryEntryLabel.Activations]: {
        entries: activationEntries,
        displayPct: toPercentage(totalActivationSizeBytes, peakUsageBytes),
        totalSizeBytes: totalActivationSizeBytes,
      },
      [MemoryEntryLabel.Untracked]: {
        entries: [new MemoryEntry({
          name: 'Untracked',
          sizeBytes: untrackedBytes,
          displayPct: untrackedDisplayPct,
          filePath: null,
          lineNumber: null,
        })],
        displayPct: untrackedDisplayPct,
        totalSizeBytes: untrackedBytes,
      },
    });
  }
}

export default MemoryBreakdown;
