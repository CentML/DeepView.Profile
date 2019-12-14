'use babel';

export default class MemoryUsage {
  constructor(peakUsageBytes, memoryCapacityBytes) {
    this._peakUsageBytes = peakUsageBytes;
    this._memoryCapacityBytes = memoryCapacityBytes;
  }

  get peakUsageMb() {
    return this._megabytesFromBytes(this._peakUsageBytes);
  }

  get memoryCapacityMb() {
    return this._megabytesFromBytes(this._memoryCapacityBytes);
  }

  get displayPct() {
    return this._peakUsageBytes / this._memoryCapacityBytes * 100;
  }

  _megabytesFromBytes(bytes) {
    return bytes / 1024 / 1024;
  }

  static fromMemoryUsageResponse(memoryUsageResponse) {
    return new MemoryUsage(
      memoryUsageResponse.getPeakUsageBytes(),
      memoryUsageResponse.getMemoryCapacityBytes(),
    );
  }
}
