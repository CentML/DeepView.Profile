'use babel';

import path from 'path';

const BYTE_UNITS = [
  'B',
  'KB',
  'MB',
  'GB',
];

export default {
  processFileReference(fileReferenceProto) {
    return {
      filePath: path.join(...(fileReferenceProto.getFilePath().getComponentsList())),
      lineNumber: fileReferenceProto.getLineNumber(),
    };
  },

  toPercentage(numerator, denominator) {
    return numerator / denominator * 100;
  },

  toReadableByteSize(sizeBytes) {
    let index = 0;
    let size = sizeBytes;
    for (; index < BYTE_UNITS.length; index++) {
      if (size < 1000) {
        break;
      }
      size /= 1024;
    }
    if (index == BYTE_UNITS.length) {
      index--;
    }

    return `${size.toFixed(1)} ${BYTE_UNITS[index]}`;
  },
};
