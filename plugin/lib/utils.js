'use babel';

import path from 'path';

const BYTE_UNITS = [
  'B',
  'KB',
  'MB',
  'GB',
];

export function processFileReference(fileReferenceProto) {
  return {
    filePath: path.join(...(fileReferenceProto.getFilePath().getComponentsList())),
    lineNumber: fileReferenceProto.getLineNumber(),
  };
};

export function toPercentage(numerator, denominator) {
  return numerator / denominator * 100;
};

export function toReadableByteSize(sizeBytes) {
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
};

export function scalePercentages({scaleSelector, shouldScale, applyFactor}) {
  return function(list, scaleFactor) {
    let total = 0;
    const adjusted = [];

    for (const element of list) {
      const value = scaleSelector(element);
      if (shouldScale(element)) {
        const scaled = value * scaleFactor;
        adjusted.push([scaled, element]);
        total += scaled;
      } else {
        adjusted.push([value, element]);
        total += value;
      }
    }

    return adjusted.map(([newValue, element]) =>
      applyFactor(element, toPercentage(newValue, total)));
  };
};
