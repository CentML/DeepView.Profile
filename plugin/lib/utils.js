'use babel';

import path from 'path';

const BYTE_UNITS = [
  'B',
  'KB',
  'MB',
  'GB',
];

export default {
  getTextEditor(newEditor) {
    return new Promise((res) => {
      if (newEditor) {
        return res(atom.workspace.open());
      }
      const editor = atom.workspace.getActiveTextEditor();
      if (editor) {
        return res(editor);
      }
      // Open a new text editor if one is not open
      return res(atom.workspace.open());
    });
  },

  evaluateLinearModel(model, x) {
    return model.getCoefficient() * x + model.getBias();
  },

  getBatchSizeFromUsage(usageModelMb, usageMb) {
    return (usageMb - usageModelMb.getBias()) / usageModelMb.getCoefficient();
  },

  getBatchSizeFromThroughput(runtimeModelMs, throughput) {
    const throughputMs = throughput / 1000;
    return (throughputMs * runtimeModelMs.getBias()) /
      (1 - throughputMs * runtimeModelMs.getCoefficient());
  },

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
