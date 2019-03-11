'use babel';

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
};
