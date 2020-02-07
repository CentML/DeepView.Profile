'use babel';

import React from 'react';
import {connect} from 'react-redux';

import ContextHighlight from './ContextHighlight';
import {toPercentage} from '../utils';

class ContextHighlightManager extends React.Component {
  render() {
    const {editorsByPath, operationTree} = this.props;
    if (operationTree == null) {
      return null;
    }

    const {iterationRunTimeMs, peakUsageBytes} = this.props;
    return operationTree.contextInfos.flatMap((contextInfo) => {
      const {context} = contextInfo;
      if (!editorsByPath.has(context.filePath)) {
        return [];
      }

      const {runTimeMs, sizeBytes, invocations} = contextInfo;
      return editorsByPath.get(context.filePath).map((editor) => (
        <ContextHighlight
          key={`${editor.id}-${context.filePath}-${context.lineNumber}`}
          editor={editor}
          lineNumber={context.lineNumber}
          invocations={invocations}
          runTimePct={toPercentage(runTimeMs, iterationRunTimeMs)}
          memoryPct={toPercentage(sizeBytes, peakUsageBytes)}
        />
      ));
    });
  }
}

const mapStateToProps = (state) => ({
  editorsByPath: state.editorsByPath,
  operationTree: state.breakdown.operationTree,
  iterationRunTimeMs: state.iterationRunTimeMs,
  peakUsageBytes: state.peakUsageBytes,
});

export default connect(mapStateToProps)(ContextHighlightManager);
