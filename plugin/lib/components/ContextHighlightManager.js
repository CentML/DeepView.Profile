'use babel';

import React from 'react';
import {connect} from 'react-redux';

import ContextHighlight from './ContextHighlight';

class ContextHighlightManager extends React.Component {
  render() {
    const {editorsByPath, operationTree} = this.props;
    if (operationTree == null) {
      return null;
    }

    const {iterationRunTimeMs, peakUsageBytes, currentView} = this.props;
    const results = [];

    operationTree.contextInfos.forEach((contextInfo, contextKey) => {
      const {context} = contextInfo;
      if (!editorsByPath.has(context.filePath)) {
        return;
      }

      const isScoped = currentView != null;
      const scopedContextInfo = isScoped
        ? currentView.contextInfos.get(contextKey)
        : null;

      editorsByPath.get(context.filePath).forEach((editor) => {
        results.push(
          <ContextHighlight
            key={`${editor.id}-${context.filePath}-${context.lineNumber}`}
            editor={editor}
            lineNumber={context.lineNumber}
            iterationRunTimeMs={iterationRunTimeMs}
            peakUsageBytes={peakUsageBytes}
            isScoped={isScoped}
            contextInfo={contextInfo}
            scopedContextInfo={scopedContextInfo}
          />
        );
      });
    });

    return results;
  }
}

const mapStateToProps = (state) => ({
  editorsByPath: state.editorsByPath,
  operationTree: state.breakdown.operationTree,
  currentView: state.breakdown.currentView,
  iterationRunTimeMs: state.iterationRunTimeMs,
  peakUsageBytes: state.peakUsageBytes,
});

export default connect(mapStateToProps)(ContextHighlightManager);
