'use babel';

import React from 'react';
import {connect} from 'react-redux';

import ContextHighlight from './ContextHighlight';
import PerfVisState from '../models/PerfVisState';

class ContextHighlightManager extends React.Component {
  render() {
    const {editorsByPath, operationTree, perfVisState} = this.props;
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

      const isScoped = currentView != null &&
        perfVisState === PerfVisState.EXPLORING_OPERATIONS;
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
            scopeName={isScoped ? currentView.name : null}
          />
        );
      });
    });

    return results;
  }
}

const mapStateToProps = (state, ownProps) => ({
  editorsByPath: state.editorsByPath,
  operationTree: state.breakdown.operationTree,
  currentView: state.breakdown.currentView,
  iterationRunTimeMs: state.iterationRunTimeMs,
  peakUsageBytes: state.peakUsageBytes,
  ...ownProps,
});

export default connect(mapStateToProps)(ContextHighlightManager);
