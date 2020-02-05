'use babel';

import path from 'path';
import React from 'react';
import {connect} from 'react-redux';

import PerfBar from './generic/PerfBar';
import UsageHighlight from './UsageHighlight';
import AnalysisActions from '../redux/actions/analysis';

import Events from '../telemetry/events';
import TelemetryClientContext from '../telemetry/react_context';

class RunTimePerfBar extends React.Component {
  constructor(props) {
    super(props);
    this._onClick = this._onClick.bind(this);
    this._onDoubleClick = this._onDoubleClick.bind(this);
    this._renderPerfHints = this._renderPerfHints.bind(this);
    this._onActiveChange = this._onActiveChange.bind(this);
  }

  _generateTooltipHTML() {
    const {operationNode, overallPct} = this.props;
    return `<strong>${operationNode.name}</strong><br/>` +
      `${operationNode.runTimeMs.toFixed(2)} ms<br/>` +
      `${overallPct.toFixed(2)}%`;
  }

  _renderPerfHints(perfHintState) {
    const {editorsByPath, operationNode, isActive} = this.props;

    return operationNode.contexts.flatMap(({filePath, lineNumber}) => {
      if (!editorsByPath.has(filePath)) {
        return [];
      }
      return editorsByPath.get(filePath).map((editor) => (
        <UsageHighlight
          key={`time-${editor.id}-${filePath}-${lineNumber}`}
          editor={editor}
          lineNumber={lineNumber}
          show={isActive}
        />
      ));
    });
  }

  _onClick() {
    const {operationNode, projectRoot} = this.props;
    if (operationNode.contexts.length == 0 || projectRoot == null) {
      return;
    }

    // Atom uses 0-based line numbers, but INNPV uses 1-based line numbers
    const context = operationNode.contexts[0];
    const absoluteFilePath = path.join(projectRoot, context.filePath);
    atom.workspace.open(absoluteFilePath, {initialLine: context.lineNumber - 1});
    this.context.record(
      Events.Interaction.CLICKED_RUN_TIME_ENTRY,
      {label: operationNode.name},
    );
  }

  _onDoubleClick() {
    const {operationNode} = this.props;
    if (operationNode.children.length === 0) {
      return;
    }
    this.props.dispatch(
      AnalysisActions.exploreOperation({newView: operationNode}),
    );
  }

  _onActiveChange(isActive) {
    const {dispatch, operationNode} = this.props;
    const currentlyActive = isActive ? operationNode : null;
    dispatch(
      AnalysisActions.setActive({currentlyActive}),
    );
  }

  render() {
    const {operationNode, editorsByPath, ...rest} = this.props;
    return (
      <PerfBar
        clickable={operationNode.contexts.length > 0}
        renderPerfHints={this._renderPerfHints}
        tooltipHTML={this._generateTooltipHTML()}
        onClick={this._onClick}
        onDoubleClick={this._onDoubleClick}
        onActiveChange={this._onActiveChange}
        {...rest}
      />
    );
  }
}

RunTimePerfBar.defaultProps = {
  editorsByPath: new Map(),
};

RunTimePerfBar.contextType = TelemetryClientContext;

export default connect(null)(RunTimePerfBar);
