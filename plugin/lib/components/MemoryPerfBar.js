'use babel';

import path from 'path';
import React from 'react';

import PerfBar from './generic/PerfBar';
import UsageHighlight from './UsageHighlight';
import {toReadableByteSize} from '../utils';

import Events from '../telemetry/events';
import TelemetryClientContext from '../telemetry/react_context';

class MemoryPerfBar extends React.Component {
  constructor(props) {
    super(props);
    this._renderPerfHints = this._renderPerfHints.bind(this);
    this._onClick = this._onClick.bind(this);
  }

  _generateTooltipHTML() {
    const {memoryNode, overallPct} = this.props;
    return `<strong>${memoryNode.name}</strong><br/>` +
      `${toReadableByteSize(memoryNode.sizeBytes)}<br/>` +
      `${overallPct.toFixed(2)}%`;
  }

  _renderPerfHints(isActive, perfHintState) {
    const {editorsByPath, memoryNode} = this.props;

    return memoryNode.contexts.flatMap(({filePath, lineNumber}) => {
      if (!editorsByPath.has(filePath)) {
        return [];
      }
      return editorsByPath.get(filePath).map((editor) => (
        <UsageHighlight
          key={`memory-${editor.id}-${filePath}-${lineNumber}`}
          editor={editor}
          lineNumber={lineNumber}
          show={isActive}
        />
      ));
    });
  }

  _onClick() {
    const {memoryNode, projectRoot} = this.props;
    if (memoryNode.contexts.length === 0 || projectRoot == null) {
      return;
    }

    // Atom uses 0-based line numbers, but INNPV uses 1-based line numbers
    const context = memoryNode.contexts[0];
    const absoluteFilePath = path.join(projectRoot, context.filePath);
    atom.workspace.open(absoluteFilePath, {initialLine: context.lineNumber - 1});
    this.context.record(
      Events.Interaction.CLICKED_MEMORY_ENTRY,
      {label: memoryNode.name},
    );
  }

  render() {
    const {memoryNode, editorsByPath, ...rest} = this.props;
    return (
      <PerfBar
        clickable={memoryNode.contexts.length > 0}
        renderPerfHints={this._renderPerfHints}
        tooltipHTML={this._generateTooltipHTML()}
        onClick={this._onClick}
        {...rest}
      />
    );
  }
}

MemoryPerfBar.defaultProps = {
  editors: [],
};

MemoryPerfBar.contextType = TelemetryClientContext;

export default MemoryPerfBar;
