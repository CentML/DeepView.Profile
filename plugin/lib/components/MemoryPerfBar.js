'use babel';

import path from 'path';
import React from 'react';

import PerfBar from './generic/PerfBar';
import UsageHighlight from './UsageHighlight';
import {toReadableByteSize} from '../utils';

class MemoryPerfBar extends React.Component {
  constructor(props) {
    super(props);
    this._renderPerfHints = this._renderPerfHints.bind(this);
    this._onClick = this._onClick.bind(this);
  }

  _generateTooltipHTML() {
    const {memoryEntry, overallPct} = this.props;
    return `<strong>${memoryEntry.name}</strong><br/>` +
      `${toReadableByteSize(memoryEntry.sizeBytes)}<br/>` +
      `${overallPct.toFixed(2)}%`;
  }

  _renderPerfHints(isActive, perfHintState) {
    const {editors, memoryEntry} = this.props;

    return editors.map(editor => (
      <UsageHighlight
        key={editor.id}
        editor={editor}
        lineNumber={memoryEntry.lineNumber}
        show={isActive}
      />
    ));
  }

  _onClick() {
    const {memoryEntry, projectRoot} = this.props;
    if (memoryEntry.filePath == null || projectRoot == null) {
      return;
    }

    // Atom uses 0-based line numbers, but INNPV uses 1-based line numbers
    const absoluteFilePath = path.join(projectRoot, memoryEntry.filePath);
    atom.workspace.open(absoluteFilePath, {initialLine: memoryEntry.lineNumber - 1});
  }

  render() {
    const {memoryEntry, editors, ...rest} = this.props;
    return (
      <PerfBar
        clickable={memoryEntry.filePath != null}
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

export default MemoryPerfBar;
