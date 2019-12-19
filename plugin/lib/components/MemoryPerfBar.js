'use babel';

import React from 'react';

import PerfBar from './generic/PerfBar';
import UsageHighlight from './UsageHighlight';
import {toReadableByteSize} from '../utils';

class MemoryPerfBar extends React.Component {
  constructor(props) {
    super(props);
    this._renderPerfHints = this._renderPerfHints.bind(this);
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

  render() {
    const {memoryEntry, editors, ...rest} = this.props;
    return (
      <PerfBar
        renderPerfHints={this._renderPerfHints}
        tooltipHTML={this._generateTooltipHTML()}
        {...rest}
      />
    );
  }
}

MemoryPerfBar.defaultProps = {
  editors: [],
};

export default MemoryPerfBar;
