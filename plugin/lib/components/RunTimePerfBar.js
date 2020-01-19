'use babel';

import path from 'path';
import React from 'react';

import PerfBar from './generic/PerfBar';
import UsageHighlight from './UsageHighlight';

export default class RunTimePerfBar extends React.Component {
  constructor(props) {
    super(props);
    this._onClick = this._onClick.bind(this);
    this._renderPerfHints = this._renderPerfHints.bind(this);
  }

  _generateTooltipHTML() {
    const {runTimeEntry, overallPct} = this.props;
    return `<strong>${runTimeEntry.name}</strong><br/>` +
      `${runTimeEntry.runTimeMs.toFixed(2)} ms<br/>` +
      `${overallPct.toFixed(2)}%`;
  }

  _renderPerfHints(isActive, perfHintState) {
    const {editors, runTimeEntry} = this.props;

    return editors.map(editor => (
      <UsageHighlight
        key={editor.id}
        editor={editor}
        lineNumber={runTimeEntry.lineNumber}
        show={isActive}
      />
    ));
  }

  _onClick() {
    const {runTimeEntry, projectRoot} = this.props;
    if (runTimeEntry.filePath == null || projectRoot == null) {
      return;
    }

    // Atom uses 0-based line numbers, but INNPV uses 1-based line numbers
    const absoluteFilePath = path.join(projectRoot, runTimeEntry.filePath);
    atom.workspace.open(absoluteFilePath, {initialLine: runTimeEntry.lineNumber - 1});
  }

  render() {
    const {runTimeEntry, editors, ...rest} = this.props;
    return (
      <PerfBar
        clickable={runTimeEntry.filePath != null}
        renderPerfHints={this._renderPerfHints}
        tooltipHTML={this._generateTooltipHTML()}
        onClick={this._onClick}
        {...rest}
      />
    );
  }
}
