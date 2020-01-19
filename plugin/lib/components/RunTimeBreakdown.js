'use babel';

import React from 'react';
import {connect} from 'react-redux';

import RunTimePerfBar from './RunTimePerfBar';
import RunTimeEntryLabel from '../models/RunTimeEntryLabel';
import PerfBarContainer from './generic/PerfBarContainer';
import PerfVisState from '../models/PerfVisState';
import {toPercentage} from '../utils';

const DEFAULT_LABELS = [
  {label: RunTimeEntryLabel.Forward, percentage: 30, clickable: true},
  {label: RunTimeEntryLabel.Backward, percentage: 60, clickable: true},
  {label: RunTimeEntryLabel.Other, percentage: 10, clickable: false},
];

const LABEL_ORDER = DEFAULT_LABELS.map(({label}) => label);

const COLORS_BY_LABEL = {
  [RunTimeEntryLabel.Forward]: [
    'innpv-green-color-1',
    'innpv-green-color-2',
    'innpv-green-color-3',
    'innpv-green-color-4',
    'innpv-green-color-5',
  ],
  [RunTimeEntryLabel.Backward]: [
    'innpv-blue-color-1',
    'innpv-blue-color-2',
    'innpv-blue-color-3',
    'innpv-blue-color-4',
    'innpv-blue-color-5',
  ],
  [RunTimeEntryLabel.Other]: [
    'innpv-untracked-color',
  ],
}

class RunTimeBreakdown extends React.Component {
  constructor(props) {
    super(props);
    this._renderPerfBars = this._renderPerfBars.bind(this);
  }

  _getLabels() {
    const {runTimeBreakdown} = this.props;
    if (runTimeBreakdown == null) {
      return DEFAULT_LABELS;
    }

    const iterationRunTimeMs = runTimeBreakdown.getIterationRunTimeMs();
    return DEFAULT_LABELS.map(({label, ...rest}) => ({
      ...rest,
      label,
      percentage: toPercentage(runTimeBreakdown.getTotalTimeMsByLabel(label), iterationRunTimeMs),
    }));
  }

  _entryKey(entry, index) {
    if (entry.filePath != null && entry.lineNumber != null) {
      return `${entry.name}-${entry.filePath}-${entry.lineNumber}`;
    }
    return `${entry.name}-idx${index}`;
  }

  _renderPerfBars(expanded) {
    const {editorsByPath, projectRoot, runTimeBreakdown} = this.props;
    if (runTimeBreakdown == null) {
      return null;
    }

    const results = [];
    const iterationRunTimeMs = runTimeBreakdown.getIterationRunTimeMs();

    // [].flatMap() is a nicer way to do this, but it is not yet available.
    LABEL_ORDER.forEach(label => {
      const totalTimeMs = runTimeBreakdown.getTotalTimeMsByLabel(label);
      const colors = COLORS_BY_LABEL[label];

      runTimeBreakdown.getEntriesByLabel(label).forEach((entry, index) => {
        const overallPct = toPercentage(entry.runTimeMs, iterationRunTimeMs);

        // This helps account for "expanded" labels
        let displayPct = 0.001;
        if (expanded == null) {
          displayPct = overallPct;
        } else if (label === expanded) {
          displayPct = toPercentage(entry.runTimeMs, totalTimeMs);
        }

        const editors = entry.filePath != null ? editorsByPath.get(entry.filePath) : [];

        results.push(
          <RunTimePerfBar
            key={this._entryKey(entry, index)}
            runTimeEntry={entry}
            projectRoot={projectRoot}
            editors={editors}
            overallPct={overallPct}
            resizable={false}
            percentage={displayPct}
            colorClass={colors[index % colors.length]}
          />
        );
      })
    });

    return results;
  }

  render() {
    const {perfVisState} = this.props;
    const disabled = perfVisState === PerfVisState.MODIFIED ||
      perfVisState === PerfVisState.ANALYZING;

    return (
      <PerfBarContainer
        disabled={disabled}
        labels={this._getLabels()}
        renderPerfBars={this._renderPerfBars}
      />
    );
  }
}

const mapStateToProps = (state, ownProps) => ({
  editorsByPath: state.editorsByPath,
  runTimeBreakdown: state.runTimeBreakdown,
  ...ownProps,
});

export default connect(mapStateToProps)(RunTimeBreakdown);
