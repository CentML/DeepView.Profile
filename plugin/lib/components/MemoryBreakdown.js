'use babel';

import React from 'react';
import {connect} from 'react-redux';

import PerfBarContainer from './generic/PerfBarContainer';
import MemoryEntryLabel from '../models/MemoryEntryLabel';
import PerfVisState from '../models/PerfVisState';
import MemoryPerfBar from './MemoryPerfBar';
import {toPercentage} from '../utils';

const DEFAULT_MEMORY_LABELS = [
  {label: MemoryEntryLabel.Weights, percentage: 30, clickable: true},
  {label: MemoryEntryLabel.Activations, percentage: 55, clickable: true},
  {label: MemoryEntryLabel.Untracked, percentage: 15, clickable: false},
];

const MEMORY_LABEL_ORDER = DEFAULT_MEMORY_LABELS.map(({label}) => label);

const COLORS_BY_LABEL = {
  [MemoryEntryLabel.Weights]: [
    'innpv-green-color-1',
    'innpv-green-color-2',
    'innpv-green-color-3',
    'innpv-green-color-4',
    'innpv-green-color-5',
  ],
  [MemoryEntryLabel.Activations]: [
    'innpv-blue-color-1',
    'innpv-blue-color-2',
    'innpv-blue-color-3',
    'innpv-blue-color-4',
    'innpv-blue-color-5',
  ],
  [MemoryEntryLabel.Untracked]: [
    'innpv-untracked-color',
  ],
}

class MemoryBreakdown extends React.Component {
  constructor(props) {
    super(props);
    this._renderPerfBars = this._renderPerfBars.bind(this);
  }

  _getLabels() {
    const {memoryBreakdown} = this.props;
    if (memoryBreakdown == null) {
      return DEFAULT_MEMORY_LABELS;
    }

    const peakUsageBytes = memoryBreakdown.getPeakUsageBytes();
    return DEFAULT_MEMORY_LABELS.map(({label, ...rest}) => ({
      ...rest,
      label,
      percentage: toPercentage(memoryBreakdown.getTotalSizeBytesByLabel(label), peakUsageBytes),
    }));
  }

  _entryKey(entry, index) {
    // TODO: Use a stable identifier (presumably an id from the report database)
    return `${entry.name}-idx${index}`;
  }

  _renderPerfBars(expanded) {
    const {editorsByPath, projectRoot, memoryBreakdown} = this.props;
    if (memoryBreakdown == null) {
      return null;
    }

    const results = [];
    const peakUsageBytes = memoryBreakdown.getPeakUsageBytes();

    // [].flatMap() is a nicer way to do this, but it is not yet available.
    MEMORY_LABEL_ORDER.forEach(label => {
      const totalSizeBytes = memoryBreakdown.getTotalSizeBytesByLabel(label);
      const colors = COLORS_BY_LABEL[label];

      memoryBreakdown.getEntriesByLabel(label).forEach((entry, index) => {
        const overallPct = toPercentage(entry.sizeBytes, peakUsageBytes);

        // This helps account for "expanded" labels
        let displayPct = 0.001;
        if (expanded == null) {
          displayPct = overallPct;
        } else if (label === expanded) {
          displayPct = toPercentage(entry.sizeBytes, totalSizeBytes);
        }

        const editors = entry.filePath != null ? editorsByPath.get(entry.filePath) : [];

        results.push(
          <MemoryPerfBar
            key={this._entryKey(entry, index)}
            memoryEntry={entry}
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
  memoryBreakdown: state.memoryBreakdown,
  ...ownProps,
});

export default connect(mapStateToProps)(MemoryBreakdown);
