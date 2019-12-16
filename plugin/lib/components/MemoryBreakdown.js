'use babel';

import React from 'react';

import AnalysisStore from '../stores/analysis_store';
import PerfBarContainer from './generic/PerfBarContainer';
import MemoryEntryLabel from '../models/MemoryEntryLabel';
import PerfBar from './generic/PerfBar';
import {toReadableByteSize} from '../utils';

const DEFAULT_MEMORY_LABELS = [
  {label: MemoryEntryLabel.Weights, percentage: 30},
  {label: MemoryEntryLabel.Activations, percentage: 55},
  {label: MemoryEntryLabel.Untracked, percentage: 15},
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

export default class MemoryPerfBarContainer extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      memoryBreakdown: AnalysisStore.getMemoryBreakdown(),
    };

    this._onStoreUpdate = this._onStoreUpdate.bind(this);
  }

  componentDidMount() {
    AnalysisStore.addListener(this._onStoreUpdate);
  }

  componentWillUnmount() {
    AnalysisStore.removeListener(this._onStoreUpdate);
  }

  _onStoreUpdate() {
    this.setState({
      memoryBreakdown: AnalysisStore.getMemoryBreakdown(),
    });
  }

  _getLabels() {
    const {memoryBreakdown} = this.state;
    if (memoryBreakdown == null) {
      return DEFAULT_MEMORY_LABELS;
    }

    return DEFAULT_MEMORY_LABELS.map(({label}) => ({
      label,
      percentage: memoryBreakdown.getOverallDisplayPctByLabel(label),
    }));
  }

  _entryKey(entry, index) {
    if (entry.filePath != null && entry.lineNumber != null) {
      return `${entry.name}-${entry.filePath}-${entry.lineNumber}`;
    }
    return `${entry.name}-idx${index}`;
  }

  _entryTooltipHTML(entry) {
    return `<strong>${entry.name}</strong><br/>` +
      `${toReadableByteSize(entry.sizeBytes)}<br/>` +
      `${entry.displayPct.toFixed(2)}%`;
  }

  _renderPerfBars() {
    const {memoryBreakdown} = this.state;
    if (memoryBreakdown == null) {
      return;
    }

    // [].flatMap() is a nicer way to do this, but it is not yet available.
    const results = [];
    MEMORY_LABEL_ORDER.forEach(label => {
      memoryBreakdown.getEntriesByLabel(label).forEach((entry, index) => {
        const colors = COLORS_BY_LABEL[label];
        results.push(
          <PerfBar
            key={this._entryKey(entry, index)}
            resizable={false}
            percentage={entry.displayPct}
            colorClass={colors[index % colors.length]}
            tooltipHTML={this._entryTooltipHTML(entry)}
          />
        );
      })
    });

    return results;
  }

  render() {
    return (
      <PerfBarContainer labels={this._getLabels()}>
        {this._renderPerfBars()}
      </PerfBarContainer>
    );
  }
}
