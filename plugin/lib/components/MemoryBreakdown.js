'use babel';

import React from 'react';

import AnalysisStore from '../stores/analysis_store';
import ProjectStore from '../stores/project_store';
import PerfBarContainer from './generic/PerfBarContainer';
import MemoryEntryLabel from '../models/MemoryEntryLabel';
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

export default class MemoryPerfBarContainer extends React.Component {
  constructor(props) {
    super(props);
    const memoryBreakdown = AnalysisStore.getMemoryBreakdown();
    this.state = {
      memoryBreakdown,
      textEditorMap: this._getTextEditorMap(memoryBreakdown),
      expanded: null,
    };

    this._onLabelClick = this._onLabelClick.bind(this);
    this._onAnalysisStoreUpdate = this._onAnalysisStoreUpdate.bind(this);
    this._onProjectStoreUpdate = this._onProjectStoreUpdate.bind(this);
  }

  componentDidMount() {
    AnalysisStore.addListener(this._onAnalysisStoreUpdate);
    ProjectStore.addListener(this._onProjectStoreUpdate);
  }

  componentWillUnmount() {
    AnalysisStore.removeListener(this._onAnalysisStoreUpdate);
    ProjectStore.removeListener(this._onProjectStoreUpdate);
  }

  _onAnalysisStoreUpdate() {
    const memoryBreakdown = AnalysisStore.getMemoryBreakdown();
    this.setState({
      memoryBreakdown,
      textEditorMap: this._getTextEditorMap(memoryBreakdown),
    });
  }

  _onProjectStoreUpdate() {
    this.setState({
      textEditorMap: this._getTextEditorMap(this.state.memoryBreakdown),
    });
  }

  _getTextEditorMap(memoryBreakdown) {
    const editorMap = new Map();
    if (memoryBreakdown == null) {
      return editorMap;
    }

    [MemoryEntryLabel.Weights, MemoryEntryLabel.Activations].forEach(label => {
      memoryBreakdown.getEntriesByLabel(label).forEach(entry => {
        if (entry.filePath == null || editorMap.has(entry.filePath)) {
          return;
        }
        editorMap.set(entry.filePath, ProjectStore.getTextEditorsFor(entry.filePath));
      });
    });

    return editorMap;
  }

  _getLabels() {
    const {memoryBreakdown, expanded} = this.state;
    if (memoryBreakdown == null) {
      return DEFAULT_MEMORY_LABELS;
    }

    if (expanded != null) {
      return DEFAULT_MEMORY_LABELS.map(({label, ...rest}) => ({
        ...rest,
        label,
        percentage: expanded === label ? 100 : 0.001,
      }));
    }

    const peakUsageBytes = memoryBreakdown.getPeakUsageBytes();
    return DEFAULT_MEMORY_LABELS.map(({label, ...rest}) => ({
      ...rest,
      label,
      percentage: toPercentage(memoryBreakdown.getTotalSizeBytesByLabel(label), peakUsageBytes),
    }));
  }

  _onLabelClick(label) {
    const {expanded} = this.state;
    if (expanded == null && label !== MemoryEntryLabel.Untracked) {
      this.setState({expanded: label});
    } else if (expanded === label) {
      this.setState({expanded: null});
    }
  }

  _entryKey(entry, index) {
    if (entry.filePath != null && entry.lineNumber != null) {
      return `${entry.name}-${entry.filePath}-${entry.lineNumber}`;
    }
    return `${entry.name}-idx${index}`;
  }

  _renderPerfBars() {
    const {memoryBreakdown, expanded, textEditorMap} = this.state;
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

        const editors = entry.filePath != null ? textEditorMap.get(entry.filePath) : [];

        results.push(
          <MemoryPerfBar
            key={this._entryKey(entry, index)}
            memoryEntry={entry}
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
    return (
      <PerfBarContainer
        labels={this._getLabels()}
        onLabelClick={this._onLabelClick}
      >
        {this._renderPerfBars()}
      </PerfBarContainer>
    );
  }
}
