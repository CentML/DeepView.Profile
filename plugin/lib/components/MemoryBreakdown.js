'use babel';

import React from 'react';

import AnalysisStore from '../stores/analysis_store';
import PerfBarContainer from './generic/PerfBarContainer';
import MemoryEntryLabel from '../models/MemoryEntryLabel';

const DEFAULT_MEMORY_LABELS = [
  {label: MemoryEntryLabel.Weights, percentage: 30},
  {label: MemoryEntryLabel.Activations, percentage: 55},
  {label: MemoryEntryLabel.Untracked, percentage: 15},
];

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

  render() {
    return (
      <PerfBarContainer labels={this._getLabels()}>
      </PerfBarContainer>
    );
  }
}
