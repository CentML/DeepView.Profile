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
    const {perfVisState, currentView} = this.props;

    if (perfVisState === PerfVisState.EXPLORING_WEIGHTS) {
      return [{
        label: `${currentView.name} (Weights)`,
        percentage: 100,
        clickable: false,
      }];

    } else if (perfVisState === PerfVisState.EXPLORING_OPERATIONS) {
      return [{
        label: `${currentView.name} (Activations)`,
        percentage: 100,
        clickable: false,
      }];

    } else {
      return this._getOverviewLabels();
    }
  }

  _getOverviewLabels() {
    const {operationTree, weightTree} = this.props;
    if (operationTree == null || weightTree == null) {
      return DEFAULT_MEMORY_LABELS;
    }

    const {peakUsageBytes, memory} = this.props;
    const {untrackedBytes} = memory;

    return DEFAULT_MEMORY_LABELS.map(({label, ...rest}) => {
      let percentage = 0.;
      if (label === MemoryEntryLabel.Weights) {
        percentage = toPercentage(weightTree.sizeBytes, peakUsageBytes);
      } else if (label === MemoryEntryLabel.Activations) {
        percentage = toPercentage(operationTree.sizeBytes, peakUsageBytes);
      } else {
        percentage = toPercentage(untrackedBytes, peakUsageBytes);
      }
      return {
        ...rest,
        label,
        percentage,
      };
    });
  }

  _renderPerfBars(expanded) {
    const {perfVisState} = this.props;
    if (perfVisState !== PerfVisState.EXPLORING_WEIGHTS &&
        perfVisState !== PerfVisState.EXPLORING_OPERATIONS) {
      return this._renderOverviewPerfBars(expanded);
    }

    const {currentView, peakUsageBytes} = this.props;
    const label = perfVisState === PerfVisState.EXPLORING_WEIGHTS
      ? MemoryEntryLabel.Weights
      : MemoryEntryLabel.Activations;
    const colors = COLORS_BY_LABEL[label];

    return currentView.childrenBySize
      .filter((node) => node.sizeBytes > 0)
      .map((node, index) => {
        const overallPct = toPercentage(node.sizeBytes, peakUsageBytes);
        const displayPct = toPercentage(node.sizeBytes, currentView.sizeBytes);

        return this._renderPerfBarFrom({
          node,
          label,
          overallPct,
          displayPct,
          color: colors[index % colors.length],
        });
      });
  }

  _renderOverviewPerfBars(expanded) {
    const {operationTree, weightTree} = this.props;
    if (operationTree == null || weightTree == null) {
      return null;
    }

    const {editorsByPath, projectRoot, peakUsageBytes} = this.props;

    const results = [weightTree, operationTree].flatMap((tree, idx) => {
      const label = idx === 0
        ? MemoryEntryLabel.Weights
        : MemoryEntryLabel.Activations;
      const colors = COLORS_BY_LABEL[label];

      return tree.childrenBySize.map((node, index) => {
        const overallPct = toPercentage(node.sizeBytes, peakUsageBytes);

        // This helps account for "expanded" labels
        let displayPct = 0.001;
        if (expanded == null) {
          displayPct = overallPct;
        } else if (label === expanded) {
          displayPct = toPercentage(
            node.sizeBytes,
            label === MemoryEntryLabel.Weights
              ? weightTree.sizeBytes
              : operationTree.sizeBytes,
          );
        }

        return this._renderPerfBarFrom({
          node,
          label,
          overallPct,
          displayPct,
          color: colors[index % colors.length],
        });
      });
    });

    const {untrackedBytes, untrackedNode} = this.props.memory;
    if (untrackedNode == null) {
      return results;
    }

    const untrackedPct = toPercentage(untrackedBytes, peakUsageBytes);
    results.push(this._renderPerfBarFrom({
      node: untrackedNode,
      label: 'Untracked',
      overallPct: untrackedPct,
      displayPct: untrackedPct,
      color: COLORS_BY_LABEL[MemoryEntryLabel.Untracked][0],
    }));

    return results;
  }

  _renderPerfBarFrom({node, label, overallPct, displayPct, color}) {
    const {projectRoot, editorsByPath, currentlyActive} = this.props;
    return (
      <MemoryPerfBar
        key={`${label}-${node.id}`}
        isActive={node === currentlyActive}
        label={label}
        memoryNode={node}
        projectRoot={projectRoot}
        editorsByPath={editorsByPath}
        overallPct={overallPct}
        resizable={false}
        percentage={displayPct}
        colorClass={color}
      />
    );
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
  operationTree: state.breakdown.operationTree,
  weightTree: state.breakdown.weightTree,
  currentView: state.breakdown.currentView,
  currentlyActive: state.breakdown.currentlyActive,
  memory: state.memory,
  peakUsageBytes: state.peakUsageBytes,
  ...ownProps,
});

export default connect(mapStateToProps)(MemoryBreakdown);
