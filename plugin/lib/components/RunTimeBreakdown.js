'use babel';

import React from 'react';
import {connect} from 'react-redux';

import RunTimePerfBar from './RunTimePerfBar';
import RunTimeEntryLabel from '../models/RunTimeEntryLabel';
import PerfBarContainer from './generic/PerfBarContainer';
import PerfVisState from '../models/PerfVisState';
import {OperationNode} from '../models/Breakdown';
import {toPercentage} from '../utils';

const DEFAULT_LABELS = [
  {label: RunTimeEntryLabel.ForwardBackward, percentage: 90, clickable: true},
  {label: RunTimeEntryLabel.Untracked, percentage: 10, clickable: false},
];

const LABEL_ORDER = DEFAULT_LABELS.map(({label}) => label);

const COLORS_BY_LABEL = {
  [RunTimeEntryLabel.ForwardBackward]: [
    'innpv-green-color-1',
    'innpv-green-color-2',
    'innpv-green-color-3',
    'innpv-green-color-4',
    'innpv-green-color-5',
  ],
  [RunTimeEntryLabel.Untracked]: [
    'innpv-untracked-color',
  ],
}

class RunTimeBreakdown extends React.Component {
  constructor(props) {
    super(props);
    this._renderPerfBars = this._renderPerfBars.bind(this);
  }

  _getLabels() {
    const {operationTree} = this.props;
    if (operationTree == null) {
      return DEFAULT_LABELS;
    }

    const {iterationRunTimeMs} = this.props;
    const {untrackedMs} = this.props.runTime;

    return DEFAULT_LABELS.map(({label, ...rest}) => ({
      ...rest,
      label,
      percentage: label === RunTimeEntryLabel.ForwardBackward
        ? toPercentage(operationTree.runTimeMs, iterationRunTimeMs)
        : toPercentage(untrackedMs, iterationRunTimeMs),
    }));
  }

  _renderPerfBars(expanded) {
    const {operationTree} = this.props;
    if (operationTree == null) {
      return null;
    }

    const {editorsByPath, projectRoot, iterationRunTimeMs} = this.props;
    const colors = COLORS_BY_LABEL[RunTimeEntryLabel.ForwardBackward];

    const results = operationTree.children.map((operationNode, index) => {
      const overallPct = toPercentage(operationNode.runTimeMs, iterationRunTimeMs);

      // This helps account for "expanded" labels
      let displayPct = 0.001;
      if (expanded == null) {
        displayPct = overallPct;
      } else if (expanded === RunTimeEntryLabel.ForwardBackward) {
        displayPct = toPercentage(operationNode.runTimeMs, operationTree.runTimeMs);
      }

      return (
        <RunTimePerfBar
          key={operationNode.id}
          operationNode={operationNode}
          projectRoot={projectRoot}
          editorsByPath={editorsByPath}
          overallPct={overallPct}
          resizable={false}
          percentage={displayPct}
          colorClass={colors[index % colors.length]}
        />
      );
    });

    const {untrackedMs, untrackedNode} = this.props.runTime;
    if (untrackedNode == null) {
      return results;
    }

    // Append the extra "Untracked" category
    const otherPct = toPercentage(untrackedMs, iterationRunTimeMs);
    results.push(
      <RunTimePerfBar
        key={untrackedNode.id}
        operationNode={untrackedNode}
        projectRoot={projectRoot}
        overallPct={otherPct}
        resizable={false}
        percentage={otherPct}
        colorClass={COLORS_BY_LABEL[RunTimeEntryLabel.Untracked][0]}
      />
    );

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
  operationTree: state.breakdown.operationTree,
  runTime: state.breakdown.runTime,
  iterationRunTimeMs: state.iterationRunTimeMs,
  ...ownProps,
});

export default connect(mapStateToProps)(RunTimeBreakdown);
