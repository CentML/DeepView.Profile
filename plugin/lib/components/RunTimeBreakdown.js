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
  {label: RunTimeEntryLabel.Other, percentage: 10, clickable: false},
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
    const {breakdown} = this.props;
    if (breakdown == null) {
      return DEFAULT_LABELS;
    }

    const {operationTree, iterationRunTimeMs} = breakdown;
    const leftoverMs = Math.max(0., iterationRunTimeMs - operationTree.runTimeMs);

    return DEFAULT_LABELS.map(({label, ...rest}) => ({
      ...rest,
      label,
      percentage: label === RunTimeEntryLabel.ForwardBackward
        ? toPercentage(operationTree.runTimeMs, iterationRunTimeMs)
        : toPercentage(leftoverMs, iterationRunTimeMs),
    }));
  }

  _renderPerfBars(expanded) {
    const {editorsByPath, projectRoot, breakdown} = this.props;
    if (breakdown == null) {
      return null;
    }

    const {operationTree, iterationRunTimeMs} = breakdown;
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

    const leftoverMs = Math.max(0., iterationRunTimeMs - operationTree.runTimeMs);
    if (leftoverMs === 0.) {
      return results;
    }

    // Append the extra "Other" category
    // Note we create a "fake" operationNode here
    const otherPct = toPercentage(leftoverMs, iterationRunTimeMs);
    results.push(
      <RunTimePerfBar
        key={'other'}
        operationNode={new OperationNode({
          id: -1,
          name: 'Other',
          forwardMs: leftoverMs,
          backwardMs: 0.,
          sizeBytes: 0,
          contexts: [],
        })}
        projectRoot={projectRoot}
        overallPct={otherPct}
        resizable={false}
        percentage={otherPct}
        colorClass={COLORS_BY_LABEL[RunTimeEntryLabel.Other][0]}
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
  breakdown: state.breakdown.model,
  ...ownProps,
});

export default connect(mapStateToProps)(RunTimeBreakdown);
