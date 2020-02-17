'use babel';

import React from 'react';
import {connect} from 'react-redux';

import Subheader from './Subheader';
import BarSlider from './BarSlider';
import NumericDisplay from './NumericDisplay';
import AnalysisActions from '../redux/actions/analysis';
import {toPercentage} from '../utils';

export default class Memory extends React.Component {
  constructor(props) {
    super(props);
    this._handleResize = this._handleResize.bind(this);
  }

  _handleResize(deltaPct, basePct) {
    this.props.dispatch(AnalysisActions.dragMemory({deltaPct, basePct}));
  }

  _getCurrentPeakUsageBytes() {
    const {peakUsageBytes, peakUsageBytesModel, currentBatchSize} = this.props;
    if (peakUsageBytesModel != null && currentBatchSize != null) {
      // This is showing a prediction
      return peakUsageBytesModel.evaluate(currentBatchSize);

    } else {
      return peakUsageBytes;
    }
  }

  _toMb(bytes) {
    return bytes / 1024.0 / 1024.0;
  }

  render() {
    const {
      peakUsageBytes,
      memoryCapacityBytes,
      handleSliderHoverEnter,
      handleSliderHoverExit,
    } = this.props;
    const notReady = peakUsageBytes == null;
    const currentPeakUsageBytes = this._getCurrentPeakUsageBytes();
    const percentage = notReady
      ? 0
      : toPercentage(currentPeakUsageBytes, memoryCapacityBytes);

    return (
      <div className="innpv-memory innpv-subpanel">
        <Subheader icon="database">Peak Memory Usage</Subheader>
        <div className="innpv-subpanel-content">
          <BarSlider
            percentage={percentage}
            handleResize={this._handleResize}
            onMouseEnter={handleSliderHoverEnter}
            onMouseLeave={handleSliderHoverExit}
          />
          <div className="innpv-subpanel-sidecontent">
            <NumericDisplay
              top="Peak Usage"
              number={notReady ? '---' : this._toMb(currentPeakUsageBytes)}
              bottom="Megabytes"
            />
            <div className="innpv-separator" />
            <NumericDisplay
              top="Maximum Capacity"
              number={notReady ? '---' : this._toMb(memoryCapacityBytes)}
              bottom="Megabytes"
            />
          </div>
        </div>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => ({
  peakUsageBytes: state.peakUsageBytes,
  memoryCapacityBytes: state.memoryCapacityBytes,
  peakUsageBytesModel: state.predictionModels.peakUsageBytes,
  currentBatchSize: state.predictionModels.currentBatchSize,
  ...ownProps,
});

export default connect(mapStateToProps)(Memory);
