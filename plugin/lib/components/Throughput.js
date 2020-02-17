'use babel';

import React from 'react';
import {connect} from 'react-redux';

import Subheader from './Subheader';
import BarSlider from './BarSlider';
import NumericDisplay from './NumericDisplay';
import AnalysisActions from '../redux/actions/analysis';
import {toPercentage} from '../utils';

class Throughput extends React.Component {
  constructor(props) {
    super(props);
    this._handleResize = this._handleResize.bind(this);
  }

  _handleResize(deltaPct, basePct) {
    const {runTimeMsModel, dispatch} = this.props;
    if (runTimeMsModel == null) {
      // We won't always have a run time model available. If that
      // happens, we do not allow the user to manipulate this view.
      return;
    }
    dispatch(AnalysisActions.dragThroughput({deltaPct, basePct}));
  }

  _getPercentage(samplesPerSecond) {
    const {model} = this.props;
    if (!model.hasMaxThroughputPrediction) {
      return 100;
    }

    return toPercentage(
      samplesPerSecond,
      model.predictedMaxSamplesPerSecond,
    );
  }

  _getPredictedMaximum() {
    const {model} = this.props;
    if (!model.hasMaxThroughputPrediction) {
      return "N/A";
    }
    return model.predictedMaxSamplesPerSecond;
  }

  _getSamplesPerSecond() {
    if (this.props.model == null) {
      return null;
    }

    const {currentBatchSize, runTimeMsModel} = this.props;
    if (currentBatchSize != null && runTimeMsModel != null) {
      // This is a predicted throughput
      return currentBatchSize /
        runTimeMsModel.evaluate(currentBatchSize) * 1000;

    } else {
      return this.props.model.samplesPerSecond;
    }
  }

  render() {
    const {model, handleSliderHoverEnter, handleSliderHoverExit} = this.props;
    const notReady = model == null;
    const samplesPerSecond = this._getSamplesPerSecond();

    return (
      <div className="innpv-throughput innpv-subpanel">
        <Subheader icon="flame">Training Throughput</Subheader>
        <div className="innpv-subpanel-content">
          <BarSlider
            percentage={notReady ? 0 : this._getPercentage(samplesPerSecond)}
            handleResize={this._handleResize}
            onMouseEnter={handleSliderHoverEnter}
            onMouseLeave={handleSliderHoverExit}
          />
          <div className="innpv-subpanel-sidecontent">
            <NumericDisplay
              top="Throughput"
              number={notReady ? '---' : samplesPerSecond}
              bottom="samples/second"
            />
            <div className="innpv-separator" />
            <NumericDisplay
              top="Predicted Maximum"
              number={notReady ? '---' : this._getPredictedMaximum()}
              bottom="samples/second"
            />
          </div>
        </div>
      </div>
    );
  }
}

const mapStateToProps = (state, ownProps) => ({
  model: state.throughput,
  runTimeMsModel: state.predictionModels.runTimeMs,
  currentBatchSize: state.predictionModels.currentBatchSize,
  ...ownProps,
});

export default connect(mapStateToProps)(Throughput);
