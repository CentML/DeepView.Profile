'use babel';

import React from 'react';

import Subheader from './Subheader';
import BarSlider from './BarSlider';
import NumericDisplay from './NumericDisplay';
import BatchSizeStore from '../stores/batchsize_store';
import INNPVStore from '../stores/innpv_store';
import PerfVisState from '../models/PerfVisState';
import {toPercentage} from '../utils';

export default class Throughput extends React.Component {
  constructor(props) {
    super(props);
    this._handleResize = this._handleResize.bind(this);
  }

  _handleResize(deltaPct, basePct) {
    // TODO: Add in throughput predictions again
    return;

    BatchSizeStore.updateThroughput(deltaPct, basePct);
    INNPVStore.setPerfVisState(PerfVisState.SHOWING_PREDICTIONS);
  }

  _getPercentage() {
    const {model} = this.props;
    if (!model.hasMaxThroughputPrediction) {
      return 100;
    }

    return toPercentage(
      model.samplesPerSecond,
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

  render() {
    const {model, handleSliderHoverEnter, handleSliderHoverExit} = this.props;
    const notReady = model == null;

    return (
      <div className="innpv-throughput innpv-subpanel">
        <Subheader icon="flame">Training Throughput</Subheader>
        <div className="innpv-subpanel-content">
          <BarSlider
            percentage={notReady ? 0 : this._getPercentage()}
            handleResize={this._handleResize}
            onMouseEnter={handleSliderHoverEnter}
            onMouseLeave={handleSliderHoverExit}
          />
          <div className="innpv-subpanel-sidecontent">
            <NumericDisplay
              top="Throughput"
              number={notReady ? '---' : model.samplesPerSecond}
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
