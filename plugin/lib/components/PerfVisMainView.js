'use babel';

import React from 'react';

import PerfBarContainer from './PerfBarContainer';

export default class PerfVisMainView extends React.Component {
  _renderHeader() {
    return (
      <div className="innpv-header">
        <span className="icon icon-graph"></span>innpv
      </div>
    );
  }

  render() {
    return (
      <div className="innpv-main">
        {this._renderHeader()}
        <div className="innpv-contents">
          <PerfBarContainer
            operationInfos={this.props.operationInfos}
            editor={this.props.editor}
          />
        </div>
      </div>
    );
  }
}
