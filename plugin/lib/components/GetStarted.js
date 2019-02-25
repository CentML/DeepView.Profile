'use babel';

import React from 'react';

export default class GetStarted extends React.Component {
  render() {
    return (
      <div className="innpv-get-started">
        <h1>innpv</h1>
        <p>Visualize the training performance of your PyTorch deep neural networks.</p>
        <button
          className="btn btn-primary inline-block-tight icon icon-playback-play"
          onClick={this.props.handleClick}
        >
          Get Started
        </button>
      </div>
    );
  }
}

GetStarted.prototype.defaultProps = {
  onClick: () => {},
};
