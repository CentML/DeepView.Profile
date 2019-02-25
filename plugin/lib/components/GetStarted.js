'use babel';

import React from 'react';

export default class GetStarted extends React.Component {
  constructor(props) {
    super(props);
    this._onClick = this._onClick.bind(this);
  }

  _onClick(event) {
    console.log('Get started button clicked.');
  }

  render() {
    return (
      <div className="innpv-get-started">
        <h1>innpv</h1>
        <p>Visualize the training performance of your PyTorch deep neural networks.</p>
        <button
          className="btn btn-primary inline-block-tight icon icon-playback-play"
          onClick={this._onClick}
        >
          Get Started
        </button>
      </div>
    );
  }
}
