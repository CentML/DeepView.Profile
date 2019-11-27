'use babel';

import React from 'react';

class GetStarted extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      entrypoint: '',
      root: '',
    };

    this._handleInputChange = this._handleInputChange.bind(this);
  }

  _handleInputChange(event) {
    const target = event.target;
    const {name, value} = target;
    this.setState({
      [name]: value
    });
  }

  _isFormSubmittable() {
    return this.state.entrypoint != null &&
      this.state.root != null &&
      this.state.entrypoint.length > 0 &&
      this.state.root.length > 0;
  }

  render() {
    return (
      <div className="innpv-get-started">
        <div className="innpv-get-started-contents">
          <h1>innpv</h1>
          <p>Visualize the training performance of your PyTorch deep neural networks.</p>
          <div className="innpv-get-started-form">
            <div className="innpv-get-started-form-field">
              <p>Project Entrypoint:</p>
              <input
                className="input-text native-key-bindings"
                type="text"
                name="entrypoint"
                value={this.state.entrypoint}
                onChange={this._handleInputChange}
              />
            </div>
            <div className="innpv-get-started-form-field">
              <p>Project Root:</p>
              <input
                className="input-text native-key-bindings"
                type="text"
                name="root"
                value={this.state.root}
                onChange={this._handleInputChange}
              />
            </div>
          </div>
          <button
            className="btn btn-primary inline-block-tight icon icon-playback-play"
            onClick={this.props.handleClick}
            disabled={!this._isFormSubmittable()}
          >
            Get Started
          </button>
        </div>
      </div>
    );
  }
}

GetStarted.defaultProps = {
  onClick: () => {},
};

export default GetStarted;
