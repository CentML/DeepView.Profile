'use babel';

import React from 'react';

class GetStarted extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      optionsVisible: false,
      host: 'localhost',
      port: 60606,
    };

    this._handleConnectClick = this._handleConnectClick.bind(this);
    this._handleInputChange = this._handleInputChange.bind(this);
    this._handleOptionsClick = this._handleOptionsClick.bind(this);
  }

  _handleInputChange(event) {
    const target = event.target;
    const {name, value} = target;
    this.setState({
      [name]: value
    });
  }

  _isPortValid(port) {
    const portAsInt = parseInt(port, 10);
    return !isNaN(portAsInt) && portAsInt > 0 && portAsInt <= 65535;
  }

  _allowConnectClick() {
    return this.state.host != null &&
      this.state.port != null &&
      this.state.host.length > 0 &&
      this._isPortValid(this.state.port);
  }

  _handleOptionsClick() {
    this.setState({optionsVisible: !this.state.optionsVisible});
  }

  _handleConnectClick() {
    const portAsInt = parseInt(this.state.port, 10);
    this.props.handleClick({host: this.state.host, port: portAsInt});
  }

  renderOptions() {
    return (
      <div className="innpv-get-started-options">
        <div className="innpv-get-started-host-port-fields">
          <div className="innpv-get-started-host">
            <p>Host:</p>
            <input
              className="input-text native-key-bindings"
              type="text"
              name="host"
              value={this.state.host}
              onChange={this._handleInputChange}
            />
          </div>
          <div className="innpv-get-started-port">
            <p>Port:</p>
            <input
              className="input-text native-key-bindings"
              type="text"
              name="port"
              value={this.state.port}
              onChange={this._handleInputChange}
            />
          </div>
        </div>
      </div>
    );
  }

  render() {
    return (
      <div className="innpv-get-started">
        <div className="innpv-get-started-text">
          <h1>innpv</h1>
          <p>
            To get started, launch an INNPV server inside your PyTorch project
            and then hit Connect below.
          </p>
        </div>
        <div className="innpv-get-started-buttons">
          <button
            className="btn btn-primary inline-block-tight icon icon-playback-play"
            onClick={this._handleConnectClick}
            disabled={!this._allowConnectClick()}
          >
            Connect
          </button>
          <button
            className="btn inline-block-tight icon icon-gear"
            onClick={this._handleOptionsClick}
          />
        </div>
        {this.state.optionsVisible ? this.renderOptions() : null}
      </div>
    );
  }
}

GetStarted.defaultProps = {
  onClick: () => {},
};

export default GetStarted;
