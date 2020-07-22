'use babel';

import React from 'react';
import path from 'path';

import AppState from '../models/AppState';

class GetStarted extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      optionsVisible: false,
      host: props.initialHost || 'localhost',
      port: props.initialPort || 60120,
      projectRoot: props.initialProjectRoot || '',
    };

    this._handleConnectClick = this._handleConnectClick.bind(this);
    this._handleInputChange = this._handleInputChange.bind(this);
    this._handleOptionsClick = this._handleOptionsClick.bind(this);
    this._handleSyncProjectRootClick = this._handleSyncProjectRootClick.bind(this);
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
    const {projectRoot} = this.state;
    return this.props.appState === AppState.OPENED &&
      this.state.host != null &&
      this.state.port != null &&
      this.state.host.length > 0 &&
      this._isPortValid(this.state.port) &&
      (projectRoot === '' || path.isAbsolute(projectRoot));
  }

  _handleOptionsClick() {
    this.setState({optionsVisible: !this.state.optionsVisible});
  }

  _handleConnectClick() {
    let canonicalPath = null;
    if (this.state.projectRoot != null && this.state.projectRoot !== '') {
      const projectRoot = path.normalize(this.state.projectRoot);
      canonicalPath = (projectRoot.endsWith(path.sep) && projectRoot.length > 1)
        ? projectRoot.slice(0, -1)
        : projectRoot;
    }
    const portAsInt = parseInt(this.state.port, 10);

    this.props.handleClick({
      host: this.state.host,
      port: portAsInt,
      projectRoot: canonicalPath,
    });
  }

  _handleSyncProjectRootClick() {
    // Guess the project root using the currently open file, if any
    const editor = atom.workspace.getActiveTextEditor();
    if (editor == null) {
      return;
    }

    const editorFilePath = editor.getPath();
    if (editorFilePath == null) {
      // The editor is showing a new unsaved file
      return;
    }

    this.setState({
      projectRoot: path.dirname(editorFilePath),
    });
  }

  _renderOptions() {
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
        <div className="innpv-get-started-project-root">
          <p>Local Absolute Project Root (only if profiling remotely):</p>
          <div className="innpv-get-started-project-root-row">
            <input
              className="input-text native-key-bindings"
              type="text"
              name="projectRoot"
              placeholder="e.g., /my/project/root (on your local machine)"
              value={this.state.projectRoot}
              onChange={this._handleInputChange}
            />
            <button
              className="btn inline-block-tight icon icon-sync"
              onClick={this._handleSyncProjectRootClick}
            />
          </div>
        </div>
      </div>
    );
  }

  _renderErrorMessage() {
    const {errorMessage} = this.props;
    if (errorMessage.length === 0) {
      return null;
    }
    return <p className="text-error">{errorMessage}</p>;
  }

  render() {
    return (
      <div className="innpv-get-started">
        <div className="innpv-get-started-text">
          <h1>Skyline</h1>
          <p>
            To get started, launch a Skyline session inside your PyTorch project
            and then hit Connect below.
          </p>
        </div>
        {this._renderErrorMessage()}
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
        {this.state.optionsVisible ? this._renderOptions() : null}
      </div>
    );
  }
}

GetStarted.defaultProps = {
  onClick: () => {},
};

export default GetStarted;
