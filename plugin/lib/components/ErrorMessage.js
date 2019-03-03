'use babel';

import React from 'react';

export default class ErrorMessage extends React.Component {
  render() {
    return (
      <div className="innpv-error">
        <div className="innpv-error-inner">
          <h1>Analysis Error</h1>
          <p>{this.props.message}</p>
        </div>
      </div>
    );
  }
}
