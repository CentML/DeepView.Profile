'use babel';

import React from 'react';

class Resizable extends React.Component {
  constructor(props) {
    super(props);
    this._dragging = false;
    this._clickClientY = 0;

    this._handleMouseDown = this._handleMouseDown.bind(this);
    this._handleMouseUp = this._handleMouseUp.bind(this);
    this._handleMouseMove = this._handleMouseMove.bind(this);
    this._handleMouseLeave = this._handleMouseLeave.bind(this);
  }

  _handleMouseDown(event) {
    this._dragging = true;
    const {height} = event.currentTarget.getBoundingClientRect();
    this._clickClientY = event.clientY;
    this._sliderHeight = height / (this.props.heightPct / 100);
  }

  _handleMouseUp(event) {
    this._dragging = false;
  }

  _handleMouseLeave(event) {
    this._dragging = false;
  }

  _handleMouseMove(event) {
    if (!this._dragging) {
      return;
    }
    const diff = this._clickClientY - event.clientY;
    this.props.handleResize(diff / this._sliderHeight);
  }

  render() {
    return (
      <div
        className={`innpv-resizable ${this.props.className}`}
        style={{height: `${this.props.heightPct}%`}}
        onMouseDown={this._handleMouseDown}
        onMouseUp={this._handleMouseUp}
        onMouseMove={this._handleMouseMove}
        onMouseLeave={this._handleMouseLeave}
      >
        {this.props.children}
      </div>
    );
  }
}

Resizable.defaultProps = {
  className: '',
  handleResize: () => {},
};

export default Resizable;
