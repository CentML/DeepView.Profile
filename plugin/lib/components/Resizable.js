'use babel';

import React from 'react';

const MOUSE_MOVE_EVENT = 'mousemove';
const MOUSE_UP_EVENT = 'mouseup';

class Resizable extends React.Component {
  constructor(props) {
    super(props);
    this._dragging = false;
    this._clickClientY = 0;

    this._handleMouseDown = this._handleMouseDown.bind(this);
    this._handleMouseUp = this._handleMouseUp.bind(this);
    this._handleMouseMove = this._handleMouseMove.bind(this);
  }

  componentDidMount() {
    document.addEventListener(MOUSE_MOVE_EVENT, this._handleMouseMove);
    document.addEventListener(MOUSE_UP_EVENT, this._handleMouseUp);
  }

  componentWillUnmount() {
    document.removeEventListener(MOUSE_MOVE_EVENT, this._handleMouseMove);
    document.removeEventListener(MOUSE_UP_EVENT, this._handleMouseUp);
  }

  _handleMouseDown(event) {
    this._dragging = true;
    const {height} = event.currentTarget.getBoundingClientRect();
    this._clickClientY = event.clientY;
    this._sliderHeight = height / (this.props.heightPct / 100);
    this._initialHeightPct = this.props.heightPct;
  }

  _handleMouseUp(event) {
    this._dragging = false;
  }

  _handleMouseMove(event) {
    if (!this._dragging) {
      return;
    }
    const diff = this._clickClientY - event.clientY;
    this.props.handleResize(diff / this._sliderHeight * 100, this._initialHeightPct);
  }

  render() {
    return (
      <div
        className={`innpv-resizable ${this.props.className}`}
        style={{height: `${this.props.heightPct}%`}}
        onMouseDown={this._handleMouseDown}
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
