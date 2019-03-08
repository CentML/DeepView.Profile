'use babel';

import React from 'react';

const DRAG_MAX_PCT = 0.2;
const GAIN = 0.5;

function easing(x) {
  // y = 1 - (1 - x)^3
  const inner = 1 - x;
  const cubed = Math.pow(inner, 3);
  return 1 - cubed;
}

class Elastic extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      paddingTop: 0,
      paddingBottom: 0,
      height: 0,
    };
    this._dragging = false;
    this._clickedUpper = false;
    this._clickClientY = 0;

    this._handleMouseDown = this._handleMouseDown.bind(this);
    this._handleMouseUp = this._handleMouseUp.bind(this);
    this._handleMouseMove = this._handleMouseMove.bind(this);
    this._handleMouseLeave = this._handleMouseLeave.bind(this);
  }

  _handleMouseDown(event) {
    this._dragging = true;
    const boundingRect = event.currentTarget.getBoundingClientRect();
    const middle = boundingRect.height / 2;
    const clickPositionY = event.clientY - boundingRect.top;
    this._clickedUpper = clickPositionY < middle;
    this._clickClientY = event.clientY;
    this._targetHeight = boundingRect.height;
    this.setState({height: this._targetHeight});
  }

  _handleMouseUp(event) {
    this._clearDragging();
  }

  _handleMouseLeave(event) {
    this._clearDragging();
  }

  _clearDragging() {
    if (!this._dragging) {
      return;
    }
    this._dragging = false;
    this.setState({
      paddingTop: 0,
      paddingBottom: 0,
      height: 0,
    });
    this.props.updateMarginTop(0);
    this.props.handleSnapBack();
  }

  _handleMouseMove(event) {
    if (!this._dragging) {
      return;
    }
    // Positive means cursor moved down, negative means cursor moved up
    const deltaY = (event.clientY - this._clickClientY) * GAIN;
    const deltaYPct = Math.abs(deltaY / this._targetHeight);

    // Don't allow the element to shrink/grow beyond a certain threshold
    if (deltaYPct > DRAG_MAX_PCT) {
      return;
    }

    const easedAmount = easing(deltaYPct / DRAG_MAX_PCT) * this._targetHeight * DRAG_MAX_PCT;

    if (deltaY > 0) {
      // Cursor moved down
      if (this._clickedUpper) {
        // Shrink from the top
        this.setState({
          paddingTop: easedAmount,
        });
        this.props.handleShrink();
      } else {
        // Grow from the bottom
        this.setState({
          height: this._targetHeight + easedAmount,
        });
        this.props.handleGrow();
      }

    } else {
      // Cursor moved up
      if (this._clickedUpper) {
        // Grow from the top
        this.setState({
          height: this._targetHeight + easedAmount,
        });
        this.props.updateMarginTop(easedAmount);
        this.props.handleGrow();
      } else {
        // Shrink from the bottom
        this.setState({
          paddingBottom: easedAmount,
        });
        this.props.handleShrink();
      }
    }
  }

  render() {
    const {heightPct, className} = this.props;
    const {height, paddingTop, paddingBottom} = this.state;
    // Use the initial height unless the user has dragged the element
    const containerStyle = {
      height: height <= 0 ? `${heightPct}%` : `${height}px`,
    };
    const innerStyle = {
      paddingTop: `${paddingTop}px`,
      paddingBottom: `${paddingBottom}px`,
    };

    return (
      <div
        className={`innpv-elastic ${className}`}
        onMouseDown={this._handleMouseDown}
        onMouseUp={this._handleMouseUp}
        onMouseMove={this._handleMouseMove}
        onMouseLeave={this._handleMouseLeave}
        style={containerStyle}
      >
        <div className="innpv-elastic-inner" style={innerStyle}>
          {this.props.children}
        </div>
      </div>
    );
  }
}

Elastic.defaultProps = {
  className: '',
  handleShrink: () => {},
  handleGrow: () => {},
  handleSnapBack: () => {},
  updateMarginTop: () => {},
};

export default Elastic;
