'use babel';

import React from 'react';
import ReactDOM from 'react-dom';

import InlineHighlight from './generic/InlineHighlight';

const LINE_NUMBER_DECORATION = {
  type: 'line-number',
  class: 'innpv-contexthighlight-linenum',
};

export default class ContextHighlight extends React.Component {
  constructor(props) {
    super(props);
    this._element = document.createElement('div');
    this._overlayDecorationOption = {
      type: 'overlay',
      item: this._element,
      class: 'innpv-contexthighlight-overlay',
    };
  }

  render() {
    const {Fragment} = React;
    const {editor, lineNumber, ...rest} = this.props;
    return (
      <Fragment>
        <InlineHighlight
          editor={editor}
          decorations={[
            LINE_NUMBER_DECORATION,
            this._overlayDecorationOption,
          ]}
          lineNumber={lineNumber}
          column={999 /* HACK: Atom places this at the "end" of the line. */}
          show={true}
        />
        {ReactDOM.createPortal(
          <ContextMarker {...rest} />,
          this._element,
        )}
      </Fragment>
    );
  }
};

class ContextMarker extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      showDisplay: false,
    };
    this._onMouseEnter = this._onMouseEnter.bind(this);
    this._onMouseLeave = this._onMouseLeave.bind(this);
  }

  _onMouseEnter() {
    this.setState({showDisplay: true});
  }

  _onMouseLeave() {
    this.setState({showDisplay: false});
  }

  _renderDisplay() {
    const {runTimePct, memoryPct, invocations} = this.props;
    return (
      <div className="innpv-contextmarker-displaywrap">
        <div className="innpv-contextmarker-display">
          <div className="innpv-contextmarker-displaypointer" />
          <div className="innpv-contextmarker-displaycontent">
            <ContextPerfView runTimePct={runTimePct} memoryPct={memoryPct} />
            <div className="innpv-contextmarker-displayinfo">
              <strong>Invocations:</strong> {invocations}
            </div>
          </div>
        </div>
      </div>
    );
  }

  render() {
    const {showDisplay} = this.state;
    return (
      <div className="innpv-contextmarker-wrap">
        <div
          className="innpv-contextmarker"
          onMouseEnter={this._onMouseEnter}
          onMouseLeave={this._onMouseLeave}
        />
        {showDisplay ? this._renderDisplay() : null}
      </div>
    );
  }
}

function ContextBar(props) {
  const {label, percentage} = props;
  return (
    <div className="innpv-contextmarker-barwrap">
      <div className="innpv-contextmarker-barlabel">{label}</div>
      <div className="innpv-contextmarker-bararea">
        <div
          className="innpv-contextmarker-bar"
          style={{width: `${percentage}%`}}
        />
        <div className="innpv-contextmarker-barnum">
          {percentage.toFixed(1) + '%'}
        </div>
      </div>
    </div>
  );
}

function ContextPerfView(props) {
  const {runTimePct, memoryPct} = props;
  return (
    <div className="innpv-contextmarker-perfview">
      <ContextBar label="Run Time" percentage={runTimePct} />
      {memoryPct > 0.
        ? <ContextBar label="Memory" percentage={memoryPct} />
        : null}
    </div>
  );
}
