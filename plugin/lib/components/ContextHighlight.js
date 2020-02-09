'use babel';

import React from 'react';
import ReactDOM from 'react-dom';

import InlineHighlight from './generic/InlineHighlight';

export default class ContextHighlight extends React.Component {
  constructor(props) {
    super(props);
    this._gutterElement = document.createElement('div');
    this._overlayElement = document.createElement('div');
    this._decorationOptions = [
      {
        type: 'overlay',
        item: this._overlayElement,
        class: 'innpv-contexthighlight-overlay',
      },
      {
        type: 'gutter',
        item: this._gutterElement,
      },
    ];
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

  render() {
    const {Fragment} = React;
    const {editor, lineNumber, ...rest} = this.props;
    return (
      <Fragment>
        <InlineHighlight
          editor={editor}
          decorations={this._decorationOptions}
          lineNumber={lineNumber}
          column={999 /* HACK: Atom places this at the "end" of the line. */}
          show={true}
        />
        {this.state.showDisplay
          ? ReactDOM.createPortal(
              <ContextDisplay {...rest} />,
              this._overlayElement,
            )
          : null}
        {ReactDOM.createPortal(
          <div
            className="innpv-contexthighlight-guttermarker"
            onMouseEnter={this._onMouseEnter}
            onMouseLeave={this._onMouseLeave}
          />,
          this._gutterElement,
        )}
      </Fragment>
    );
  }
};

function ContextDisplay(props) {
  const {runTimePct, memoryPct, invocations} = props;
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
