import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';
import VideoOverlay from './VideoOverlay';
import CitationBlock from './CitationBlock';

class PaperDetails extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      showCitation: false,
      showVideo: false,
    };
    this._showCitationClick = this._showCitationClick.bind(this);
    this._toggleVideo = this._toggleVideo.bind(this);
  }

  _showCitationClick() {
    this.setState({showCitation: !this.state.showCitation});
  }

  _toggleVideo() {
    this.setState({showVideo: !this.state.showVideo});
  }

  render() {
    const {showCitation, showVideo} = this.state;
    return (
      <div className={clsx('container', styles.paperDetails)}>
        <ViewPreprint />
        <button
          onClick={this._toggleVideo}
          className="button button--secondary detailsButton"
        >
          Watch the Video
        </button>
        <button
          onClick={this._showCitationClick}
          className={clsx(
            'button button--secondary detailsButton',
            showCitation && 'button--active',
          )}
        >
          {showCitation ? 'Hide Citation' : 'Show Citation'}
        </button>
        {showCitation ? <CitationBlock /> : null}
        {showVideo ? <VideoOverlay onCloseClick={this._toggleVideo} /> : null}
      </div>
    );
  }
}

function ViewPreprint() {
  return (
    <a
      className="button button--primary detailsButton"
      href="https://arxiv.org/pdf/2008.06798.pdf"
      target="_blank"
    >
      View Preprint
    </a>
  );
}

export default PaperDetails;
