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
        <ViewPaper />
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

function ViewPaper() {
  return (
    <a
      className="button button--primary detailsButton"
      href="https://dl.acm.org/doi/10.1145/3379337.3415890?cid=99659587236"
      target="_blank"
    >
      View Paper
    </a>
  );
}

export default PaperDetails;
