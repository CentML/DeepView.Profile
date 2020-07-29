import React from 'react';
import ReactDOM from 'react-dom';
import styles from './styles.module.css';

class VideoOverlay extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      opacity: 0,
    };
  }

  componentDidMount() {
    this.setState({opacity: 1});
  }

  _computeDimensions() {
    const width = document.body.clientWidth * 0.8;
    const height = width / 16 * 9;
    return {width: Math.floor(width), height: Math.floor(height)};
  }

  _renderBody() {
    const {width, height} = this._computeDimensions();
    return (
      <div
        className={styles.videoOverlay}
        onClick={this.props.onCloseClick}
        style={{opacity: this.state.opacity}}
      >
        <div
          className="shadow--tl"
          style={{height: `${height}px`, width: `${width}px`, backgroundColor: '#000000'}}
        >
          <iframe
            width={width}
            height={height}
            src="https://www.youtube-nocookie.com/embed/qNlIH98vCgY?autoplay=1&modestbranding=1"
            frameBorder="0"
            allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        </div>
      </div>
    );
  }

  render() {
    return ReactDOM.createPortal(
      this._renderBody(),
      document.body,
    );
  }
}

VideoOverlay.defaultProps = {
  onCloseClick: () => {},
};

export default VideoOverlay;
