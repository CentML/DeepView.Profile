import React from 'react';
import clsx from 'clsx';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './styles.module.css';
import VideoOverlay from './VideoOverlay';

import CodeBlock from '@theme/CodeBlock';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

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

const bibtexCitation = `@inproceedings{skyline-yu20,
  title = {{Skyline: Interactive In-editor Computational Performance Profiling
    for Deep Neural Network Training}},
  author = {Yu, Geoffrey X. and Grossman, Tovi and Pekhimenko, Gennady},
  booktitle = {{Proceedings of the 33rd ACM Symposium on User Interface
    Software and Technology (UIST'20)}},
  year = {2020},
}`;

const textCitation = `Geoffrey X. Yu, Tovi Grossman, Gennady Pekhimenko.
Skyline: Interactive In-editor Computational Performance Profiling for
Deep Neural Network Training. In Proceedings of the 33rd ACM Symposium
on User Interface Software and Technology (UIST'20). 2020.`;

function ViewPreprint() {
  return (
    <a
      className="button button--primary detailsButton"
      href={useBaseUrl('pdf/skyline-uist20-preprint.pdf')}
      target="_blank"
    >
      View Preprint
    </a>
  );
}

function CitationBlock() {
  return (
    <div className={styles.citationBlock}>
      <Tabs
        defaultValue="bibtex"
        values={[
          { label: 'BibTeX', value: 'bibtex', },
          { label: 'Plain Text', value: 'txt', },
        ]}
      >
        <TabItem value="bibtex">
          <CodeBlock language="tex">{bibtexCitation}</CodeBlock>
        </TabItem>
        <TabItem value="txt">
          <CodeBlock language="txt">{textCitation}</CodeBlock>
        </TabItem>
      </Tabs>
    </div>
  );
}

export default PaperDetails;
