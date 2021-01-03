import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './styles.module.css';
import PaperDetails from '../components/PaperDetails';

const UofT = {name: 'University of Toronto', url: 'https://web.cs.toronto.edu'};
const Vector = {name: 'Vector Institute', url: 'https://vectorinstitute.ai'};

function FrontMatter() {
  return (
    <div className="container">
      <div className={styles.paperTitle}>
        <h1>
          Skyline: Interactive In-Editor Computational Performance Profiling
          for Deep Neural Network Training
        </h1>
        <p>
          <a className="modest-link" href="https://uist.acm.org/uist2020/" target="_blank">
            Proceedings of the 33rd ACM Symposium on User Interface Software and Technology (UIST'20)
          </a>
        </p>
      </div>
      <AuthorsList />
    </div>
  );
}

function Abstract() {
  return (
    <div className={clsx('container', styles.paperAbstract)}>
      <strong>Abstract</strong>
      <p>
        Training a state-of-the-art deep neural network (DNN) is a
        computationally-expensive and time-consuming process, which
        incentivizes deep learning developers to debug their DNNs for
        computational performance. However, effectively performing this
        debugging requires intimate knowledge about the underlying software and
        hardware systems—something that the typical deep learning developer may
        not have. To help bridge this gap, we present Skyline: a new
        interactive tool for DNN training that supports in-editor computational
        performance profiling, visualization, and debugging. Skyline's key
        contribution is that it leverages special computational properties of
        DNN training to provide (i) interactive performance predictions and
        visualizations, and (ii) directly manipulatable visualizations that,
        when dragged, mutate the batch size in the code. As an in-editor tool,
        Skyline allows users to leverage these diagnostic features to debug
        the performance of their DNNs during development. An exploratory
        qualitative user study of Skyline produced promising results; all the
        participants found Skyline to be useful and easy to use.
      </p>
    </div>
  );
}

function Author(props) {
  return (
    <div className={clsx('avatar', styles.paperAuthor)}>
      <div className="avatar__intro">
        <h4 className="avatar__name">
          <a className="modest-link" href={props.website} target="_blank">{props.name}</a>
        </h4>
        <small className="avatar__subtitle">{
          props.affiliations.map(({name, url}, idx) => {
            const link = (<a className="modest-link" href={url} target="_blank">{name}</a>);
            if (idx == props.affiliations.length - 1) {
              return <span key={url}>{link}</span>;
            } else {
              return <span key={url}>{link}, </span>;
            }
          })
        }</small>
      </div>
    </div>
  );
}

function AuthorsList() {
  return (
    <div className={clsx('row', styles.authorsList)}>
      <div className="col col--4">
        <Author
          name="Geoffrey X. Yu"
          affiliations={[UofT, Vector]}
          website="https://www.geoffreyyu.com"
        />
      </div>
      <div className="col col--4">
        <Author
          name="Tovi Grossman"
          affiliations={[UofT]}
          website="https://www.tovigrossman.com"
        />
      </div>
      <div className="col col--4">
        <Author
          name="Gennady Pekhimenko"
          affiliations={[UofT, Vector]}
          website="https://www.cs.toronto.edu/~pekhimenko"
        />
      </div>
    </div>
  );
}

function PaperPage() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  return (
    <Layout
      title="Research Paper"
      description="Details about the Skyline research paper, published at UIST'20."
    >
      <main>
        <FrontMatter />
        <Abstract />
        <PaperDetails />
      </main>
    </Layout>
  );
}

export default PaperPage;
