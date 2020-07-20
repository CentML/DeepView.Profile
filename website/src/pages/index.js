import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useThemeContext from '@theme/hooks/useThemeContext';
import styles from './styles.module.css';

const features = [
  {
    title: <>🏖 In-editor Profiling</>,
    description: (
      <>
        Profile your models from the comfort of your text editor! Skyline works
        as a plugin for <a href="https://atom.io" target="_blank">Atom</a>. It
        displays overviews and breakdowns of key performance metrics such as
        training throughput and memory usage.
      </>
    ),
  },
  {
    title: <>🔮 Interactive Visualizations</>,
    description: (
      <>
        Skyline's performance visualizations are interactive! Hovering over a
        visualization will reveal its associated line(s) of code. When dragging
        certain visualizations, Skyline can also make predictions about the
        throughput and memory usage of different batch sizes.
      </>
    ),
  },
  {
    title: <>🎈 Profile While You Develop</>,
    description: (
      <>
        No more performance surprises! Skyline transparently profiles your
        model in the background as you make changes during development.  You'll
        be the first to know when changes to your model affect its computational
        performance.
      </>
    ),
  },
];

function Feature({imageUrl, title, description}) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={clsx('col col--4', styles.feature)}>
      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function SkylineHeader() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  const {isDarkTheme} = useThemeContext();
  return (
    <header className="hero">
      <div className="container">
        <div className={clsx('row', styles.heroBannerRow)}>
          <div className={clsx('col col--6', styles.heroText)}>
            <img alt="Skyline" src="img/skyline_wordmark.svg" />
            <p className="hero__subtitle">{siteConfig.tagline}</p>
            <Link
              className={clsx(
                'button button--primary button--lg',
                styles.getStarted,
              )}
              to={useBaseUrl('docs/')}>
              Get Started
            </Link>
            <Link
              className={clsx(
                'button button--secondary button--lg',
                styles.getStarted,
              )}
              to={useBaseUrl('docs/')}>
              Read the Docs
            </Link>
          </div>
          <div className="col col--6">
            <img
              className={clsx(styles.heroBannerImage, 'shadow--tl')}
              alt="A screenshot of Skyline's user interface, running in the Atom text editor."
              src={`img/skyline_${isDarkTheme ? 'dark' : 'light'}.png`}
            />
          </div>
        </div>
      </div>
    </header>
  );
}

function Home() {
  const context = useDocusaurusContext();
  const {siteConfig = {}} = context;
  return (
    <Layout
      title="Interactive in-editor performance profiling for PyTorch"
      description={siteConfig.tagline}
    >
      <SkylineHeader />
      <main>
        {features && features.length > 0 && (
          <section className={styles.features}>
            <div className="container">
              <div className="row">
                {features.map((props, idx) => (
                  <Feature key={idx} {...props} />
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </Layout>
  );
}

export default Home;
