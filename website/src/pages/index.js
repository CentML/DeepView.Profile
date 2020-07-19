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
    title: <>Easy to Use</>,
    description: (
      <>
        Docusaurus was designed from the ground up to be easily installed and
        used to get your website up and running quickly.
      </>
    ),
  },
  {
    title: <>Focus on What Matters</>,
    description: (
      <>
        Docusaurus lets you focus on your docs, and we&apos;ll do the chores. Go
        ahead and move your docs into the <code>docs</code> directory.
      </>
    ),
  },
  {
    title: <>Powered by React</>,
    description: (
      <>
        Extend or customize your website layout by reusing React. Docusaurus can
        be extended while reusing the same header and footer.
      </>
    ),
  },
];

function Feature({imageUrl, title, description}) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={clsx('col col--4', styles.feature)}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
        </div>
      )}
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
