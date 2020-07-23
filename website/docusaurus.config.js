const mainUrl = 'https://skylineprof.github.io/';
const githubUrl = 'https://github.com/skylineprof/skyline/';

module.exports = {
  title: 'Skyline',
  tagline: 'Interactive in-editor computational performance profiling, visualization, and debugging for PyTorch deep neural networks.',
  url: mainUrl,
  baseUrl: '/',
  favicon: 'img/skyline64x64.png',
  organizationName: 'skylineprof',
  projectName: 'skyline',
  themeConfig: {
    navbar: {
      title: 'Skyline',
      image: 'img/skyline_social.png',
      logo: {
        alt: 'Skyline Logo',
        src: 'img/skyline.svg',
      },
      links: [
        {
          to: 'docs/',
          activeBasePath: 'docs',
          label: 'Docs',
          position: 'left',
        },
        {
          href: githubUrl,
          label: 'GitHub',
          position: 'left',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'About Skyline',
              to: 'docs/',
            },
            {
              label: 'Getting Started',
              to: 'docs/getting-started/',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Skyline on GitHub',
              href: githubUrl,
            },
            {
              label: 'EcoSystem Research Group',
              href: 'https://www.cs.toronto.edu/ecosystem/',
            },
          ],
        },
      ],
    },
    gtag: {
      trackingID: 'UA-156567771-2',
      anonymizeIP: true,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          // It is recommended to set document id as docs home page (`docs/` path).
          homePageId: 'intro',
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl: githubUrl + '/edit/master/website/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
