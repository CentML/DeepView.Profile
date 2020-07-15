const mainUrl = '';
const githubUrl = 'https://github.com/geoffxy/skyline';

module.exports = {
  title: 'Skyline',
  tagline: 'Interactive in-editor performance profiling, visualization, and debugging for PyTorch neural networks.',
  url: mainUrl,
  baseUrl: '/',
  favicon: 'img/skyline64x64.png',
  organizationName: 'facebook', // Usually your GitHub org/user name.
  projectName: 'docusaurus', // Usually your repo name.
  themeConfig: {
    navbar: {
      title: 'Skyline',
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
          href: '#',
          label: 'Code Examples',
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
              label: 'About',
              to: 'docs/',
            },
            {
              label: 'Getting Started',
              to: 'docs/doc2/',
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
              href: 'https://www.cs.toronto.edu/ecosystem',
            },
          ],
        },
      ],
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          // It is recommended to set document id as docs home page (`docs/` path).
          homePageId: 'doc1',
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            githubUrl + '/edit/master/website/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
