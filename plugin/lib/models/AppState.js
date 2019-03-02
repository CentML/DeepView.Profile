'use babel';

export default {
  // When the plugin is activated by Atom but the user has not "opened" it
  // by running the PerfVis:Open command.
  ACTIVATED: 'activated',

  // When the plugin has been opened by the user but they have not clicked
  // the "Get Started" button.
  OPENED: 'opened',

  // When the plugin is performing set up: connecting to the server, binding
  // to the text buffer (and creating one if needed), etc.
  CONNECTING: 'connecting',

  // When the plugin is operating normally.
  CONNECTED: 'connected',
};
